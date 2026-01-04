import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.predictor import TimesNetPredictor
from core.track_buffer import TrackWindowBuffer
from features.extractor import FeatureExtractor, FeatureNormalizer, FEATURE_COLS
from udp.parser import parse_packet
from udp.publisher import MulticastPublisher
from udp.receiver import MulticastReceiver


def parse_args():
    parser = argparse.ArgumentParser(description="组播航迹接收 + TimesNet 预测 + 组播发布")
    parser.add_argument("--in_group", required=True, help="输入组播地址")
    parser.add_argument("--in_port", type=int, required=True, help="输入端口")
    parser.add_argument("--in_iface", default="0.0.0.0", help="输入网卡IP")
    parser.add_argument("--bind_ip", default="", help="绑定IP，默认0.0.0.0")
    parser.add_argument("--timeout", type=float, default=2.0, help="接收超时秒数")
    parser.add_argument("--skip_checksum", action="store_true", help="跳过累加和校验")

    parser.add_argument("--out_group", required=True, help="输出组播地址")
    parser.add_argument("--out_port", type=int, required=True, help="输出端口")
    parser.add_argument("--out_iface", default="0.0.0.0", help="输出网卡IP")
    parser.add_argument("--ttl", type=int, default=1, help="组播TTL")

    parser.add_argument("--model_path", required=True, help="模型 checkpoint 路径")
    parser.add_argument("--model", default="TimesNet", help="模型类型")
    parser.add_argument("--num_classes", type=int, default=2, help="类别数")
    parser.add_argument("--seq_len", type=int, default=20, help="序列长度")
    parser.add_argument("--min_seq_len", type=int, default=20, help="最小序列长度")
    parser.add_argument("--stats_path", default="", help="归一化统计文件 stats.json")
    parser.add_argument("--device", default="auto", help="设备: auto|cpu|cuda")

    parser.add_argument("--label_len", type=int, default=48, help="label_len")
    parser.add_argument("--d_model", type=int, default=64, help="d_model")
    parser.add_argument("--d_ff", type=int, default=256, help="d_ff")
    parser.add_argument("--e_layers", type=int, default=2, help="e_layers")
    parser.add_argument("--top_k", type=int, default=2, help="top_k")
    parser.add_argument("--num_kernels", type=int, default=6, help="num_kernels")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--embed", type=str, default="timeF", help="embed")
    parser.add_argument("--freq", type=str, default="h", help="freq")

    parser.add_argument("--max_age_s", type=float, default=10.0, help="轨迹最大保留秒数")
    parser.add_argument("--publish_interval_ms", type=int, default=0, help="发布节流毫秒(0=不节流)")
    return parser.parse_args()


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main():
    args = parse_args()
    device = resolve_device(args.device)
    min_seq_len = max(1, min(args.min_seq_len, args.seq_len))

    normalizer = FeatureNormalizer.from_stats_file(args.stats_path, FEATURE_COLS) if args.stats_path else FeatureNormalizer()
    extractor = FeatureExtractor(normalizer=normalizer)
    buffer = TrackWindowBuffer(seq_len=args.seq_len, max_age_s=args.max_age_s)

    receiver = MulticastReceiver(
        group=args.in_group,
        port=args.in_port,
        iface=args.in_iface,
        bind_ip=args.bind_ip,
        timeout_s=args.timeout,
    ).open()
    publisher = MulticastPublisher(
        group=args.out_group,
        port=args.out_port,
        iface=args.out_iface,
        ttl=args.ttl,
    ).open()

    model_cfg = {
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "e_layers": args.e_layers,
        "top_k": args.top_k,
        "num_kernels": args.num_kernels,
        "dropout": args.dropout,
        "embed": args.embed,
        "freq": args.freq,
    }
    predictor = TimesNetPredictor(
        model_path=args.model_path,
        model_name=args.model,
        num_classes=args.num_classes,
        device=device,
        model_cfg=model_cfg,
    )
    predictor.load(num_features=len(FEATURE_COLS))

    last_publish_ts = 0.0
    print("开始接收组播并预测...")

    while True:
        try:
            data, addr = receiver.recv()
        except Exception:
            continue

        targets = parse_packet(data, skip_checksum=args.skip_checksum)
        if not targets:
            continue

        for tar in targets:
            feat = extractor.extract(tar)
            if feat is None:
                continue
            buffer.update(tar.get("track_id", 0), feat, timestamp_s=tar.get("timestamp"))

        buffer.cleanup()
        track_ids, batch, lengths = buffer.build_batch(min_seq_len=min_seq_len)
        if batch is None:
            continue

        now = time.time()
        if args.publish_interval_ms > 0:
            if (now - last_publish_ts) * 1000 < args.publish_interval_ms:
                continue

        preds, probs = predictor.predict(batch, lengths)
        items = []
        for i, tid in enumerate(track_ids):
            prob_uav = float(probs[i, 1]) if probs.shape[1] > 1 else float(probs[i, 0])
            prob_bird = float(probs[i, 0]) if probs.shape[1] > 1 else 1.0 - prob_uav
            items.append({
                "track_id": tid,
                "timestamp_ms": int((buffer.get_last_timestamp(tid) or 0.0) * 1000),
                "pred": int(preds[i]),
                "prob_uav": prob_uav,
                "prob_bird": prob_bird,
            })

        publisher.send(items)
        last_publish_ts = now


if __name__ == "__main__":
    main()
