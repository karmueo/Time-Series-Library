import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.predictor import TimesNetPredictor
from core.track_buffer import TrackWindowBuffer
from features.extractor import FeatureNormalizer
from features.feature_config import FEATURE_MAP, get_feature_cols
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
    parser.add_argument("--pred_len", type=int, default=0, help="pred_len")
    parser.add_argument("--d_model", type=int, default=64, help="d_model")
    parser.add_argument("--n_heads", type=int, default=8, help="n_heads")
    parser.add_argument("--d_ff", type=int, default=256, help="d_ff")
    parser.add_argument("--e_layers", type=int, default=2, help="e_layers")
    parser.add_argument("--d_layers", type=int, default=1, help="d_layers")
    parser.add_argument("--top_k", type=int, default=2, help="top_k")
    parser.add_argument("--num_kernels", type=int, default=6, help="num_kernels")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--embed", type=str, default="timeF", help="embed")
    parser.add_argument("--freq", type=str, default="s", help="freq")

    parser.add_argument("--max_age_s", type=float, default=10.0, help="轨迹最大保留秒数")
    parser.add_argument("--publish_interval_ms", type=int, default=0, help="发布节流毫秒(0=不节流)")
    parser.add_argument("--window_step", type=int, default=0, help="滑窗步长(0=每条都推理)")
    parser.add_argument("--print_targets", action="store_true", help="打印解析后的目标信息")
    parser.add_argument("--print_features", action="store_true", help="打印送入模型的特征值")
    parser.add_argument("--use_batch_ema", action="store_true", help="同批号目标使用EMA平滑概率并统一判别")
    parser.add_argument("--ema_alpha", type=float, default=0.6, help="EMA平滑系数(0-1)")
    return parser.parse_args()


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main():
    args = parse_args()
    device = resolve_device(args.device)
    min_seq_len = max(1, min(args.min_seq_len, args.seq_len))

    feature_cols = get_feature_cols()
    normalizer = FeatureNormalizer.from_stats_file(args.stats_path, feature_cols) if args.stats_path else FeatureNormalizer()
    batch_buffers = {}
    batch_last_seen = {}
    batch_last_publish_ts = {}
    batch_ema_prob_uav = {}

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
        "pred_len": args.pred_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
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
    predictor.load(num_features=len(feature_cols))

    print("开始接收组播并预测...")

    while True:
        try:
            data, addr = receiver.recv()
        except Exception:
            continue

        targets = parse_packet(data, skip_checksum=args.skip_checksum)
        if not targets:
            continue
        if args.print_targets:
            out = {
                "src": f"{addr[0]}:{addr[1]}",
                "count": len(targets),
                "targets": targets,
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))

        for tar in targets:
            tar_km = dict(tar)
            if "r_m" in tar_km:
                tar_km["r_m"] = float(tar_km["r_m"]) / 1000.0
            if "pr_m" in tar_km:
                tar_km["pr_m"] = float(tar_km["pr_m"]) / 1000.0
            raw_values = []
            for col in feature_cols:
                src_key = FEATURE_MAP.get(col)
                if src_key is None:
                    raw_values.append(0.0)
                else:
                    raw_values.append(float(tar_km.get(src_key, 0.0)))
            raw_vec = np.asarray(raw_values, dtype=np.float32)
            feat = normalizer.normalize(raw_vec)
            if feat is None:
                continue
            if args.print_features:
                raw_map = {feature_cols[i]: float(raw_vec[i]) for i in range(len(feature_cols))}
                norm_map = {feature_cols[i]: float(feat[i]) for i in range(len(feature_cols))}
                print(json.dumps({"raw": raw_map, "normalized": norm_map}, ensure_ascii=False))
            batch_id = tar.get("目标批号", tar.get("batch_id", tar.get("track_id", 0)))
            track_id = tar.get("track_id", tar.get("tar_seq", batch_id))
            buffer = batch_buffers.get(batch_id)
            if buffer is None:
                buffer = TrackWindowBuffer(seq_len=args.seq_len, max_age_s=args.max_age_s)
                batch_buffers[batch_id] = buffer
            batch_last_seen[batch_id] = time.time()
            buffer.update(track_id, feat, timestamp_s=tar.get("timestamp"))

        now = time.time()
        expired_batches = [bid for bid, ts in batch_last_seen.items() if now - ts > args.max_age_s]
        for bid in expired_batches:
            batch_buffers.pop(bid, None)
            batch_last_seen.pop(bid, None)
            batch_last_publish_ts.pop(bid, None)
            batch_ema_prob_uav.pop(bid, None)

        for batch_id, buffer in list(batch_buffers.items()):
            buffer.cleanup()
            pending = buffer.get_pending_progress(min_seq_len=min_seq_len, window_step=args.window_step)
            if pending:
                for tid, (count, needed) in pending.items():
                    print(f"batch_id={batch_id} track_id={tid} [{count}/{needed}]")
            track_ids, batch, lengths = buffer.build_batch(min_seq_len=min_seq_len, window_step=args.window_step)
            if batch is None:
                continue

            now = time.time()
            if args.publish_interval_ms > 0:
                last_ts = batch_last_publish_ts.get(batch_id, 0.0)
                if (now - last_ts) * 1000 < args.publish_interval_ms:
                    continue

            preds, probs = predictor.predict(batch, lengths)
            items = []
            for i, tid in enumerate(track_ids):
                prob_uav = float(probs[i, 1]) if probs.shape[1] > 1 else float(probs[i, 0])
                prob_bird = float(probs[i, 0]) if probs.shape[1] > 1 else 1.0 - prob_uav
                last_ts = float(buffer.get_last_timestamp(tid) or 0.0)
                items.append({
                    "batch_id": batch_id,
                    "track_id": tid,
                    "timestamp_ms": int(last_ts * 1000),
                    "time_25us": int(round(last_ts / 25e-6)),
                    "pred": int(preds[i]),
                    "prob_uav": prob_uav,
                    "prob_bird": prob_bird,
                })

            if args.use_batch_ema and items:
                alpha = max(0.0, min(1.0, float(args.ema_alpha)))
                current_prob = float(np.mean([item["prob_uav"] for item in items]))
                prev_prob = batch_ema_prob_uav.get(batch_id)
                ema_prob = current_prob if prev_prob is None else alpha * current_prob + (1.0 - alpha) * prev_prob
                batch_ema_prob_uav[batch_id] = ema_prob
                batch_pred = 1 if ema_prob >= 0.5 else 0
                for item in items:
                    item["pred"] = batch_pred
                    item["prob_uav"] = ema_prob
                    item["prob_bird"] = 1.0 - ema_prob

            print(json.dumps({"batch_id": batch_id, "count": len(items), "items": items}, ensure_ascii=False))
            publisher.send(items)
            buffer.mark_inferred(track_ids)
            batch_last_publish_ts[batch_id] = now


if __name__ == "__main__":
    main()
