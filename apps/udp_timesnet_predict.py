import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.predictor import OnnxTimesNetPredictor, TimesNetPredictor
from core.track_buffer import TrackWindowBuffer
from features.extractor import FeatureNormalizer
from features.feature_config import FEATURE_MAP, get_feature_cols
from udp.parser import parse_packet
from udp.publisher import MulticastPublisher
from udp.receiver import MulticastReceiver


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """将配置文件的值合并到参数中（仅在命令行未显式指定时）"""
    # 记录命令行显式指定的参数
    cmd_line_args = set()
    for argv in sys.argv[1:]:
        if argv.startswith("--"):
            arg_name = argv[2:].split("=")[0]
            cmd_line_args.add(arg_name)

    # 配置文件字段到命令行参数的映射
    config_mapping = {
        "receiver": {
            "group": "in_group",
            "port": "in_port",
            "iface": "in_iface",
            "timeout_s": "timeout",
        },
        "publisher": {
            "group": "out_group",
            "port": "out_port",
            "iface": "out_iface",
        },
        "local_test": {
            "enabled": "local_test",
            "path": "local_test_path",
            "points": "local_test_points",
        },
        # 其他字段直接映射
        "buffer": None,
        "normalizer": None,
        "predictor": None,
        "inference": None,
    }

    for section, values in config.items():
        if not isinstance(values, dict):
            continue

        mapping = config_mapping.get(section, {})
        for key, value in values.items():
            # 确定目标参数名
            if mapping and key in mapping:
                arg_name = mapping[key]
            elif mapping is None:
                # 对于没有映射的 section，直接使用 key
                arg_name = key
            else:
                continue

            # 只有当命令行未显式指定时才使用配置文件的值
            if arg_name not in cmd_line_args and hasattr(args, arg_name):
                # 特殊处理 boolean 类型的 flag
                if key == "enabled" and isinstance(value, bool):
                    if value:
                        setattr(args, arg_name, True)
                else:
                    setattr(args, arg_name, value)

    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description="组播航迹接收 + TimesNet 预测 + 组播发布",
        epilog="配置文件中的参数会被命令行参数覆盖。使用 --config 指定 YAML 配置文件。"
    )
    parser.add_argument("--config", default="", help="YAML 配置文件路径")

    # 接收器配置
    parser.add_argument("--in_group", default="", help="输入组播地址")
    parser.add_argument("--in_port", type=int, default=0, help="输入端口")
    parser.add_argument("--in_iface", default="0.0.0.0", help="输入网卡IP")
    parser.add_argument("--bind_ip", default="", help="绑定IP，默认0.0.0.0")
    parser.add_argument("--timeout", type=float, default=2.0, help="接收超时秒数")
    parser.add_argument("--skip_checksum", action="store_true", help="跳过累加和校验")

    # 发布器配置
    parser.add_argument("--out_group", default="", help="输出组播地址")
    parser.add_argument("--out_port", type=int, default=0, help="输出端口")
    parser.add_argument("--out_iface", default="0.0.0.0", help="输出网卡IP")
    parser.add_argument("--ttl", type=int, default=1, help="组播TTL")

    # 预测器配置
    parser.add_argument("--model_path", default="", help="模型 checkpoint 路径")
    parser.add_argument("--model", default="TimesNet", help="模型类型")
    parser.add_argument("--num_classes", type=int, default=2, help="类别数")
    parser.add_argument("--seq_len", type=int, default=20, help="序列长度")
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

    # 缓冲区配置
    parser.add_argument("--min_seq_len", type=int, default=20, help="最小序列长度")
    parser.add_argument("--max_age_s", type=float, default=10.0, help="轨迹最大保留秒数")
    parser.add_argument("--window_step", type=int, default=0, help="滑窗步长(0=每条都推理)")

    # 归一化配置
    parser.add_argument("--stats_path", default="", help="归一化统计文件 stats.json")

    # 推理配置
    parser.add_argument("--publish_interval_ms", type=int, default=0, help="发布节流毫秒(0=不节流)")
    parser.add_argument("--print_targets", action="store_true", help="打印解析后的目标信息")
    parser.add_argument("--print_features", action="store_true", help="打印送入模型的特征值")
    parser.add_argument("--use_batch_ema", action="store_true", help="同批号目标使用EMA平滑概率并统一判别")
    parser.add_argument("--ema_alpha", type=float, default=0.6, help="EMA平滑系数(0-1)")

    # 本地测试配置
    parser.add_argument("--local_test", action="store_true", help="启用本地文件测试(替代组播)")
    parser.add_argument("--local_test_path", default="", help="本地 .xls/.csv 路径")
    parser.add_argument("--local_test_points", type=int, default=20, help="读取点数(默认前20)")

    # 解析命令行参数
    args = parser.parse_args()

    # 如果指定了配置文件，加载并合并配置
    if args.config:
        config = load_config(args.config)
        args = _merge_config_with_args(config, args)

    return args


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _detect_delimiter(path: Path) -> str:
    return "\t" if path.suffix.lower() == ".xls" else ","


def _parse_float(value: str, field: str, line_no: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"第{line_no}行字段{field}无法转换为浮点数: {value}") from exc


def load_local_trajectory(path: str, max_points: int, feature_cols: list[str]) -> list[np.ndarray]:
    if max_points <= 0:
        raise ValueError("local_test_points 必须大于 0")

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"本地文件不存在: {file_path}")

    delimiter = _detect_delimiter(file_path)
    features = []
    with file_path.open("r", encoding="gbk") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("本地文件缺少表头")
        for line_no, row in enumerate(reader, start=2):
            if len(features) >= max_points:
                break
            values = []
            for col in feature_cols:
                key = col if col in row else FEATURE_MAP.get(col, col)
                raw = row.get(key, "")
                raw = raw.strip() if isinstance(raw, str) else raw
                if raw in ("", None):
                    val = 0.0
                else:
                    val = _parse_float(str(raw), col, line_no)
                if col in {"径向距离", "点迹距离", "全速度", "径向速度"}:
                    val = val / 1000.0
                values.append(val)
            vec = np.asarray(values, dtype=np.float32)
            features.append(vec)

    if not features:
        raise ValueError("本地文件无有效数据行")
    return features


def run_local_file_inference(args, feature_cols, normalizer, predictor):
    local_features = load_local_trajectory(args.local_test_path, args.local_test_points, feature_cols)
    if len(local_features) < args.seq_len:
        raise ValueError(f"本地文件点数不足: {len(local_features)} < seq_len={args.seq_len}")

    seq = np.stack(local_features[:args.seq_len], axis=0).astype(np.float32)
    seq = np.asarray([normalizer.normalize(seq)], dtype=np.float32)
    lengths = np.asarray([args.seq_len], dtype=np.int64)

    preds, probs = predictor.predict(seq, lengths)
    prob_uav = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
    prob_bird = float(probs[0, 0]) if probs.shape[1] > 1 else 1.0 - prob_uav

    item = {
        "batch_id": 1,
        "track_id": 1,
        "timestamp_ms": 0,
        "time_25us": 0,
        "pred": int(preds[0]),
        "prob_uav": prob_uav,
        "prob_bird": prob_bird,
    }
    print(json.dumps({"batch_id": 1, "count": 1, "items": [item]}, ensure_ascii=False))


def main():
    args = parse_args()
    if args.local_test:
        if not args.local_test_path:
            raise ValueError("启用 --local_test 时必须提供 --local_test_path")
        if args.local_test_points <= 0:
            raise ValueError("--local_test_points 必须大于 0")
    else:
        if not args.in_group:
            raise ValueError("未启用本地测试时必须提供 --in_group")
        if args.in_port <= 0:
            raise ValueError("未启用本地测试时必须提供 --in_port")
        if not args.out_group:
            raise ValueError("未启用本地测试时必须提供 --out_group")
        if args.out_port <= 0:
            raise ValueError("未启用本地测试时必须提供 --out_port")
    device = resolve_device(args.device)
    min_seq_len = max(1, min(args.min_seq_len, args.seq_len))

    feature_cols = get_feature_cols()
    normalizer = FeatureNormalizer.from_stats_file(args.stats_path, feature_cols) if args.stats_path else FeatureNormalizer()
    batch_buffers = {}
    batch_last_seen = {}
    batch_last_publish_ts = {}
    batch_ema_prob_uav = {}

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
    model_path = Path(args.model_path)
    if model_path.suffix.lower() == ".onnx":
        predictor = OnnxTimesNetPredictor(
            model_path=args.model_path,
            device=device,
        )
    else:
        predictor = TimesNetPredictor(
            model_path=args.model_path,
            model_name=args.model,
            num_classes=args.num_classes,
            device=device,
            model_cfg=model_cfg,
        )
    predictor.load(num_features=len(feature_cols))

    if args.local_test:
        run_local_file_inference(args, feature_cols, normalizer, predictor)
        return

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

        # 批量推理优化：收集所有批号的待推理数据
        all_batch_ids = []
        all_track_ids = []
        all_buffers = []
        total_samples = 0

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

            all_batch_ids.append(batch_id)
            all_track_ids.append(track_ids)
            all_buffers.append((batch, lengths, buffer))
            total_samples += len(track_ids)

        # 批量推理：合并所有批号的数据一次性推理
        if all_buffers:
            merged_batch = np.concatenate([b[0] for b in all_buffers], axis=0)
            merged_lengths = np.concatenate([b[1] for b in all_buffers], axis=0)

            preds, probs = predictor.predict(merged_batch, merged_lengths)

            # 分发结果到各个批号
            sample_offset = 0
            for idx, batch_id in enumerate(all_batch_ids):
                track_ids = all_track_ids[idx]
                buffer = all_buffers[idx][2]
                num_samples = len(track_ids)

                batch_preds = preds[sample_offset:sample_offset + num_samples]
                batch_probs = probs[sample_offset:sample_offset + num_samples]

                items = []
                for i, tid in enumerate(track_ids):
                    prob_uav = float(batch_probs[i, 1]) if batch_probs.shape[1] > 1 else float(batch_probs[i, 0])
                    prob_bird = float(batch_probs[i, 0]) if batch_probs.shape[1] > 1 else 1.0 - prob_uav
                    last_ts = float(buffer.get_last_timestamp(tid) or 0.0)
                    items.append({
                        "batch_id": batch_id,
                        "track_id": tid,
                        "timestamp_ms": int(last_ts * 1000),
                        "time_25us": int(round(last_ts / 25e-6)),
                        "pred": int(batch_preds[i]),
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
                batch_last_publish_ts[batch_id] = time.time()

                sample_offset += num_samples


if __name__ == "__main__":
    main()
