"""
基于轨迹数据生成归一化统计文件（mean/std）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "高（目标-滤波后）", "径向距离", "方位", "俯仰",
    "点迹距离", "点迹方位", "点迹俯仰",
    "全速度", "径向速度", "方位速度", "俯仰速度",
    "多普勒展宽", "JEM", "RCS",
]


def load_gbk_xls(path, feature_cols):
    df = pd.read_csv(path, sep="\t", encoding="gbk")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")
    df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.interpolate(limit_direction="both")
    df = df.ffill().bfill()
    return df


def collect_files(root_path, pattern="*.xls"):
    root = Path(root_path)
    files = []
    for p in root.rglob(pattern):
        if p.is_file():
            files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser(description="生成归一化统计文件")
    parser.add_argument("--data_dir", required=True, help="数据目录")
    parser.add_argument("--output", required=True, help="输出 stats.json 路径")
    parser.add_argument("--pattern", default="*.xls", help="文件匹配模式")
    parser.add_argument("--feature_cols", default="", help="逗号分隔的特征列名列表")
    args = parser.parse_args()

    if args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    else:
        feature_cols = FEATURE_COLS

    files = collect_files(args.data_dir, args.pattern)
    if not files:
        raise SystemExit("未找到任何数据文件")

    frames = []
    for fpath in files:
        try:
            df = load_gbk_xls(fpath, feature_cols)
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] skip {fpath}: {exc}")

    if not frames:
        raise SystemExit("没有有效样本，无法生成统计")

    concat = pd.concat(frames, axis=0)
    mean = concat.mean()
    std = concat.std()
    std = std.replace(0, 1.0)

    stats = {
        "mean": {k: float(mean[k]) for k in feature_cols},
        "std": {k: float(std[k]) for k in feature_cols},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"统计已保存到: {out_path}")


if __name__ == "__main__":
    main()
