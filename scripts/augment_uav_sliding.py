import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate augmented trajectory samples via sliding window for a chosen class; copy others.")
    parser.add_argument("--input_dir", required=True, help="Root directory of original dataset (contains class subfolders).")
    parser.add_argument("--output_dir", required=True, help="Directory to write augmented samples.")
    parser.add_argument("--window_size", type=int, default=20, help="Number of points per window.")
    parser.add_argument("--stride", type=int, default=5, help="Sliding stride between windows.")
    parser.add_argument(
        "--include_partial",
        action="store_true",
        help="If set, also generate the last partial window when length is not enough for a full window.",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=None,
        help="Minimum sequence length to process; default equals window_size.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.xls",
        help="Glob pattern for files; default to xls (tab-separated GBK text in this repo).",
    )
    parser.add_argument(
        "--augment_class",
        type=str,
        default="all",
        help="Class to apply sliding windows to; supports comma-separated for multiple classes (e.g., 'uav,bird'). "
        "The other class will be copied as-is. Use 'all' to augment both classes.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="If set, scan files recursively in subfolders (default: True).",
    )
    parser.add_argument(
        "--feature_cols",
        type=str,
        default=None,
        help="Comma-separated list of feature columns to keep. If None, keep all columns.",
    )
    return parser.parse_args()


def classify_folder(folder_name: str) -> str:
    """根据文件夹名称分类"""
    folder_lower = folder_name.lower()
    # uav 关键词
    uav_keywords = ['uav', 'drone', '无人机']
    # bird 关键词
    bird_keywords = ['bird', 'birds', '鸟']

    if any(kw in folder_lower for kw in uav_keywords):
        return "uav"
    if any(kw in folder_lower for kw in bird_keywords):
        return "bird"
    return "other"


def load_gbk_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="gbk")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
    return df


def filter_columns(df: pd.DataFrame, feature_cols: list[str] | None) -> pd.DataFrame:
    """只保留指定的特征列"""
    if feature_cols is None:
        return df
    # 只保留存在的列
    cols = [c for c in feature_cols if c in df.columns]
    return df[cols]


def save_window(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", encoding="gbk", index=False)


def copy_file(src: Path, dst: Path, feature_cols: list[str] | None = None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if feature_cols is None:
        shutil.copy2(src, dst)
    else:
        # 读取并过滤列后再保存
        df = load_gbk_tsv(src)
        df = filter_columns(df, feature_cols)
        df.to_csv(dst, sep="\t", encoding="gbk", index=False)


def main():
    args = parse_args()
    input_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 解析 augment_class 参数，支持逗号分隔或 "all"
    augment_classes = set(args.augment_class.split(','))
    if 'all' in augment_classes:
        augment_classes = {'uav', 'bird'}
    # 验证类别有效性
    augment_classes = augment_classes & {'uav', 'bird'}
    if not augment_classes:
        print(f"[ERROR] Invalid augment_class: {args.augment_class}")
        return

    # 解析特征列
    feature_cols = None
    if args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(',')]

    min_len = args.min_len or args.window_size

    # 扫描一级子文件夹，按文件夹名称分类
    folder_map = {}  # {folder_path: class_name}
    for folder in input_root.iterdir():
        if folder.is_dir():
            cls = classify_folder(folder.name)
            folder_map[folder] = cls

    processed = 0
    skipped = 0
    generated = 0
    copied = 0
    # 统计每个类别生成的文件数量
    generated_by_class = {"uav": 0, "bird": 0, "other": 0}
    copied_by_class = {"uav": 0, "bird": 0, "other": 0}

    for folder, cls in folder_map.items():
        # 收集该文件夹下所有匹配的文件
        if args.recursive:
            files = list(folder.rglob(args.pattern))
        else:
            files = list(folder.glob(args.pattern))
        for path in files:
            if cls == "other":
                # other 类直接复制到对应文件夹
                copy_file(path, output_root / cls / path.name, feature_cols)
                copied += 1
                copied_by_class["other"] += 1
                continue

            if cls not in augment_classes:
                # 不需要增强的类别也直接复制
                copy_file(path, output_root / cls / path.name, feature_cols)
                copied += 1
                copied_by_class[cls] += 1
                continue

            try:
                df = load_gbk_tsv(path)
                # 过滤特征列
                df = filter_columns(df, feature_cols)
            except Exception as e:
                print(f"[WARN] skip {path}: load error {e}")
                skipped += 1
                continue

            if len(df) < min_len:
                copy_file(path, output_root / cls / path.name, feature_cols)
                copied += 1
                copied_by_class[cls] += 1
                continue

            windows = []
            start = 0
            while start + args.window_size <= len(df):
                windows.append((start, start + args.window_size))
                start += args.stride

            if args.include_partial and start < len(df):
                windows.append((len(df) - args.window_size, len(df)) if len(df) >= args.window_size else (0, len(df)))

            for _, (s, e) in enumerate(windows):
                wdf = df.iloc[s:e]
                out_name = f"{path.stem}_win{s}_{e}{path.suffix}"
                out_path = output_root / cls / out_name
                save_window(wdf, out_path)
                generated += 1
                generated_by_class[cls] += 1

            processed += 1

    # 统计各文件夹信息
    print("\n=== Folder Summary ===")
    for folder, cls in folder_map.items():
        if args.recursive:
            file_count = len(list(folder.rglob(args.pattern)))
        else:
            file_count = len(list(folder.glob(args.pattern)))
        status = "augment" if cls in augment_classes else "copy"
        print(f"  {folder.name}: {file_count} files -> class={cls} ({status})")

    # 输出各类别生成的文件数量
    print("\n=== Generated Files by Class ===")
    for cls in ["uav", "bird", "other"]:
        if generated_by_class[cls] > 0 or copied_by_class[cls] > 0:
            print(f"  {cls}: generated={generated_by_class[cls]}, copied={copied_by_class[cls]}")

    print(f"\nProcessed (augmented) files: {processed}")
    print(f"Generated windows: {generated}")
    print(f"Copied (other classes or short): {copied}")
    print(f"Skipped (error): {skipped}")
    print(f"Output dir: {output_root}")


if __name__ == "__main__":
    main()
