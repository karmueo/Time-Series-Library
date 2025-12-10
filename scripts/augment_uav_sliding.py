import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate augmented trajectory samples via sliding window for a chosen class; copy others.")
    parser.add_argument("--input_dir", required=True, help="Root directory of original dataset.")
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
        choices=["uav", "bird"],
        default="uav",
        help="Which class to apply sliding windows to; the other class will be copied as-is.",
    )
    return parser.parse_args()


def classify_path(path: Path) -> str:
    parts = set(path.parts)
    if any("无人机" in p for p in parts):
        return "uav"
    if any("鸟" == p or p.endswith("鸟") for p in parts):
        return "bird"
    return "unknown"


def load_gbk_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="gbk")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
    return df


def save_window(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", encoding="gbk", index=False)


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    input_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    min_len = args.min_len or args.window_size

    files = [p for p in input_root.rglob(args.pattern) if p.is_file()]
    processed = 0
    skipped = 0
    generated = 0
    copied = 0

    for path in files:
        cls = classify_path(path.relative_to(input_root))
        rel = path.relative_to(input_root)

        if cls == "unknown":
            # copy unknown class to keep structure
            copy_file(path, output_root / rel)
            copied += 1
            continue

        if cls != args.augment_class:
            copy_file(path, output_root / rel)
            copied += 1
            continue

        try:
            df = load_gbk_tsv(path)
        except Exception as e:
            print(f"[WARN] skip {path}: load error {e}")
            skipped += 1
            continue

        if len(df) < min_len:
            copy_file(path, output_root / rel)
            copied += 1
            continue

        windows = []
        start = 0
        while start + args.window_size <= len(df):
            windows.append((start, start + args.window_size))
            start += args.stride

        if args.include_partial and start < len(df):
            windows.append((len(df) - args.window_size, len(df)) if len(df) >= args.window_size else (0, len(df)))

        for idx, (s, e) in enumerate(windows):
            wdf = df.iloc[s:e]
            out_name = f"{path.stem}_win{s}_{e}{path.suffix}"
            out_path = output_root / rel.parent / out_name
            save_window(wdf, out_path)
            generated += 1

        processed += 1

    print(f"Processed (augmented) files: {processed}")
    print(f"Generated windows: {generated}")
    print(f"Copied (other classes or short/unknown): {copied}")
    print(f"Skipped (error): {skipped}")
    print(f"Output dir: {output_root}")


if __name__ == "__main__":
    main()
