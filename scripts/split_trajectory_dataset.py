"""
航迹数据集分割脚本

功能：
- 按指定数量或比例将航迹数据拆分成两份
- 支持移动或复制两种输出模式
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Union


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split trajectory dataset by class into train/test sets."
    )
    parser.add_argument(
        "--root_dir", required=True, help="Root directory of dataset (contains class subfolders)."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for split results."
    )
    parser.add_argument(
        "--classes",
        required=True,
        help="Classes to split, comma-separated (e.g., 'uav,bird') or 'all' for all classes.",
    )
    parser.add_argument(
        "--split_value",
        required=True,
        type=float,
        help="Split value: if < 1, treated as ratio (e.g., 0.4 = 40%%); if >= 1, treated as count.",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.xls",
        help="Glob pattern for trajectory files (default: *.xls).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["move", "copy"],
        default="move",
        help="Output mode: 'move' = move split files, keep rest in place; 'copy' = copy both sets to new dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2021,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan files recursively in subfolders.",
    )
    return parser.parse_args()


def get_files_by_class(root_dir: Path, classes: set, pattern: str, recursive: bool) -> dict:
    """获取每个类别的文件列表"""
    class_files = {cls: [] for cls in classes}

    for folder in root_dir.iterdir():
        if not folder.is_dir():
            continue
        if folder.name not in classes:
            continue

        if recursive:
            files = list(folder.rglob(pattern))
        else:
            files = list(folder.glob(pattern))

        class_files[folder.name] = files

    return class_files


def calculate_split_count(total: int, split_value: float) -> tuple[int, int]:
    """
    计算分割数量

    Args:
        total: 总文件数
        split_value: 分割值（数量或比例）

    Returns:
        (first_set_count, second_set_count)
    """
    if split_value < 1:
        # 比例模式
        first_count = int(total * split_value)
        # 确保至少有一个文件
        first_count = max(1, first_count)
    else:
        # 数量模式
        first_count = min(int(split_value), total - 1)  # 至少保留1个给第二份
        first_count = max(1, first_count)

    second_count = total - first_count
    return first_count, second_count


def split_and_process(
    root_dir: Path,
    output_dir: Path,
    class_files: dict,
    split_value: float,
    mode: str,
    seed: int,
):
    """执行分割操作"""
    random.seed(seed)

    summary = {}

    for cls, files in class_files.items():
        if not files:
            print(f"[WARN] No files found for class: {cls}")
            summary[cls] = {"total": 0, "split1": 0, "split2": 0}
            continue

        total = len(files)
        split1_count, split2_count = calculate_split_count(total, split_value)

        # 随机打乱文件
        shuffled = files.copy()
        random.shuffle(shuffled)

        split1_files = shuffled[:split1_count]
        split2_files = shuffled[split1_count:]

        if mode == "move":
            # 移动模式：split1 保留在原目录，split2 直接移动到输出目录
            output_cls_dir = output_dir / cls
            output_cls_dir.mkdir(parents=True, exist_ok=True)

            # 将 split2（剩余数据）直接移动到输出目录的类别文件夹
            for f in split2_files:
                dst = output_cls_dir / f.name
                if dst.exists():
                    dst.unlink()
                try:
                    f.rename(dst)
                except OSError:
                    shutil.move(str(f), str(dst))

            print(f"[{cls}] Total: {total}")
            print(f"[{cls}] split1 (kept): {len(split1_files)} files (stay in {root_dir / cls})")
            print(f"[{cls}] split2 (moved): {len(split2_files)} files -> {output_cls_dir}")

        else:  # copy模式
            split1_dir = output_dir / cls / "split1"
            split2_dir = output_dir / cls / "split2"

            split1_dir.mkdir(parents=True, exist_ok=True)
            split2_dir.mkdir(parents=True, exist_ok=True)

            for f in split1_files:
                dst = split1_dir / f.name
                shutil.copy2(str(f), str(dst))

            for f in split2_files:
                dst = split2_dir / f.name
                shutil.copy2(str(f), str(dst))

            print(f"[{cls}] Total: {total} -> split1: {len(split1_files)} (copied to {split1_dir})")
            print(f"[{cls}] split2: {len(split2_files)} (copied to {split2_dir})")

        summary[cls] = {
            "total": total,
            "split1": len(split1_files),
            "split2": len(split2_files),
        }

    return summary


def main():
    args = parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not root_dir.exists():
        print(f"[ERROR] Root directory does not exist: {root_dir}")
        return

    # 解析类别
    classes = set(args.classes.split(','))
    if 'all' in classes:
        classes = {f.name for f in root_dir.iterdir() if f.is_dir()}
    else:
        # 只保留存在的类别
        existing = {f.name for f in root_dir.iterdir() if f.is_dir()}
        classes = classes & existing

    if not classes:
        print(f"[ERROR] No valid classes found in {root_dir}")
        return

    # 获取每个类别的文件
    class_files = get_files_by_class(root_dir, classes, args.file_pattern, args.recursive)

    # 显示分割预览
    print("\n=== Split Preview ===")
    for cls, files in class_files.items():
        if files:
            total = len(files)
            split1_count, split2_count = calculate_split_count(total, args.split_value)
            mode_desc = "ratio" if args.split_value < 1 else "count"
            print(f"  {cls}: {total} files -> split1: {split1_count}, split2: {split2_count} ({mode_desc})")
        else:
            print(f"  {cls}: no files found")

    # 确认执行
    mode_desc = "移动" if args.mode == "move" else "复制"
    print(f"\nMode: {mode_desc}")
    print(f"Output: {output_dir}")

    confirm = input("\nProceed with split? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    # 执行分割
    if args.mode == "move":
        output_dir.mkdir(parents=True, exist_ok=True)

    summary = split_and_process(
        root_dir, output_dir, class_files, args.split_value, args.mode, args.seed
    )

    # 打印汇总
    print("\n=== Summary ===")
    total_all = sum(s["total"] for s in summary.values())
    split1_all = sum(s["split1"] for s in summary.values())
    split2_all = sum(s["split2"] for s in summary.values())

    for cls, stats in summary.items():
        print(f"  {cls}: {stats['total']} -> split1: {stats['split1']}, split2: {stats['split2']}")

    print(f"\nTotal: {total_all} files")
    print(f"  split1: {split1_all} files")
    print(f"  split2: {split2_all} files")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
