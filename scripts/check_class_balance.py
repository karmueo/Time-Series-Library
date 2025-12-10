import argparse
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Count bird/uav samples to check class balance.")
    parser.add_argument("--data_dir", required=True, help="Root directory of dataset (e.g., mydataset or augmented dir).")
    parser.add_argument("--pattern", type=str, default="*.xls", help="Glob pattern for files.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/Test split ratio by file count.")
    parser.add_argument("--seed", type=int, default=2, help="Random seed for shuffling.")
    return parser.parse_args()


def classify_path(path: Path) -> str:
    parts = set(path.parts)
    if any("无人机" in p for p in parts):
        return "uav"
    if any("鸟" == p or p.endswith("鸟") for p in parts):
        return "bird"
    return "unknown"


def main():
    import random

    args = parse_args()
    root = Path(args.data_dir).expanduser().resolve()
    files = [p for p in root.rglob(args.pattern) if p.is_file()]

    labeled = []
    for p in files:
        cls = classify_path(p.relative_to(root))
        if cls == "unknown":
            continue
        labeled.append((p, cls))

    if not labeled:
        print("No labeled files found.")
        return

    random.seed(args.seed)
    random.shuffle(labeled)

    split = int(len(labeled) * args.train_ratio)
    train = labeled[:split]
    test = labeled[split:]

    def summarize(name, items):
        c = Counter(cls for _, cls in items)
        total = sum(c.values())
        print(f"{name}: total={total}, counts={dict(c)}, ratios={{k: v/total for k,v in c.items()}}")

    summarize("Train", train)
    summarize("Test", test)
    summarize("All", labeled)


if __name__ == "__main__":
    main()
