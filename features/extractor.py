import json
from pathlib import Path

import numpy as np

from features.feature_config import FEATURE_MAP, get_feature_cols

FEATURE_COLS = get_feature_cols()


class FeatureNormalizer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    @classmethod
    def from_stats_file(cls, stats_path, feature_cols=None):
        if not stats_path:
            return cls()
        feature_cols = feature_cols or get_feature_cols()
        path = Path(stats_path)
        with path.open("r", encoding="utf-8") as f:
            stats = json.load(f)

        mean = stats.get("mean")
        std = stats.get("std")
        if isinstance(mean, dict) and isinstance(std, dict):
            mean_arr = [mean.get(c, 0.0) for c in feature_cols]
            std_arr = [std.get(c, 1.0) for c in feature_cols]
        elif isinstance(mean, list) and isinstance(std, list):
            if len(mean) != len(feature_cols) or len(std) != len(feature_cols):
                raise ValueError("统计维度与特征维度不一致")
            mean_arr = mean
            std_arr = std
        else:
            raise ValueError("stats.json 结构不支持，仅支持 mean/std 的 list 或 dict")

        mean_arr = np.asarray(mean_arr, dtype=np.float32)
        std_arr = np.asarray(std_arr, dtype=np.float32)
        std_arr = np.where(std_arr == 0, 1.0, std_arr)
        return cls(mean_arr, std_arr)

    def normalize(self, x):
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / self.std


class FeatureExtractor:
    def __init__(self, normalizer=None, feature_cols=None, feature_map=None):
        self.feature_cols = feature_cols or get_feature_cols()
        self.feature_map = feature_map or FEATURE_MAP
        self.normalizer = normalizer or FeatureNormalizer()

    def extract(self, target):
        values = []
        for col in self.feature_cols:
            src_key = self.feature_map.get(col)
            if src_key is None:
                values.append(0.0)
            else:
                values.append(float(target.get(src_key, 0.0)))
        vec = np.asarray(values, dtype=np.float32)
        return self.normalizer.normalize(vec)
