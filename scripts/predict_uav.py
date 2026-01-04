"""
使用训练好的模型对新数据进行预测。
"""
import argparse
import json
import importlib
import os
import sys
from pathlib import Path

import time
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.uea import Normalizer, padding_mask


FEATURE_COLS = [
    "高（目标-滤波后）", "径向距离", "方位", "俯仰",
    "点迹距离", "点迹方位", "点迹俯仰",
    "全速度", "径向速度", "方位速度", "俯仰速度",
    "多普勒展宽", "JEM", "RCS"
]


def load_stats(stats_path, feature_cols):
    with open(stats_path, "r", encoding="utf-8") as f:
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
    mean_arr = pd.Series(mean_arr, index=feature_cols, dtype="float32")
    std_arr = pd.Series(std_arr, index=feature_cols, dtype="float32").replace(0, 1.0)
    return mean_arr, std_arr


def load_gbk_xls(path, feature_cols):
    """加载 GBK 编码的 xls 文件，并对齐特征列"""
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
    """收集目录下所有 xls 文件"""
    root = Path(root_path)
    files = []
    for p in root.rglob(pattern):
        if p.is_file():
            files.append(p)
    return files


def prepare_samples(file_paths, points_per_sample=20, feature_cols=None):
    """
    将文件列表转换为样本列表，每个文件作为一个样本
    """
    feature_cols = feature_cols or FEATURE_COLS
    samples = []
    for fpath in file_paths:
        try:
            df = load_gbk_xls(fpath, feature_cols)
            # 取最后 points_per_sample 行
            data = df.iloc[-points_per_sample:].values
            samples.append({
                'path': str(fpath),
                'data': data,
                'label': None  # 新数据无标签
            })
        except Exception as e:
            print(f"[WARN] skip {fpath}: {e}")
    return samples


class Args:
    """模拟 args 对象，用于模型初始化"""
    def __init__(self, cli_args, num_features, num_classes):
        self.task_name = 'classification'
        self.seq_len = cli_args.seq_len
        self.label_len = cli_args.label_len
        self.pred_len = 0
        self.d_model = cli_args.d_model
        self.enc_in = num_features
        self.e_layers = cli_args.e_layers
        self.d_ff = cli_args.d_ff
        self.top_k = cli_args.top_k
        self.num_kernels = cli_args.num_kernels
        self.embed = cli_args.embed
        self.freq = cli_args.freq
        self.dropout = cli_args.dropout
        self.num_class = num_classes


def main():
    parser = argparse.ArgumentParser(description='对新数据进行预测')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='新数据目录')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='输出结果文件')
    parser.add_argument('--model', type=str, default='TimesNet',
                        help='模型类型')
    parser.add_argument('--seq_len', type=int, default=20, help='序列长度')
    parser.add_argument('--label_len', type=int, default=48, help='与训练一致的 label_len')
    parser.add_argument('--points_per_sample', type=int, default=20, help='每个样本的点数')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数')
    parser.add_argument('--device', type=str, default='auto', help='设备: auto|cpu|cuda')
    parser.add_argument('--d_model', type=int, default=64, help='d_model')
    parser.add_argument('--d_ff', type=int, default=256, help='d_ff')
    parser.add_argument('--e_layers', type=int, default=2, help='e_layers')
    parser.add_argument('--top_k', type=int, default=2, help='top_k')
    parser.add_argument('--num_kernels', type=int, default=6, help='num_kernels')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='embed')
    parser.add_argument('--freq', type=str, default='h', help='freq')
    parser.add_argument('--norm_dir', type=str, default=None,
                        help='归一化统计使用的数据目录(默认=--data_dir)')
    parser.add_argument('--stats_path', type=str, default='',
                        help='归一化统计文件(stats.json)，优先生效')
    cli_args = parser.parse_args()

    if cli_args.device == 'auto':
        if torch.cuda.is_available():
            cli_args.device = 'cuda'
        else:
            cli_args.device = 'cpu'

    # 1. 收集数据文件
    print(f"收集数据文件: {cli_args.data_dir}")
    files = collect_files(cli_args.data_dir)
    print(f"找到 {len(files)} 个文件")

    # 2. 准备样本
    samples = prepare_samples(files, cli_args.points_per_sample, FEATURE_COLS)
    print(f"成功加载 {len(samples)} 个样本")

    if len(samples) == 0:
        print("没有找到有效样本，退出")
        return

    # 获取特征维度（从第一个样本）
    num_features = samples[0]['data'].shape[1]
    print(f"特征维度: {num_features}")

    # 归一化（优先使用 stats.json，其次使用训练数据统计）
    if cli_args.stats_path:
        mean, std = load_stats(cli_args.stats_path, FEATURE_COLS)
        for sample in samples:
            df = pd.DataFrame(sample["data"], columns=FEATURE_COLS)
            df_norm = (df - mean) / (std + 1e-8)
            sample["data"] = df_norm.values.astype("float32")
    else:
        norm_dir = cli_args.norm_dir or cli_args.data_dir
        if os.path.abspath(norm_dir) == os.path.abspath(cli_args.data_dir):
            # 如果norm_dir与data_dir相同，默认使用训练数据目录
            train_path = os.path.join(os.path.dirname(cli_args.data_dir), '..')
            if os.path.exists(os.path.join(train_path, 'bird')) or os.path.exists(os.path.join(train_path, '鸟')):
                norm_dir = train_path
                print(f"自动使用训练数据目录进行归一化: {norm_dir}")

        norm_files = collect_files(norm_dir)
        norm_samples = prepare_samples(norm_files, cli_args.points_per_sample, FEATURE_COLS)
        if len(norm_samples) == 0:
            print("归一化统计没有有效样本，退出")
            return
        norm_frames = []
        for idx, sample in enumerate(norm_samples):
            df = pd.DataFrame(sample["data"], columns=FEATURE_COLS)
            df.index = pd.Index([idx] * len(df))
            norm_frames.append(df)
        normalizer = Normalizer()
        train_concat = pd.concat(norm_frames, axis=0)
        normalizer.normalize(train_concat)
        for sample in samples:
            df = pd.DataFrame(sample["data"], columns=FEATURE_COLS)
            df_norm = normalizer.normalize(df.copy())
            sample["data"] = df_norm.values.astype("float32")

    # 3. 加载模型
    print(f"加载模型: {cli_args.model_path}")

    # 创建模拟 args 对象
    model_args = Args(cli_args, num_features, cli_args.num_classes)

    # 动态导入模型类（从 models.<model_name> 导入，类名都是 Model）
    model_module = importlib.import_module(f'models.{cli_args.model}')
    model_class = getattr(model_module, 'Model')
    model = model_class(model_args).float()
    model.load_state_dict(torch.load(cli_args.model_path, map_location='cpu'))
    model = model.to(cli_args.device)
    model.eval()

    # 4. 预测
    predictions = []
    inference_times = []

    print("开始预测...")
    for sample in tqdm(samples, disable=not sys.stdout.isatty(), mininterval=1.0):
        seq = torch.FloatTensor(sample['data'])
        if seq.shape[0] > cli_args.seq_len:
            seq = seq[-cli_args.seq_len:, :]
        length = seq.shape[0]
        if length < cli_args.seq_len:
            pad = torch.zeros(cli_args.seq_len - length, seq.shape[1])
            seq = torch.cat([seq, pad], dim=0)
        data = seq.unsqueeze(0).to(cli_args.device)  # (1, seq_len, feat)
        padding = padding_mask(torch.tensor([length], dtype=torch.int16),
                               max_len=cli_args.seq_len).to(cli_args.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(data, padding, None, None)
            prob = torch.nn.functional.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            prob_uav = prob[0, 1].item()  # uav 概率
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000
        inference_times.append(inference_time_ms)

        predictions.append({
            'file': os.path.basename(sample['path']),
            'prediction': 'uav' if pred == 1 else 'bird',
            'confidence': prob_uav if pred == 1 else 1 - prob_uav,
            'prob_uav': prob_uav,
            'prob_bird': prob[0, 0].item()
        })

    # 5. 保存结果
    result_df = pd.DataFrame(predictions)
    result_df.to_csv(cli_args.output, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存到: {cli_args.output}")

    # 统计
    uav_count = sum(1 for p in predictions if p['prediction'] == 'uav')
    bird_count = len(predictions) - uav_count
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0

    print(f"\n预测统计:")
    print(f"  无人机 (uav): {uav_count}")
    print(f"  鸟 (bird): {bird_count}")
    print(f"\n推理时间统计:")
    print(f"  平均推理时间: {avg_time:.2f} ms/条")


if __name__ == '__main__':
    main()
