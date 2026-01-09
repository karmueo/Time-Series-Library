#!/usr/bin/env python3
"""
导出 TimesNet 分类模型到 ONNX 格式
使用预计算周期列表避免 FFT 操作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import onnx
import onnxruntime as ort
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../..')
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from utils.tools import dotdict


class TimesBlock_Precomputed(nn.Module):
    """使用预计算周期的 TimesBlock（可导出到 ONNX）"""
    def __init__(self, configs, period_list):
        super(TimesBlock_Precomputed, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.period_list = period_list  # 预计算的周期列表

        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        total_len = self.seq_len + self.pred_len

        res = []
        for period in self.period_list:
            if period == 0:
                period = 1

            # 计算 padding
            if total_len % period != 0:
                length = ((total_len // period) + 1) * period
                padding_len = length - total_len
                padding = torch.zeros([x.shape[0], padding_len, x.shape[2]], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x

            # Reshape 和 permute
            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2).contiguous()

            # 2D convolution
            out = self.conv(out)

            # Reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total_len, :])

        res = torch.stack(res, dim=-1)

        # 使用等权重（简化版本）
        period_weight = torch.ones(1, 1, 1, self.k, device=x.device) / self.k

        # 加权求和
        res = (res * period_weight).sum(dim=-1)

        # 残差连接
        res = res + x
        return res


def compute_period_list(x, k=5):
    """计算输入数据的周期列表"""
    B, T, N = x.size()

    # FFT
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = torch.abs(xf).mean(0).mean(-1)
    frequency_list = frequency_list.clone()
    frequency_list[0] = 0

    # Top-k
    _, top_k_indices = torch.topk(frequency_list, k)

    # 转换为周期
    period_list = T // (top_k_indices + 1)

    return period_list.tolist()


class Model_Precomputed(nn.Module):
    """使用预计算周期的 TimesNet 分类模型"""
    def __init__(self, configs, period_list):
        super(Model_Precomputed, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.period_list = period_list

        self.model = nn.ModuleList([TimesBlock_Precomputed(configs, period_list)
                                    for _ in range(configs.e_layers)])

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.d_model * configs.seq_len, configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None


def load_and_prepare_model(checkpoint_path, sample_input=None, config_override=None, period_list=None):
    """加载模型并准备 ONNX 导出

    Args:
        checkpoint_path: checkpoint 文件路径
        sample_input: 样本输入 tensor，用于计算 period_list（如果未指定）
        config_override: 配置覆盖
        period_list: 预计算的 period_list，如果为 None 则从 sample_input 计算
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 初始化 config 对象（必须在任何 config 属性赋值之前）
    config = dotdict()

    proj_weight = checkpoint.get('projection.weight', None)
    num_class = proj_weight.shape[0] if proj_weight is not None else 2

    conv_keys = [k for k in checkpoint.keys() if k.startswith('model.0.conv.0.kernels.') and 'weight' in k]
    num_kernels = len(conv_keys)

    # 从 enc_embedding 的 tokenConv 权重提取 enc_in（特征数）
    # shape: [d_model, enc_in, kernel_size] 或 [d_model, enc_in, 3]
    token_conv_weight = checkpoint.get('enc_embedding.value_embedding.tokenConv.weight', None)
    if token_conv_weight is not None:
        config.enc_in = token_conv_weight.shape[1]  # shape: [d_model, enc_in, kernel_size]
    else:
        # 备用方案：检查 temporal_embedding
        embed_weight = checkpoint.get('enc_embedding.temporal_embedding.embed.weight', None)
        if embed_weight is not None:
            config.enc_in = embed_weight.shape[0] if len(embed_weight.shape) > 1 else 1
        else:
            config.enc_in = 1

    embed_weight = checkpoint.get('enc_embedding.temporal_embedding.embed.weight', None)
    # 根据 TimeFeatureEmbedding 的 freq_map: {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
    if embed_weight is not None:
        shape_to_freq = {1: 'm', 2: 'w', 3: 'd', 4: 'h', 5: 't', 6: 's'}
        freq = shape_to_freq.get(embed_weight.shape[1], 't')
    else:
        freq = 't'

    checkpoint_name = os.path.basename(os.path.dirname(checkpoint_path))
    config.task_name = 'classification'

    if 'sl' in checkpoint_name:
        idx = checkpoint_name.find('sl') + 2
        seq_len = ''
        while idx < len(checkpoint_name) and checkpoint_name[idx].isdigit():
            seq_len += checkpoint_name[idx]
            idx += 1
        config.seq_len = int(seq_len)
    else:
        config.seq_len = 96

    config.label_len = 48
    config.pred_len = 0

    if '_dm' in checkpoint_name:
        idx = checkpoint_name.find('dm') + 2
        d_model = ''
        while idx < len(checkpoint_name) and checkpoint_name[idx].isdigit():
            d_model += checkpoint_name[idx]
            idx += 1
        config.d_model = int(d_model)
    else:
        config.d_model = 16

    if '_nh' in checkpoint_name:
        idx = checkpoint_name.find('nh') + 2
        n_heads = ''
        while idx < len(checkpoint_name) and checkpoint_name[idx].isdigit():
            n_heads += checkpoint_name[idx]
            idx += 1
        config.n_heads = int(n_heads)
    else:
        config.n_heads = 8

    if '_el' in checkpoint_name:
        idx = checkpoint_name.find('el') + 2
        e_layers = ''
        while idx < len(checkpoint_name) and checkpoint_name[idx].isdigit():
            e_layers += checkpoint_name[idx]
            idx += 1
        config.e_layers = int(e_layers)
    else:
        config.e_layers = 2

    if '_df' in checkpoint_name:
        idx = checkpoint_name.find('df') + 2
        d_ff = ''
        while idx < len(checkpoint_name) and checkpoint_name[idx].isdigit():
            d_ff += checkpoint_name[idx]
            idx += 1
        config.d_ff = int(d_ff)
    else:
        config.d_ff = 32

    config.d_layers = 1
    config.dropout = 0.1
    config.embed = 'timeF'
    config.freq = freq
    config.activation = 'gelu'
    config.num_class = num_class

    # 解析 top_k
    if '_topk' in checkpoint_name:
        idx = checkpoint_name.find('_topk') + 5
        top_k = ''
        while idx < len(checkpoint_name) and checkpoint_name[idx].isdigit():
            top_k += checkpoint_name[idx]
            idx += 1
        config.top_k = int(top_k)
    else:
        config.top_k = 5

    config.num_kernels = num_kernels
    config.c_out = 1

    if config_override:
        for k, v in config_override.items():
            config[k] = v

    # 使用预计算的 period_list，或从样本输入计算
    if period_list is None:
        if sample_input is None:
            sample_input = torch.randn(1, config.seq_len, config.enc_in)
        period_list = compute_period_list(sample_input, k=config.top_k)

    print(f"Model config:")
    print(f"  seq_len: {config.seq_len}")
    print(f"  enc_in: {config.enc_in}")
    print(f"  d_model: {config.d_model}")
    print(f"  num_class: {config.num_class}")
    print(f"  top_k: {config.top_k}")
    print(f"  period_list: {period_list}")

    # 创建预计算周期的模型
    model = Model_Precomputed(config, period_list)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    return model, config, period_list


def export_to_onnx(model, config, output_path, opset_version=13):
    """导出模型到 ONNX 格式"""
    batch_size = 1
    seq_len = config.seq_len
    enc_in = config.enc_in

    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.ones(batch_size, seq_len)

    print(f"Exporting with input shapes: x_enc={x_enc.shape}, x_mark_enc={x_mark_enc.shape}")

    torch.onnx.export(
        model,
        (x_enc, x_mark_enc, None, None),
        output_path,
        input_names=['x_enc', 'x_mark_enc', 'x_dec', 'x_mark_dec'],
        output_names=['output'],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes={
            'x_enc': {0: 'batch_size', 1: 'seq_len'},
            'x_mark_enc': {0: 'batch_size', 1: 'seq_len'},
            'output': {0: 'batch_size'}
        },
        training=torch.onnx.TrainingMode.EVAL
    )

    print(f"ONNX model exported to: {output_path}")


def verify_onnx_model(onnx_path, pytorch_model, config, period_list):
    """验证 ONNX 模型"""
    print("\nVerifying ONNX model...")

    batch_size = 1
    seq_len = config.seq_len
    enc_in = config.enc_in

    # 使用相同的随机种子确保输入一致
    torch.manual_seed(42)
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        pytorch_output = pytorch_model(x_enc, x_mark_enc, None, None)

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {
        'x_enc': x_enc.numpy(),
        'x_mark_enc': x_mark_enc.numpy(),
        'x_dec': None,
        'x_mark_dec': None
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_output = ort_outputs[0]

    torch_output = pytorch_output.numpy()
    max_diff = float(np.max(np.abs(torch_output - onnx_output)))
    mean_diff = float(np.mean(np.abs(torch_output - onnx_output)))

    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("Verification PASSED!")
        return True
    else:
        print("Verification FAILED!")
        return False


def test_onnx_gpu(onnx_path, config, period_list):
    """测试 ONNX GPU 推理"""
    print("\nTesting ONNX GPU inference...")

    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")

    if 'CUDAExecutionProvider' not in providers:
        print("CUDAExecutionProvider not available!")
        return False

    batch_size = 1
    seq_len = config.seq_len
    enc_in = config.enc_in

    torch.manual_seed(42)
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.ones(batch_size, seq_len)

    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider'],
        provider_options=[{'device_id': 0}]
    )

    ort_inputs = {
        'x_enc': x_enc.numpy(),
        'x_mark_enc': x_mark_enc.numpy(),
        'x_dec': None,
        'x_mark_dec': None
    }
    _ = ort_session.run(None, ort_inputs)

    import time
    n_runs = 10
    start = time.time()
    for _ in range(n_runs):
        ort_outputs = ort_session.run(None, ort_inputs)
    end = time.time()

    avg_time = (end - start) / n_runs * 1000
    print(f"Average inference time (GPU): {avg_time:.2f} ms")

    output = ort_outputs[0]
    predicted_class = int(np.argmax(output[0]))
    print(f"Predicted class: {predicted_class}")

    return True


def run_accuracy_comparison(checkpoint_path, onnx_path, config, period_list, n_samples=50):
    """精度对比测试"""
    print("\n" + "="*60)
    print("Accuracy Comparison Test")
    print("="*60)

    batch_size = 1
    seq_len = config.seq_len
    enc_in = config.enc_in

    # 加载预计算周期的模型
    pytorch_model = Model_Precomputed(config, period_list)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pytorch_model.load_state_dict(checkpoint, strict=False)
    pytorch_model.eval()

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    gpu_session = None
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        gpu_session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider'],
            provider_options=[{'device_id': 0}]
        )

    print(f"\nTesting with {n_samples} samples...")
    print("-"*60)

    pytorch_correct = 0
    onnx_cpu_correct = 0
    onnx_gpu_correct = 0

    for i in range(n_samples):
        torch.manual_seed(i)
        x_enc = torch.randn(batch_size, seq_len, enc_in)
        x_mark_enc = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            pytorch_output = pytorch_model(x_enc, x_mark_enc, None, None)
        pytorch_pred = int(torch.argmax(pytorch_output, dim=1).item())

        ort_inputs = {
            'x_enc': x_enc.numpy(),
            'x_mark_enc': x_mark_enc.numpy(),
            'x_dec': None,
            'x_mark_dec': None
        }
        onnx_cpu_output = ort_session.run(None, ort_inputs)[0]
        onnx_cpu_pred = int(np.argmax(onnx_cpu_output[0]))

        if gpu_session:
            onnx_gpu_output = gpu_session.run(None, ort_inputs)[0]
            onnx_gpu_pred = int(np.argmax(onnx_gpu_output[0]))
        else:
            onnx_gpu_pred = onnx_cpu_pred

        pytorch_prob = torch.softmax(pytorch_output[0], dim=0).numpy()
        onnx_cpu_prob = onnx_cpu_output[0]

        max_diff = float(np.max(np.abs(pytorch_prob - onnx_cpu_prob)))

        if max_diff < 1e-4:
            pytorch_correct += 1
            onnx_cpu_correct += 1
            if gpu_session:
                onnx_gpu_correct += 1

        if (i + 1) % 10 == 0:
            print(f"  Sample {i+1}/{n_samples}: PyTorch vs ONNX CPU max_diff={max_diff:.6f}")

    print("-"*60)
    print(f"Results (output consistency with PyTorch):")
    print(f"  PyTorch:      {pytorch_correct}/{n_samples} ({100*pytorch_correct/n_samples:.1f}%)")
    print(f"  ONNX CPU:     {onnx_cpu_correct}/{n_samples} ({100*onnx_cpu_correct/n_samples:.1f}%)")
    if gpu_session:
        print(f"  ONNX GPU:     {onnx_gpu_correct}/{n_samples} ({100*onnx_gpu_correct/n_samples:.1f}%)")

    return pytorch_correct == n_samples


def load_sample_data(sample_data_path: str, seq_len: int, enc_in: int, for_period: bool = True):
    """
    加载真实航迹报文数据用于计算 period_list

    Args:
        sample_data_path: 航迹文件路径 (.xls 或 .csv, GBK 编码)
        seq_len: 序列长度（用于填充/截断）
        enc_in: 特征数
        for_period: 如果为 True，用原始数据长度计算 period_list（不填充）

    Returns:
        torch.Tensor: shape (1, seq_len, enc_in)
    """
    import pandas as pd

    if not os.path.exists(sample_data_path):
        raise FileNotFoundError(f"Sample data not found: {sample_data_path}")

    # 读取数据 (支持 .xls 和 .csv, GBK 编码)
    if sample_data_path.endswith('.xls') or sample_data_path.endswith('.csv'):
        df = pd.read_csv(sample_data_path, encoding='gbk', sep='\t' if sample_data_path.endswith('.xls') else ',')
    else:
        df = pd.read_csv(sample_data_path, encoding='gbk')

    # 删除 Unnamed 列
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in str(col)], errors='ignore')

    # 选择前 enc_in 列作为特征
    feature_cols = df.columns[:enc_in].tolist()

    # 转换为数值并处理缺失值
    df = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df = df.interpolate(limit_direction='both')
    df = df.ffill().bfill()

    # 转换为 numpy 数组
    data = df.values.astype(np.float32)

    # 如果用于计算 period_list，保持原始长度
    if for_period:
        print(f"Loaded sample data for period_list: shape={data.shape}, columns={feature_cols}")
        return torch.from_numpy(data).unsqueeze(0)

    # 否则填充/截断到 seq_len（用于推理）
    if len(data) > seq_len:
        data = data[-seq_len:]
    elif len(data) < seq_len:
        last_row = data[-1] if len(data) > 0 else np.zeros(enc_in, dtype=np.float32)
        padding = np.tile(last_row, (seq_len - len(data), 1))
        data = np.vstack([data, padding])

    print(f"Loaded sample data for inference: shape={data.shape}, columns={feature_cols}")
    return torch.from_numpy(data).unsqueeze(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Export TimesNet to ONNX')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/classification_TrajGBK_TimesNet_trajxls_ftM_sl20_ll48_pl0_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/checkpoint.pth',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX file path')
    parser.add_argument('--sample_data', type=str,
                        default='mydataset/radar_augv3/uav/P7_Sn3884171_win0_20.xls',
                        help='Path to sample trajectory file for period_list calculation (GBK encoded .xls/.csv)')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Top-k period value used during training (default: 3)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify ONNX model against PyTorch')
    parser.add_argument('--test-gpu', action='store_true',
                        help='Test ONNX GPU inference')
    parser.add_argument('--accuracy', action='store_true',
                        help='Run accuracy comparison test')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset version')

    args = parser.parse_args()

    if args.output is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        checkpoint_name = os.path.basename(checkpoint_dir)
        args.output = os.path.join(checkpoint_dir, f"{checkpoint_name}.onnx")

    print(f"Loading model from: {args.checkpoint}")

    # 从 checkpoint 解析配置，获取 seq_len 和 enc_in
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    proj_weight = checkpoint.get('projection.weight', None)
    num_class = proj_weight.shape[0] if proj_weight is not None else 2

    checkpoint_name = os.path.basename(os.path.dirname(args.checkpoint))
    if 'sl' in checkpoint_name:
        idx = checkpoint_name.find('sl') + 2
        seq_len = ''
        while idx < len(checkpoint_name) and checkpoint_name[idx].isdigit():
            seq_len += checkpoint_name[idx]
            idx += 1
        seq_len = int(seq_len)
    else:
        seq_len = 96

    # 从 checkpoint 权重提取 enc_in
    checkpoint_temp = torch.load(args.checkpoint, map_location='cpu')
    token_conv_weight = checkpoint_temp.get('enc_embedding.value_embedding.tokenConv.weight', None)
    if token_conv_weight is not None:
        enc_in = token_conv_weight.shape[1]
    else:
        enc_in = 1

    print(f"Model config: seq_len={seq_len}, enc_in={enc_in}, num_class={num_class}")

    # 加载真实报文数据用于计算周期（保持原始长度，不填充）
    if os.path.exists(args.sample_data):
        print(f"Using sample data: {args.sample_data}")
        period_input = load_sample_data(args.sample_data, seq_len, enc_in, for_period=True)
        period_list = compute_period_list(period_input, k=args.top_k)
        print(f"Period list from sample data (top_k={args.top_k}): {period_list}")

        # 准备用于模型加载的输入（填充到 seq_len）
        model_input = load_sample_data(args.sample_data, seq_len, enc_in, for_period=False)
    else:
        print(f"Sample data not found: {args.sample_data}, using random input")
        period_list = None
        model_input = torch.randn(1, seq_len, enc_in)

    # 使用 config_override 覆盖 top_k
    model, config, period_list = load_and_prepare_model(
        args.checkpoint,
        sample_input=model_input,
        period_list=period_list,
        config_override={'top_k': args.top_k}
    )

    print(f"\nExporting to ONNX: {args.output}")
    export_to_onnx(model, config, args.output, opset_version=args.opset)

    if args.verify:
        verify_onnx_model(args.output, model, config, period_list)

    if args.test_gpu:
        test_onnx_gpu(args.output, config, period_list)

    if args.accuracy:
        run_accuracy_comparison(args.checkpoint, args.output, config, period_list, n_samples=50)

    print("\nDone!")


if __name__ == '__main__':
    main()
