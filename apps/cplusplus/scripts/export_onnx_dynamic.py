"""
导出支持动态 period_list 的 ONNX 模型

关键改进：
1. 模型在 forward 时动态计算 period_list（使用 FFT）
2. 使用 ONNX 兼容的动态 reshape
3. 无需预先固定 period_list
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

# 添加项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # 从 scripts 目录往上两级到项目根目录
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.TimesNet import (
    Model,
    TimesBlock,
    Inception_Block_V1,
    DataEmbedding,
    FFT_for_Period,
)


class TimesBlock_Dynamic(nn.Module):
    """支持动态 period_list 的 TimesBlock（ONNX 兼容）"""

    def __init__(self, configs):
        super(TimesBlock_Dynamic, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, N]

        Returns:
            res: [B, T, N]
        """
        B, T, N = x.size()

        # 动态计算 period_list（使用 FFT）
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # 避免 period=0
            period = torch.where(period == 0, torch.tensor(1, device=period.device), period)

            # 动态计算 padding
            total_len = self.seq_len + self.pred_len

            # 计算需要的长度
            length = total_len + (period - total_len % period) % period

            # Padding
            if length > total_len:
                padding_len = length - total_len
                padding = torch.zeros([x.shape[0], padding_len, x.shape[2]], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x

            # 动态 reshape - 关键：使用 ONNX 兼容的方式
            # out: [B, length, N] -> [B, length // period, period, N]
            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2).contiguous()

            # 2D convolution
            out = self.conv(out)

            # Reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total_len, :])

        # Stack and weight
        res = torch.stack(res, dim=-1)  # [B, T, N, k]

        # period_weight: [B, k] -> [B, 1, 1, k]
        period_weight = period_weight.unsqueeze(1).unsqueeze(1)

        # Weighted sum
        res = (res * period_weight).sum(dim=-1)  # [B, T, N]

        # Residual connection
        res = res + x
        return res


class Model_Dynamic(Model):
    """
    支持动态 period_list 的 TimesNet 模型

    继承原始 Model 类，只替换 TimesBlock
    """

    def __init__(self, configs):
        super(Model_Dynamic, self).__init__(configs)

        # 替换为动态 TimesBlock
        self.model = nn.ModuleList(
            [TimesBlock_Dynamic(configs) for _ in range(configs.e_layers)]
        )


def load_and_prepare_model(checkpoint_path, config_override=None):
    """加载模型并准备导出"""
    print(f"Loading model from: {checkpoint_path}")

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 获取配置
    config = {
        "task_name": "classification",  # 重要！必须设置任务类型
        "seq_len": 20,
        "enc_in": 14,
        "num_class": 2,
        "d_model": 128,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 256,
        "top_k": 3,
        "num_kernels": 6,
        "embed": "timeF",
        "freq": "s",
        "label_len": 48,
        "pred_len": 0,
        "dropout": 0.1,
        "c_out": 14,  # 添加这个配置
    }

    # 覆盖配置
    if config_override:
        for k, v in config_override.items():
            config[k] = v

    print(f"Model config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 创建动态模型
    from utils.tools import dotdict

    args = dotdict(config)
    model = Model_Dynamic(args).float()

    # 加载权重
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    return model, config


def export_to_onnx(model, config, output_path, opset_version=17):
    """导出 ONNX 模型（支持动态 period_list）"""
    print(f"\nExporting to ONNX: {output_path}")
    print(f"Using opset version: {opset_version}")

    # 准备示例输入
    batch_size = 1
    x_enc = torch.randn(batch_size, config["seq_len"], config["enc_in"])
    x_mark_enc = torch.randn(batch_size, config["seq_len"])

    # 导出
    torch.onnx.export(
        model,
        (x_enc, x_mark_enc, None, None),
        output_path,
        input_names=["x_enc", "x_mark_enc"],
        output_names=["output"],
        dynamic_axes={
            "x_enc": {0: "batch_size"},
            "x_mark_enc": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )

    print(f"✅ ONNX model exported to: {output_path}")

    # 验证模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX model validation passed")

    return True


def verify_onnx_model(onnx_path, pytorch_model, config):
    """验证 ONNX 模型精度"""
    print("\n" + "=" * 60)
    print("Verifying ONNX model...")
    print("=" * 60)

    # 准备测试数据
    batch_size = 1
    x_enc = torch.randn(batch_size, config["seq_len"], config["enc_in"])
    x_mark_enc = torch.randn(batch_size, config["seq_len"])

    # PyTorch 推理
    with torch.no_grad():
        pytorch_output = pytorch_model(x_enc, x_mark_enc, None, None)

    # ONNX 推理
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"x_enc": x_enc.numpy(), "x_mark_enc": x_mark_enc.numpy()})[0]

    # 对比
    pytorch_output = pytorch_output.numpy()
    diff = np.abs(pytorch_output - onnx_output)

    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"Max difference: {diff.max():.6f}")
    print(f"Mean difference: {diff.mean():.6f}")

    if diff.max() < 1e-3:
        print("✅ Verification PASSED!")
        return True
    else:
        print("❌ Verification FAILED!")
        return False


def test_dynamic_period(onnx_path, config, num_samples=5):
    """测试动态 period_list 功能"""
    print("\n" + "=" * 60)
    print("Testing dynamic period_list...")
    print("=" * 60)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    for i in range(num_samples):
        # 生成随机输入（模拟不同数据）
        x_enc = torch.randn(1, config["seq_len"], config["enc_in"])
        x_mark_enc = torch.randn(1, config["seq_len"])

        # ONNX 推理
        onnx_output = session.run(
            None, {"x_enc": x_enc.numpy(), "x_mark_enc": x_mark_enc.numpy()}
        )[0]

        prob = np.exp(onnx_output) / np.exp(onnx_output).sum(axis=1, keepdims=True)
        pred = np.argmax(prob, axis=1)

        print(f"\nSample {i + 1}:")
        print(f"  Prediction: {pred[0]}")
        print(f"  Probabilities: {prob[0]}")

    print("\n✅ Dynamic period_list test completed!")


def main():
    parser = argparse.ArgumentParser(description="导出支持动态 period_list 的 ONNX 模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint 文件路径")
    parser.add_argument("--output", type=str, default="timesnet_dynamic.onnx", help="输出 ONNX 文件")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    parser.add_argument("--verify", action="store_true", help="验证 ONNX 模型")
    parser.add_argument("--test", action="store_true", help="测试动态 period 功能")
    parser.add_argument("--seq_len", type=int, default=20, help="序列长度")
    parser.add_argument("--enc_in", type=int, default=14, help="特征数")
    parser.add_argument("--num_class", type=int, default=2, help="类别数")

    args = parser.parse_args()

    print("=" * 60)
    print("导出支持动态 period_list 的 ONNX 模型")
    print("=" * 60)

    # 配置覆盖
    config_override = {"seq_len": args.seq_len, "enc_in": args.enc_in, "num_class": args.num_class}

    # 加载模型
    model, config = load_and_prepare_model(args.checkpoint, config_override)

    # 导出 ONNX
    export_to_onnx(model, config, args.output, opset_version=args.opset)

    # 验证
    if args.verify:
        verify_onnx_model(args.output, model, config)

    # 测试动态 period
    if args.test:
        test_dynamic_period(args.output, config, num_samples=5)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
