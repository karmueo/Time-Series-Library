#!/usr/bin/env python3
"""
验证固定 period_list 方案

对比三种推理方式的结果：
1. 原始 PyTorch 模型（动态 period_list）
2. 固定 period_list 的 PyTorch 模型
3. ONNX 模型（固定 period_list）
"""

import sys
from pathlib import Path

import numpy as np
import torch

# 添加项目根目录到 Python 路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.predictor import TimesNetPredictor
from core.predictor_fixed_period import TimesNetPredictorFixedPeriod
from core.predictor import OnnxTimesNetPredictor


def load_test_data(file_path):
    """加载测试数据"""
    import pandas as pd

    # 使用 GBK 编码读取文件
    df = pd.read_csv(
        file_path,
        sep="\t" if file_path.endswith(".xls") else ",",
        encoding="gbk",
        engine="python",
    )
    data = df.values.astype(np.float32)
    return data[:20]  # 取前20个点


def main():
    print("=" * 70)
    print("验证固定 period_list 方案")
    print("=" * 70)

    # 配置
    checkpoint_path = (
        "checkpoints/classification_TrajGBK_TimesNet_trajxls_ftM_sl20_ll48_pl0_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/checkpoint.pth"
    )
    onnx_path = "apps/cplusplus/models/timesnet.onnx"
    test_file = "mydataset/radar_augv3/uav/P1005_Sn955698_win0_20.xls"

    # ONNX 导出时使用的 period_list
    ONNX_PERIOD_LIST = [10, 2, 1]

    print(f"\n配置:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  ONNX 模型: {onnx_path}")
    print(f"  测试数据: {test_file}")
    print(f"  ONNX period_list: {ONNX_PERIOD_LIST}")

    # 加载测试数据
    print(f"\n加载测试数据...")
    data = load_test_data(test_file)
    batch = np.expand_dims(data, axis=0)  # [1, 20, 14]
    lengths = np.array([20])

    # 模型配置
    model_cfg = {
        "seq_len": 20,
        "label_len": 48,
        "pred_len": 0,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 256,
        "top_k": 3,
        "num_kernels": 6,
        "e_layers": 2,
        "d_layers": 1,
        "embed": "timeF",
        "freq": "s",
        "dropout": 0.1,
    }

    # 1. 原始 PyTorch 模型（动态 period_list）
    print(f"\n{'='*70}")
    print(f"[1/3] 原始 PyTorch 模型（动态 period_list）")
    print(f"{'='*70}")

    predictor_pytorch = TimesNetPredictor(
        model_path=checkpoint_path,
        model_name="TimesNet",
        num_classes=2,
        device="cpu",
        model_cfg=model_cfg,
    )
    predictor_pytorch.load(num_features=14)
    pred_pytorch, prob_pytorch = predictor_pytorch.predict(batch, lengths)

    print(f"预测结果:")
    print(f"  类别: {pred_pytorch[0]}")
    print(f"  概率: UAV={prob_pytorch[0][1]:.6f}, Bird={prob_pytorch[0][0]:.6f}")

    # 2. 固定 period_list 的 PyTorch 模型
    print(f"\n{'='*70}")
    print(f"[2/3] 固定 period_list 的 PyTorch 模型")
    print(f"{'='*70}")

    predictor_fixed = TimesNetPredictorFixedPeriod(
        model_path=checkpoint_path,
        period_list=ONNX_PERIOD_LIST,
        model_name="TimesNet",
        num_classes=2,
        device="cpu",
        model_cfg=model_cfg,
    )
    predictor_fixed.load(num_features=14)
    pred_fixed, prob_fixed = predictor_fixed.predict(batch, lengths)

    print(f"预测结果:")
    print(f"  类别: {pred_fixed[0]}")
    print(f"  概率: UAV={prob_fixed[0][1]:.6f}, Bird={prob_fixed[0][0]:.6f}")

    # 3. ONNX 模型
    print(f"\n{'='*70}")
    print(f"[3/3] ONNX 模型（固定 period_list）")
    print(f"{'='*70}")

    predictor_onnx = OnnxTimesNetPredictor(model_path=onnx_path, device="cpu")
    predictor_onnx.load(num_features=14)
    pred_onnx, prob_onnx = predictor_onnx.predict(batch, lengths)

    print(f"预测结果:")
    print(f"  类别: {pred_onnx[0]}")
    print(f"  概率: UAV={prob_onnx[0][1]:.6f}, Bird={prob_onnx[0][0]:.6f}")

    # 对比结果
    print(f"\n{'='*70}")
    print(f"结果对比")
    print(f"{'='*70}")

    print(f"\n1. 原始 PyTorch vs ONNX:")
    print(f"   类别一致: {pred_pytorch[0] == pred_onnx[0]}")
    diff_pytorch_onnx = np.abs(prob_pytorch - prob_onnx)
    print(f"   概率差异: UAV={diff_pytorch_onnx[0][1]:.6f}, Bird={diff_pytorch_onnx[0][0]:.6f}")

    print(f"\n2. 固定 period PyTorch vs ONNX:")
    print(f"   类别一致: {pred_fixed[0] == pred_onnx[0]}")
    diff_fixed_onnx = np.abs(prob_fixed - prob_onnx)
    print(f"   概率差异: UAV={diff_fixed_onnx[0][1]:.6f}, Bird={diff_fixed_onnx[0][0]:.6f}")

    print(f"\n3. 原始 PyTorch vs 固定 period PyTorch:")
    print(f"   类别一致: {pred_pytorch[0] == pred_fixed[0]}")
    diff_pytorch_fixed = np.abs(prob_pytorch - prob_fixed)
    print(f"   概率差异: UAV={diff_pytorch_fixed[0][1]:.6f}, Bird={diff_pytorch_fixed[0][0]:.6f}")

    # 结论
    print(f"\n{'='*70}")
    print(f"结论")
    print(f"{'='*70}")

    if diff_fixed_onnx[0][1] < 0.001:
        print(f"✅ 固定 period_list 方案成功！")
        print(f"   PyTorch(固定) 与 ONNX 的概率差异 < 0.001")
        print(f"   可以在 Python 脚本中使用 TimesNetPredictorFixedPeriod")
    else:
        print(f"❌ 固定 period_list 方案失败")
        print(f"   PyTorch(固定) 与 ONNX 的概率差异过大")

    print(f"\n如果需要使用固定 period_list，请修改 apps/udp_timesnet_predict.py")
    print(f"将 TimesNetPredictor 替换为 TimesNetPredictorFixedPeriod")


if __name__ == "__main__":
    main()
