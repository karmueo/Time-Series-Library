#!/usr/bin/env python3
"""
测试程序：加载保存的输入向量并进行推理

功能与 C++ 版本的 test_saved_input 相同

使用方法:
    python test_saved_input.py <input.json> <model.onnx> [OPTIONS]

示例:
    python test_saved_input.py apps/cplusplus/tmp/input_20260108_175826_372_0.json apps/cplusplus/models/timesnet.onnx
    python test_saved_input.py input.json model.onnx --num_features 14 --use_gpu
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

try:
    import numpy as np
    import onnxruntime as ort
except ImportError:
    print("错误: 需要安装 numpy 和 onnxruntime")
    print("请运行: pip install numpy onnxruntime-gpu  # 或 onnxruntime")
    sys.exit(1)


def load_input_from_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    从 JSON 文件加载输入数据

    Args:
        filepath: JSON 文件路径

    Returns:
        (data, lengths, track_ids)
        - data: 输入数据数组 [batch, seq_len, num_features]
        - lengths: 序列长度列表
        - track_ids: 轨迹ID列表
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        # 读取数据
        batch_size = input_data.get("batch_size", 0)
        seq_len = input_data.get("seq_len", 0)
        num_features = input_data.get("num_features", 0)
        track_ids = input_data.get("track_ids", [])
        lengths = input_data.get("lengths", [])
        data = input_data.get("data", [])

        # 转换为 numpy 数组
        data_array = np.array(data, dtype=np.float32)

        print(f"已加载输入数据: {filepath}")
        print(f"  batch_size: {batch_size}")
        print(f"  seq_len: {seq_len}")
        print(f"  num_features: {num_features}")
        print(f"  track_ids: {track_ids}")
        print()

        return data_array, np.array(lengths, dtype=np.int64), track_ids

    except FileNotFoundError:
        print(f"错误: 文件不存在: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 加载数据失败: {e}")
        sys.exit(1)


def create_onnx_session(
    model_path: str,
    use_gpu: bool = False,
    gpu_device_id: int = 0
) -> ort.InferenceSession:
    """
    创建 ONNX Runtime 推理会话

    Args:
        model_path: ONNX 模型路径
        use_gpu: 是否使用 GPU
        gpu_device_id: GPU 设备 ID

    Returns:
        ONNX Runtime 推理会话
    """
    try:
        # 设置提供者
        if use_gpu:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': gpu_device_id
                }),
                'CPUExecutionProvider'  # 回退到 CPU
            ]
        else:
            providers = ['CPUExecutionProvider']

        # 创建会话
        session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        # 打印实际使用的提供者
        print(f"ONNX 模型已加载: {model_path}")
        print(f"使用的提供者: {session.get_providers()}")

        # 打印模型信息
        input_info = session.get_inputs()
        output_info = session.get_outputs()

        print(f"模型有 {len(input_info)} 个输入:")
        for i, info in enumerate(input_info):
            print(f"  输入 {i}: {info.name} shape: {info.shape}")

        print(f"模型有 {len(output_info)} 个输出:")
        for i, info in enumerate(output_info):
            print(f"  输出 {i}: {info.name} shape: {info.shape}")
        print()

        return session

    except Exception as e:
        print(f"错误: 加载 ONNX 模型失败: {e}")
        sys.exit(1)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax 函数

    Args:
        x: 输入数组 [batch, num_classes]

    Returns:
        Softmax 后的概率数组 [batch, num_classes]
    """
    # 减去最大值以提高数值稳定性
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def run_inference(
    session: ort.InferenceSession,
    data: np.ndarray,
    lengths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    运行推理

    Args:
        session: ONNX Runtime 会话
        data: 输入数据 [batch, seq_len, num_features]
        lengths: 序列长度 [batch]

    Returns:
        (predictions, probabilities)
        - predictions: 预测类别 [batch]
        - probabilities: 无人机概率 [batch]
    """
    try:
        batch_size, seq_len, num_features = data.shape

        # 准备输入
        # x_enc: [batch, seq_len, num_features]
        x_enc = data.astype(np.float32)

        # x_mark_enc: [batch, seq_len] (掩码，全为1)
        x_mark_enc = np.ones((batch_size, seq_len), dtype=np.float32)

        # 获取输入输出名称
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        # 准备输入字典
        inputs = {}
        if 'x_enc' in input_names:
            inputs['x_enc'] = x_enc
        elif len(input_names) > 0:
            inputs[input_names[0]] = x_enc

        if 'x_mark_enc' in input_names:
            inputs['x_mark_enc'] = x_mark_enc
        elif len(input_names) > 1:
            inputs[input_names[1]] = x_mark_enc

        # 运行推理
        start_time = time.time()
        outputs = session.run(output_names, inputs)
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒

        # 处理输出
        if len(outputs) == 0:
            print("错误: 模型没有输出")
            sys.exit(1)

        logits = outputs[0]  # [batch, num_classes]

        # 使用 softmax 将 logits 转换为概率
        probs = softmax(logits)  # [batch, num_classes]

        # 获取预测类别和概率
        predictions = np.argmax(probs, axis=1)  # [batch]
        probabilities = probs[:, 1]  # 取第1列（无人机概率）

        return predictions, probabilities, inference_time

    except Exception as e:
        print(f"错误: 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_results(
    track_ids: List[int],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    inference_time: float
):
    """
    打印推理结果

    Args:
        track_ids: 轨迹ID列表
        predictions: 预测类别
        probabilities: 无人机概率
        inference_time: 推理时间（毫秒）
    """
    print("=" * 50)
    print("推理结果")
    print("=" * 50)
    print(f"推理时间: {inference_time:.3f} ms")
    print(f"批次大小: {len(predictions)}")
    print()

    for i, track_id in enumerate(track_ids):
        pred = predictions[i]
        prob_uav = probabilities[i]
        prob_bird = 1.0 - prob_uav

        label = "UAV" if pred == 1 else "Bird"

        print(f"Track {track_id}:")
        print(f"  预测: {pred} ({label})")
        print(f"  无人机概率: {prob_uav:.6f}")
        print(f"  鸟类概率: {prob_bird:.6f}")
        print()


def create_json_output(
    track_ids: List[int],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    inference_time: float
) -> Dict[str, Any]:
    """
    创建 JSON 格式的输出

    Args:
        track_ids: 轨迹ID列表
        predictions: 预测类别
        probabilities: 无人机概率
        inference_time: 推理时间（毫秒）

    Returns:
        JSON 输出字典
    """
    results = []
    for i, track_id in enumerate(track_ids):
        pred = int(predictions[i])
        prob_uav = float(probabilities[i])
        prob_bird = 1.0 - prob_uav

        results.append({
            "track_id": int(track_id),
            "prediction": pred,
            "prediction_label": "UAV" if pred == 1 else "Bird",
            "prob_uav": prob_uav,
            "prob_bird": prob_bird
        })

    output = {
        "inference_time_ms": float(inference_time),
        "batch_size": len(predictions),
        "results": results
    }

    return output


def main():
    parser = argparse.ArgumentParser(
        description='测试程序：加载保存的输入向量并进行推理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s input.json model.onnx
  %(prog)s input.json model.onnx --num_features 14
  %(prog)s input.json model.onnx --use_gpu
  %(prog)s input.json model.onnx --use_gpu --gpu_device_id 1
        """
    )

    parser.add_argument('input_json', help='输入 JSON 文件路径')
    parser.add_argument('model_onnx', help='ONNX 模型文件路径')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数 (默认: 2)')
    parser.add_argument('--num_features', type=int, default=14, help='特征数 (默认: 14)')
    parser.add_argument('--seq_len', type=int, default=20, help='序列长度 (默认: 20)')
    parser.add_argument('--use_gpu', action='store_true', help='使用 GPU 推理')
    parser.add_argument('--gpu_device_id', type=int, default=0, help='GPU 设备 ID (默认: 0)')

    args = parser.parse_args()

    print("=" * 50)
    print("测试保存的输入向量")
    print("=" * 50)
    print(f"输入文件: {args.input_json}")
    print(f"模型文件: {args.model_onnx}")
    print(f"num_classes: {args.num_classes}")
    print(f"num_features: {args.num_features}")
    print(f"seq_len: {args.seq_len}")
    print(f"use_gpu: {args.use_gpu}")
    print()

    # 1. 加载输入数据
    data, lengths, track_ids = load_input_from_file(args.input_json)

    # 2. 创建 ONNX 会话
    session = create_onnx_session(
        args.model_onnx,
        use_gpu=args.use_gpu,
        gpu_device_id=args.gpu_device_id
    )

    # 3. 运行推理
    print("=" * 50)
    print("运行推理")
    print("=" * 50)
    print()

    predictions, probabilities, inference_time = run_inference(session, data, lengths)

    # 4. 打印结果
    print_results(track_ids, predictions, probabilities, inference_time)

    # 5. JSON 输出
    json_output = create_json_output(track_ids, predictions, probabilities, inference_time)
    print("=" * 50)
    print("JSON 输出")
    print("=" * 50)
    print(json.dumps(json_output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
