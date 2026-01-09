# 保存推理输入向量功能说明

## 概述

本功能允许在推理前保存输入模型的向量数据，以便进行离线测试和调试。

**重要**：`save_input_path` 应该设置为**文件夹路径**，每一批送入模型的向量会保存为独立的文件。

## 配置方式

在 `config.yaml` 中添加 `save_input_path` 配置项：

```yaml
# 推理配置
inference:
  min_seq_len: 20
  window_step: 0
  ema_alpha: 0.4
  publish_interval_ms: 0
  print_targets: false
  print_features: false
  save_input_path: "apps/cplusplus/tmp"  # 保存推理输入向量的文件夹路径
```

- **设置为文件夹路径**：每一批推理的输入向量会保存为独立的文件
- **设置为空或注释掉**：不保存输入向量
- **文件夹会自动创建**（如果不存在）

## 文件命名规则

每批输入向量会保存为独立的 JSON 文件，文件名格式：

```
input_YYYYMMDD_HHMMSS_mmm_counter.json
```

例如：
- `input_20260108_175230_123_0.json` - 第 1 批
- `input_20260108_175235_456_1.json` - 第 2 批
- `input_20260108_175240_789_2.json` - 第 3 批

其中：
- `YYYYMMDD_HHMMSS`: 日期和时间
- `mmm`: 毫秒数（3位）
- `counter`: 批次计数器（从 0 开始）

## 使用流程

### 1. 保存输入向量

运行主程序，配置 `save_input_path`，当累积 20 个点进行推理时，输入向量会自动保存：

```bash
./TimesNetPredictor --config config.yaml
```

保存的 JSON 文件格式：

```json
{
  "batch_size": 1,
  "seq_len": 20,
  "num_features": 14,
  "track_ids": [12345],
  "lengths": [20],
  "data": [[
    [0.5, 0.3, ...],  // 第 1 个时间步的特征
    [0.6, 0.4, ...],  // 第 2 个时间步的特征
    ...
  ]]
}
```

### 2. 使用测试程序进行推理

使用 `test_saved_input` 程序加载保存的向量并进行推理：

```bash
./test_saved_input <input.json> <model.onnx> [OPTIONS]
```

**参数说明：**
- `input.json`: 保存的输入向量文件
- `model.onnx`: ONNX 模型文件路径
- `--num_classes N`: 类别数 (默认: 2)
- `--num_features N`: 特征数 (默认: 14)
- `--seq_len LEN`: 序列长度 (默认: 20)
- `--use_gpu`: 使用 GPU 推理
- `--gpu_device_id ID`: GPU 设备 ID (默认: 0)

**示例：**

```bash
# 基本用法
./test_saved_input apps/cplusplus/tmp/input_20260108_175230_123_0.json models/timesnet.onnx

# 指定特征数
./test_saved_input apps/cplusplus/tmp/input_20260108_175230_123_0.json models/timesnet.onnx --num_features 14

# 使用 GPU
./test_saved_input apps/cplusplus/tmp/input_20260108_175230_123_0.json models/timesnet.onnx --use_gpu
```

**输出示例：**

```
=== Test Saved Input ===
Input file: apps/cplusplus/tmp/input_20260108_175230_123_0.json
Model file: models/timesnet.onnx
num_classes: 2
num_features: 14
seq_len: 20
use_gpu: false

Loaded input data from: apps/cplusplus/tmp/input_20260108_175230_123_0.json
  batch_size: 1
  seq_len: 20
  num_features: 14
  track_ids: 12345

=== Running Inference ===

=== Inference Results ===
Inference time: 12.345 ms
Batch size: 1

Track 12345:
  Prediction: 1 (UAV)
  Prob(UAV): 0.8765
  Prob(Bird): 0.1235

=== JSON Output ===
{
  "inference_time_ms": 12.345,
  "batch_size": 1,
  "results": [
    {
      "track_id": 12345,
      "prediction": 1,
      "prediction_label": "UAV",
      "prob_uav": 0.8765,
      "prob_bird": 0.1235
    }
  ]
}
```

## 应用场景

1. **离线调试**：保存实际运行时的输入数据，用于离线分析模型行为
2. **性能测试**：使用相同的输入数据测试不同模型的性能
3. **问题复现**：保存异常情况的输入数据，方便后续调试
4. **模型验证**：验证 ONNX 模型转换的正确性
5. **批量测试**：保存多批数据，用于批量测试和对比

## 注意事项

1. **文件夹模式**：`save_input_path` 必须是文件夹路径，不是文件路径
2. **持续保存**：每一批推理的输入向量都会保存，不会自动停止
3. **归一化特征**：保存的是归一化后的特征向量
4. **磁盘空间**：长时间运行会生成大量文件，注意监控磁盘空间
5. **文件大小**：每个文件大小取决于 `batch_size * seq_len * num_features`

## 构建说明

```bash
cd apps/cplusplus

# 配置
cmake --preset linux-debug

# 构建
cmake --build build-debug -j4

# 测试程序位于
./build-debug/test_saved_input
```
