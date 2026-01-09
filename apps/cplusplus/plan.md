# C++ TimesNet 航迹预测重写计划

## 1 项目概述

将 `apps/udp_timesnet_predict.py` 重写为 C++ 实现，使用 ONNX Runtime GPU 进行模型推理，保留完整的组播接收-预测-发送功能。

## 2 目录结构

```
apps/cplusplus/
├── CMakeLists.txt              # CMake 构建配置
├── CMakePresets.json           # CMake 预设配置
├── src/
│   ├── main.cpp                # 主入口
│   ├── receiver/
│   │   ├── multicast_receiver.h
│   │   └── multicast_receiver.cpp      # 组播接收实现
│   ├── publisher/
│   │   ├── multicast_publisher.h
│   │   └── multicast_publisher.cpp     # 组播发送实现
│   ├── parser/
│   │   ├── packet_parser.h
│   │   └── packet_parser.cpp           # UDP 报文解析
│   ├── predictor/
│   │   ├── onnx_predictor.h
│   │   └── onnx_predictor.cpp          # ONNX 模型推理
│   ├── buffer/
│   │   ├── track_buffer.h
│   │   └── track_buffer.cpp            # 轨迹窗口缓冲
│   ├── normalizer/
│   │   ├── feature_normalizer.h
│   │   └── feature_normalizer.cpp      # 特征归一化
│   └── utils/
│       ├── logger.h
│       ├── json_helper.h               # JSON 辅助函数
│       └── timestamp.h
├── include/
│   └── (对外暴露的头文件)
├── tests/
│   ├── CMakeLists.txt                  # 测试构建配置
│   ├── test_receiver.cpp               # 组播接收单元测试
│   ├── test_publisher.cpp              # 组播发送单元测试
│   ├── test_parser.cpp                 # 报文解析单元测试
│   ├── test_predictor.cpp              # ONNX 推理单元测试
│   ├── test_buffer.cpp                 # 轨迹缓冲单元测试
│   └── test_normalizer.cpp             # 归一化单元测试
├── scripts/
│   ├── convert_to_onnx.py              # PyTorch -> ONNX 转换脚本
│   └── compare_accuracy.py              # ONNX vs PyTorch 精度对比
├── third_party/
│   ├── onnxruntime-linux-x64-gpu/      # ONNX Runtime GPU (下载)
│   └── (其他依赖)
├── models/                             # 模型目录
│   └── (onnx 模型文件)
├── data/
│   └── (测试数据、stats.json)
└── CMakeLists.txt                      # 根 CMake 配置
```

## 3 依赖项

| 依赖 | 版本 | 用途 |
|------|------|------|
| C++ | C++17 最低 | 现代语言特性 |
| CMake | 3.20+ | 构建系统 |
| ONNX Runtime | 1.22.0+ GPU | 模型推理 |
| googletest | 1.14.0 | 单元测试框架 |
| googlemock | 1.14.0 | Mock 框架 |
| nlohmann/json | 3.11.0 | JSON 解析 |
| Boost.Asio | 1.74+ | 异步网络IO |

## 4 核心模块设计

### 4.1 组播接收器 (MulticastReceiver)

```cpp
class MulticastReceiver {
public:
    struct Config {
        std::string group;
        int port;
        std::string iface;
        std::string bind_ip;
        double timeout_s;
    };

    explicit MulticastReceiver(const Config& config);
    bool open();
    std::optional<std::vector<uint8_t>> recv();
    void close();
};
```

### 4.2 组播发布器 (MulticastPublisher)

```cpp
class MulticastPublisher {
public:
    struct Config {
        std::string group;
        int port;
        std::string iface;
        int ttl;
    };

    explicit MulticastPublisher(const Config& config);
    bool open();
    bool send(const std::vector<PredictionResult>& results);
    void close();
};
```

### 4.3 报文解析器 (PacketParser)

```cpp
class PacketParser {
public:
    struct ParsedTarget {
        uint32_t batch_id;
        uint32_t track_id;
        double timestamp;
        // ... 其他字段
    };

    std::vector<ParsedTarget> parse(const uint8_t* data, size_t len);
};
```

### 4.4 ONNX 预测器 (OnnxPredictor)

```cpp
class OnnxPredictor {
public:
    struct Config {
        std::string model_path;
        int num_classes;
        int seq_len;
        int num_features;
        bool use_gpu;
        std::string gpu_device_id;
    };

    explicit OnnxPredictor(const Config& config);
    bool load();
    std::pair<std::vector<int>, std::vector<float>> predict(
        const std::vector<std::vector<float>>& batch,
        const std::vector<int>& lengths
    );
};
```

### 4.5 轨迹窗口缓冲 (TrackWindowBuffer)

```cpp
class TrackWindowBuffer {
public:
    struct Config {
        int seq_len;
        double max_age_s;
    };

    explicit TrackWindowBuffer(const Config& config);
    void update(uint32_t track_id, const std::vector<float>& features, double timestamp);
    std::optional<std::pair<std::vector<uint32_t>, BatchData>> get_batch();
    void cleanup();
};
```

## 5 脚本开发

### 5.1 PyTorch -> ONNX 转换脚本 (`scripts/convert_to_onnx.py`)

```python
#!/usr/bin/env python3
"""
将 TimesNet checkpoint.pth 转换为 ONNX 格式
"""
import argparse
import torch
import onnx
from pathlib import Path

def convert(
    checkpoint_path: str,
    output_path: str,
    seq_len: int = 20,
    num_features: int = 8,
    num_classes: int = 2,
    d_model: int = 64,
    n_heads: int = 8,
    d_ff: int = 256,
    e_layers: int = 2,
    d_layers: int = 1,
    top_k: int = 2,
    num_kernels: int = 6,
    dropout: float = 0.1,
    embed: str = "timeF",
    freq: str = "s",
):
    # 1. 加载 checkpoint
    # 2. 构建 TimesNet 模型
    # 3. 导出为 ONNX
    # 4. 验证导出模型
    pass
```

### 5.2 精度对比脚本 (`scripts/compare_accuracy.py`)

```python
#!/usr/bin/env python3
"""
对比 PyTorch 和 ONNX 模型推理结果
"""
import argparse
import numpy as np
import torch
import onnxruntime
from pathlib import Path

def compare(
    checkpoint_path: str,
    onnx_path: str,
    test_data_path: str,
    output_path: str = "accuracy_report.json",
):
    # 1. 加载 PyTorch 模型
    # 2. 加载 ONNX 模型
    # 3. 生成/加载测试数据
    # 4. 推理并对比
    # 5. 输出报告: MAE, RMSE, 分类准确率, 预测差异分布
    pass
```

## 6 单元测试计划

### 6.1 测试框架

- **GoogleTest (GTest)**: 基础单元测试框架
- **GoogleMock (GMock)**: Mock 网络组件和依赖

### 6.2 测试用例

#### 6.2.1 组播接收器测试 (`test_receiver.cpp`)

| 用例ID | 测试内容 | Mock方式 |
|--------|----------|----------|
| T-REC-001 | 正常接收组播报文 | Mock UDP Socket |
| T-REC-002 | 超时返回空 | Mock select() 超时 |
| T-REC-003 | 校验和校验失败 | Mock 校验计算 |
| T-REC-004 | 绑定地址配置 | Mock bind() |
| T-REC-005 | 错误的组播地址 | Mock 错误场景 |

#### 6.2.2 组播发布器测试 (`test_publisher.cpp`)

| 用例ID | 测试内容 | Mock方式 |
|--------|----------|----------|
| T-PUB-001 | 发送预测结果 | Mock UDP Socket |
| T-PUB-002 | TTL 配置生效 | Mock setsockopt() |
| T-PUB-003 | 发送失败处理 | Mock sendto() 错误 |
| T-PUB-004 | 空数据不发 | 边界测试 |

#### 6.2.3 报文解析器测试 (`test_parser.cpp`)

| 用例ID | 测试内容 | Mock方式 |
|--------|----------|----------|
| T-PAR-001 | 解析有效报文 | 无需 Mock |
| T-PAR-002 | 解析空报文 | 边界测试 |
| T-PAR-003 | 解析损坏报文 | 无需 Mock |
| T-PAR-004 | 多目标报文 | 无需 Mock |
| T-PAR-005 | 字段边界值 | 无需 Mock |

#### 6.2.4 ONNX 预测器测试 (`test_predictor.cpp`)

| 用例ID | 测试内容 | Mock方式 |
|--------|----------|----------|
| T-PRE-001 | 加载有效模型 | Mock 文件系统 |
| T-PRE-002 | 加载无效模型 | Mock 文件系统 |
| T-PRE-003 | 单样本推理 | Mock ONNX Runtime |
| T-PRE-004 | 批量推理 | Mock ONNX Runtime |
| T-PRE-005 | GPU 推理回退 CPU | Mock ONNX Session |
| T-PRE-006 | 输入形状校验 | 无需 Mock |

#### 6.2.5 轨迹缓冲测试 (`test_buffer.cpp`)

| 用例ID | 测试内容 | Mock方式 |
|--------|----------|----------|
| T-BUF-001 | 添加轨迹点 | 无需 Mock |
| T-BUF-002 | 构建批次 | 无需 Mock |
| T-BUF-003 | 过期清理 | 无需 Mock |
| T-BUF-004 | 多目标缓冲 | 无需 Mock |
| T-BUF-005 | 序列长度不足 | 无需 Mock |

#### 6.2.6 归一化测试 (`test_normalizer.cpp`)

| 用例ID | 测试内容 | Mock方式 |
|--------|----------|----------|
| T-NOR-001 | 正常归一化 | 无需 Mock |
| T-NOR-002 | 加载 stats.json | Mock 文件系统 |
| T-NOR-003 | 缺失字段处理 | 无需 Mock |
| T-NOR-004 | 边界值处理 | 无需 Mock |

### 6.3 测试覆盖率要求

- **行覆盖率**: >= 85%
- **分支覆盖率**: >= 80%
- **关键路径**: 100% (组播收发、模型推理、预测发送)

## 7 精度对比验证

### 7.1 对比矩阵

| 对比项 | PyTorch | Python ONNX | C++ ONNX GPU |
|--------|---------|-------------|--------------|
| PyTorch vs Python ONNX | ✓ | ✓ | - |
| PyTorch vs C++ ONNX GPU | ✓ | - | ✓ |
| Python ONNX vs C++ ONNX GPU | - | ✓ | ✓ |

### 7.2 对比指标

| 指标 | 计算方式 | 阈值 |
|------|----------|------|
| 分类准确率 | 一致预测比例 | >= 99.5% |
| 概率误差 MAE | \|p_onnx - p_pytorch\| | <= 0.01 |
| 概率误差 RMSE | sqrt(mean((p_onnx - p_pytorch)^2)) | <= 0.02 |
| 最大绝对误差 | max\|p_onnx - p_pytorch\| | <= 0.05 |

### 7.3 测试数据

- 使用 Python 脚本生成的测试报文集
- 至少 1000 条不同轨迹的报文
- 覆盖不同类别分布 (uav/bird)

### 7.4 C++ vs Python ONNX 对比脚本

```python
#!/usr/bin/env python3
"""
对比 C++ ONNX Runtime GPU 与 Python ONNX Runtime 推理结果
"""
import argparse
import numpy as np
import onnxruntime as ort

def compare_cpp_python(
    onnx_path: str,
    cpp_output_path: str,      # C++ 推理输出的 JSON
    python_output_path: str,    # Python ONNX 推理输出
    output_path: str = "cpp_python_report.json",
):
    # 1. 加载 Python ONNX 模型
    # 2. 加载 C++ 推理结果 (JSON格式: track_id, probs)
    # 3. 加载 Python 推理结果
    # 4. 逐条对比概率值
    # 5. 输出报告
    pass
```

### 7.5 C++ 推理结果输出格式

C++ ONNX 预测器需支持导出推理结果为 JSON 格式供对比：

```json
{
  "inference_id": 1,
  "timestamp_ms": 1234567890123,
  "results": [
    {
      "track_id": 1001,
      "pred": 1,
      "prob_uav": 0.9234,
      "prob_bird": 0.0766
    },
    {
      "track_id": 1002,
      "pred": 0,
      "prob_uav": 0.0456,
      "prob_bird": 0.9544
    }
  ]
}
```

## 8 实施步骤

### Phase 1: 基础架构
1. [ ] 创建 CMake 项目结构
2. [ ] 配置 ONNX Runtime GPU 依赖
3. [ ] 配置 GoogleTest/GoogleMock
4. [ ] 实现基础工具类 (Logger, JSON)

### Phase 2: 核心模块
5. [ ] 实现组播接收器 (含单元测试)
6. [ ] 实现组播发布器 (含单元测试)
7. [ ] 实现报文解析器 (含单元测试)
8. [ ] 实现特征归一化 (含单元测试)

### Phase 3: 模型推理
9. [ ] 实现 ONNX 预测器 (含单元测试)
10. [ ] 编写 convert_to_onnx.py 脚本
11. [ ] 编写 compare_accuracy.py 脚本
12. [ ] 验证 ONNX 与 PyTorch 精度一致

### Phase 4: 业务逻辑
13. [ ] 实现轨迹窗口缓冲 (含单元测试)
14. [ ] 实现主流程编排
15. [ ] 集成测试

### Phase 5: 验收
16. [ ] 端到端测试
17. [ ] 性能测试
18. [ ] 文档完善

## 9 构建命令

```bash
# 创建构建目录
mkdir build && cd build

# 配置 (Linux GPU)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DUSE_GPU=ON \
         -DONNXRuntime_DIR=../third_party/onnxruntime-linux-x64-gpu/lib/cmake/onnxruntime

# 构建
cmake --build . --config Release -j$(nproc)

# 运行测试
ctest --output-on-failure --test-dir .

# 精度对比
python ../scripts/compare_accuracy.py --checkpoint ../models/checkpoint.pth \
                                     --onnx ../models/model.onnx \
                                     --output accuracy_report.json
```
