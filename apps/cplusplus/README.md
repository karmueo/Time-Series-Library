# TimesNet Predictor C++

C++ 实现的航迹识别预测器，支持组播接收、ONNX 推理、组播发送。

## 目录结构

```
cplusplus/
├── CMakeLists.txt           # CMake 构建配置
├── CMakePresets.json        # CMake 预设配置
├── README.md                # 本文件
├── config.yaml              # 应用配置文件
├── .gitignore
├── src/                     # 源代码
│   ├── main.cpp             # 主入口
│   ├── receiver/            # 组播接收
│   ├── publisher/           # 组播发送
│   ├── parser/              # 报文解析
│   ├── predictor/           # ONNX 推理
│   ├── buffer/              # 轨迹缓冲
│   ├── normalizer/          # 特征归一化
│   ├── yaml_config_loader.cpp # YAML 配置加载
│   └── utils/               # 工具类
├── include/                 # 对外头文件
│   ├── yaml_config_loader.h # YAML 配置加载器
│   └── ...
├── tests/                   # 单元测试 (GoogleTest)
│   ├── test_parser.cpp      # 报文解析测试
│   ├── test_buffer.cpp      # 轨迹缓冲测试
│   ├── test_normalizer.cpp  # 归一化测试
│   ├── test_logger.cpp      # 日志测试
│   ├── test_yaml_config.cpp # 配置加载测试
│   └── ...
├── scripts/                 # Python 脚本
│   ├── export_onnx_accurate.py  # PyTorch -> ONNX 转换
│   ├── compare_accuracy.py      # 精度对比
│   ├── run_export_and_test.sh   # 导出并测试
│   ├── run_folder_test.sh       # 文件夹测试
│   └── run_full_comparison.sh   # 完整对比
├── models/                  # ONNX 模型
├── data/                    # 测试数据
│   └── test_data/           # 测试样本
└── third_party/             # 第三方依赖
```

## 依赖项

| 依赖 | 版本 | 用途 |
|------|------|------|
| C++ | C++17 | 现代语言特性 |
| CMake | 3.20+ | 构建系统 |
| ONNX Runtime | 1.22.0+ | 模型推理 |
| googletest | 1.14.0 | 单元测试 |
| yaml-cpp | 0.8.0 | YAML 配置解析 |
| nlohmann/json | 3.11.0 | JSON 解析 |

## 快速开始

### 1. 环境准备

```bash
# 激活 Conda 环境 (如使用)
conda activate timesnet

# 设置 ONNX Runtime 库路径
export ONNXRUNTIME_DIR=$(pwd)/third_party/onnxruntime-linux-x64-gpu-1.22.0
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

### 2. 构建项目

```bash
# 创建构建目录
cd apps/cplusplus
mkdir -p build-debug && cd build-debug

# 配置 (Debug)
cmake .. --preset=linux-debug

# 构建
cmake --build . -j$(nproc)

# 运行测试
ctest --output-on-failure
```

### 3. 配置说明

应用支持两种配置方式：**YAML 配置文件** 和 **命令行参数**。YAML 优先级更高。

#### config.yaml 示例

```yaml
receiver:
  group: "230.1.1.22"      # 输入组播地址
  port: 8002                # 输入端口
  timeout_s: 2.0            # 接收超时(秒)
  skip_checksum: false      # 跳过校验和

publisher:
  group: "230.1.1.24"      # 输出组播地址
  port: 8011                # 输出端口
  ttl: 1                    # 组播 TTL

predictor:
  model_path: "models/timesnet.onnx"  # ONNX 模型路径
  num_classes: 2             # 类别数 (鸟/无人机)
  seq_len: 20                # 序列长度
  num_features: 14           # 特征数

buffer:
  seq_len: 20                # 序列长度
  max_age_s: 10.0            # 轨迹最大保留时间(秒)

inference:
  min_seq_len: 20            # 最小序列长度
  ema_alpha: 0.4             # EMA 平滑因子
```

### 4. 转换模型

使用 `export_onnx_accurate.py` 将训练好的 TimesNet 分类模型转换为 ONNX 格式。

```bash
# 导出 ONNX 模型
python scripts/export_onnx_accurate.py \
    --checkpoint ../../checkpoints/your_checkpoint.pth \
    --output models/timesnet.onnx

# 验证 ONNX 与 PyTorch 一致性
python scripts/export_onnx_accurate.py \
    --checkpoint ../../checkpoints/your_checkpoint.pth \
    --output models/timesnet.onnx \
    --verify

# 使用真实报文数据计算 period_list
python scripts/export_onnx_accurate.py \
    --checkpoint ../../checkpoints/your_checkpoint.pth \
    --output models/timesnet.onnx \
    --sample_data mydataset/radar_augv3/uav/P7_Sn3884171_win0_20.xls
```

### 5. 精度对比

使用 `compare_accuracy.py` 对比 PyTorch 与 ONNX 模型。

```bash
# 基本对比
python scripts/compare_accuracy.py \
    --pytorch ../../checkpoints/your_checkpoint.pth \
    --onnx models/timesnet.onnx \
    --num_samples 100

# 使用文件夹中的测试数据
python scripts/compare_accuracy.py \
    --pytorch ../../checkpoints/your_checkpoint.pth \
    --onnx models/timesnet.onnx \
    --use_folder_test \
    --test_data_dir apps/cplusplus/data/test_data

# 运行完整对比报告
python scripts/run_full_comparison.sh
```

### 6. 运行预测器

```bash
# 使用默认 config.yaml
./build/TimesNetPredictor

# 使用自定义配置文件
./build/TimesNetPredictor --config /path/to/config.yaml

# 完全命令行配置
./build/TimesNetPredictor \
    --in_group 230.1.1.22 \
    --in_port 8002 \
    --out_group 230.1.1.24 \
    --out_port 8011 \
    --model_path models/timesnet.onnx \
    --num_classes 2 \
    --seq_len 20
```

## 命令行参数

### 配置文件
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | YAML 配置文件路径 | config.yaml |

### 输入组播
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--in_group` | 输入组播地址 | 必填 |
| `--in_port` | 输入端口 | 必填 |
| `--in_iface` | 输入网卡 IP | 0.0.0.0 |
| `--timeout` | 接收超时(秒) | 2.0 |
| `--skip_checksum` | 跳过校验和 | false |

### 输出组播
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--out_group` | 输出组播地址 | 必填 |
| `--out_port` | 输出端口 | 必填 |
| `--out_iface` | 输出网卡 IP | 0.0.0.0 |
| `--ttl` | 组播 TTL | 1 |

### 模型
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | ONNX 模型路径 | 必填 |
| `--num_classes` | 类别数 | 2 |
| `--seq_len` | 序列长度 | 20 |
| `--num_features` | 特征数 | 14 |
| `--stats_path` | 归一化统计文件 | 空 |

### 缓冲与推理
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max_age_s` | 轨迹最大保留时间(秒) | 10.0 |
| `--min_seq_len` | 最小序列长度 | 20 |
| `--ema_alpha` | EMA 平滑因子 | 0.4 |

## 特征定义

特征顺序与 `features/feature_cols.json` 一致，共 **14 个特征**：

| 序号 | 特征名 | 说明 |
|------|--------|------|
| 1 | 径向距离 | 目标到雷达的距离 (km) |
| 2 | 方位 | 方位角 (度) |
| 3 | 俯仰 | 俯仰角 (度) |
| 4 | 点迹距离 | 点迹径向距离 (km) |
| 5 | 点迹方位 | 点迹方位角 (度) |
| 6 | 点迹俯仰 | 点迹俯仰角 (度) |
| 7 | 全速度 | 目标速度大小 (km/s) |
| 8 | 径向速度 | 径向速度 (km/s) |
| 9 | 方位速度 | 方位角变化率 (度/s) |
| 10 | 俯仰速度 | 俯仰角变化率 (度/s) |
| 11 | 多普勒展宽 | 归一化 0-1 |
| 12 | JEM | 归一化 0-1 |
| 13 | RCS | 归一化 0-1 |
| 14 | 目标信噪比 | 归一化 0-1 |

## 报文格式

### 输入报文 (组播)
```
[Sync: 2 bytes]   = 0xAA 0x55
[Length: 2 bytes] = 报文长度 (big-endian)
[Sequence: 4 bytes] = 序列号
[Timestamp: 4 bytes] = 时间戳 (25us 单位)
[ItemCount: 2 bytes] = 目标数量
[Items: ...] = 目标数据 (每项 44 bytes)
[Checksum: 2 bytes] = 校验和
```

### 输出报文 (组播)
```
[ItemCount: 2 bytes] = 结果数量
[Items: 25 bytes each]
  [TrackID: 4 bytes]    = 轨迹 ID
  [Timestamp: 8 bytes]  = 时间戳
  [Pred: 1 byte]        = 预测类别 (0=鸟, 1=无人机)
  [Prob0: 4 bytes]      = 类别 0 概率
  [Prob1: 4 bytes]      = 类别 1 概率
```

## 测试

```bash
# 运行所有测试
cd build-debug && ctest --output-on-failure

# 运行单个测试
./tests/test_parser       # 报文解析
./tests/test_buffer       # 轨迹缓冲
./tests/test_normalizer   # 归一化
./tests/test_yaml_config  # 配置加载
```

## 常用脚本

| 脚本 | 用途 |
|------|------|
| `run_export_and_test.sh` | 导出 ONNX 并运行快速测试 |
| `run_folder_test.sh` | 使用测试文件夹进行批量测试 |
| `run_full_comparison.sh` | 完整精度对比 (PyTorch vs ONNX vs C++) |

## 许可证

MIT License
