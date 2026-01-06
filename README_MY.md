# 脚本说明

## scripts/augment_uav_sliding.py

### 作用

该脚本用于对轨迹数据进行数据增强。通过滑动窗口方式，将长轨迹切分为多个短样本，从而扩充数据量。

- 根据 `--input_dir` 下一级子文件夹的名称自动分类
- 对指定类别的轨迹数据，使用滑动窗口生成多个样本
- 对其他类别的数据，直接复制保留
- 输出目录结构扁平化：每个类别一个文件夹

### 目录结构要求

```
input_dir/
├── uav/           # 包含 uav, drone, 无人机 关键词 -> uav 类
│   └── *.xls
├── bird/          # 包含 bird, birds, 鸟 关键词 -> bird 类
│   └── *.xls
├── drones/        # -> uav 类
│   └── *.xls
├── birds/         # -> bird 类
│   └── *.xls
└── other/         # 其他 -> other 类，直接复制
    └── *.xls
```

### 分类规则

| 类别 | 识别的文件夹名称关键词 |
|------|----------------------|
| `uav` | uav, drone, 无人机 |
| `bird` | bird, birds, 鸟 |
| `other` | 其他 |

### 用法

```bash
python scripts/augment_uav_sliding.py \
  --input_dir /path/to/input \
  --output_dir /path/to/output \
  --window_size 20 \
  --stride 5 \
  --augment_class uav
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | 必填 | 原始数据集根目录（包含类别子文件夹） |
| `--output_dir` | 必填 | 增强后数据输出目录 |
| `--window_size` | 20 | 每个窗口的点数 |
| `--stride` | 5 | 滑动步长 |
| `--include_partial` | False | 末尾不足窗口大小时是否保留 |
| `--min_len` | window_size | 最小序列长度 |
| `--pattern` | `*.xls` | 文件匹配模式 |
| `--augment_class` | all | 需要增强的类别，支持逗号分隔（如 `uav,bird`）或使用 `all` 表示所有类别 |
| `--feature_cols` | None | 逗号分隔的特征列名列表，处理后只保留这些列 |

### 输出结构

```
output_dir/
├── uav/           # 增强后的 uav 样本
│   ├── file1_win0_20.xls
│   ├── file1_win5_25.xls
│   └── ...
├── bird/          # 增强后的 bird 样本
│   ├── file2_win0_20.xls
│   └── ...
└── other/         # 直接复制的 other 样本
    └── file3.xls
```

### 示例

```bash
# 对无人机轨迹进行滑动窗口增强，窗口大小20，步长5
python scripts/augment_uav_sliding.py \
  --input_dir ./mydataset/radar \
  --output_dir ./mydataset/radar_augmented \
  --window_size 20 \
  --stride 5 \
  --augment_class uav

# 同时保留尾部不完整窗口
python scripts/augment_uav_sliding.py \
  --input_dir ./mydataset/radar \
  --output_dir ./mydataset/radar_augmented \
  --window_size 20 \
  --stride 5 \
  --include_partial \
  --augment_class uav

# 同时增强两个类别（uav 和 bird）
python scripts/augment_uav_sliding.py \
  --input_dir ./mydataset/radar \
  --output_dir ./mydataset/radar_augmented \
  --window_size 20 \
  --stride 5 \
  --augment_class uav,bird

# 使用 all 关键字增强所有类别
python scripts/augment_uav_sliding.py \
  --input_dir ./mydataset/radar \
  --output_dir ./mydataset/radar_augmented \
  --window_size 20 \
  --stride 5 \
  --augment_class all

# 只保留指定特征列进行增强（雷达航迹14维特征）
python scripts/augment_uav_sliding.py \
  --input_dir ./mydataset/radar \
  --output_dir ./mydataset/radar_augmented \
  --window_size 20 \
  --stride 5 \
  --augment_class uav,bird \
  --feature_cols "高（目标-滤波后）,径向距离,方位,俯仰,点迹距离,点迹方位,点迹俯仰,全速度,径向速度,方位速度,俯仰速度,多普勒展宽,JEM,RCS"
```

---

## apps/udp_timesnet_predict.py

### 作用

接收航迹组播报文，解析后按 `track_id` 维护滑窗序列，使用 TimesNet 进行分类预测，并将结果打包为新的 UDP 组播报文发布。

### 风险提示与回滚方案

- 风险：组播地址/端口配置不当可能引发端口冲突或网络策略拦截，影响其他业务组播流量。
- 回滚：停止该脚本，恢复原有端口/组播地址配置；若修改了防火墙/路由规则，按变更记录撤销即可。

### 用法

```bash
python apps/udp_timesnet_predict.py \
  --in_group 230.1.1.22 --in_port 8002 --in_iface 0.0.0.0 \
  --out_group 239.0.0.1 --out_port 9000 --out_iface 0.0.0.0 \
  --model_path /path/to/checkpoint.pth \
  --seq_len 20 --min_seq_len 20 \
  --stats_path /path/to/stats.json \
  --device auto
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--in_group` | 必填 | 输入组播地址 |
| `--in_port` | 必填 | 输入端口 |
| `--in_iface` | `0.0.0.0` | 输入网卡IP |
| `--bind_ip` | 空 | 绑定IP，默认 `0.0.0.0` |
| `--out_group` | 必填 | 输出组播地址 |
| `--out_port` | 必填 | 输出端口 |
| `--out_iface` | `0.0.0.0` | 输出网卡IP |
| `--ttl` | 1 | 组播 TTL |
| `--model_path` | 必填 | 模型 checkpoint 路径 |
| `--model` | `TimesNet` | 模型类型 |
| `--seq_len` | 20 | 序列长度 |
| `--min_seq_len` | 20 | 最小序列长度（不足时不推理） |
| `--stats_path` | 空 | 归一化统计文件 |
| `--device` | auto | `auto|cpu|cuda` |
| `--publish_interval_ms` | 0 | 发布节流毫秒（0=不节流） |
| `--use_batch_ema` | false | 同批号目标使用EMA平滑概率并统一判别 |
| `--ema_alpha` | 0.6 | EMA平滑系数（0-1，越大越偏向最新） |

### 归一化统计文件（stats.json）

支持两种结构之一：

1) 使用 list，与特征维度严格对齐
```json
{"mean":[...],"std":[...]}
```

2) 使用 dict，按特征列名匹配
```json
{"mean":{"高（目标-滤波后）":0.0},"std":{"高（目标-滤波后）":1.0}}
```

### 特征映射

模型输入特征（14维）与报文字段对应关系如下（内部已实现）：
- 高（目标-滤波后） -> height_m
- 径向距离 -> r_m
- 方位 -> a_deg
- 俯仰 -> e_deg
- 点迹距离 -> pr_m
- 点迹方位 -> pa_deg
- 点迹俯仰 -> pe_deg
- 全速度 -> vel_m_s
- 径向速度 -> radial_vel_m_s
- 方位速度 -> az_vel_deg_s
- 俯仰速度 -> el_vel_deg_s
- 多普勒展宽 -> doppler
- JEM -> jem
- RCS -> rcs_db

### 输出组播报文格式

按 `报文.md` 中“二、目标识别结果（识别->数据处理）”输出二进制报文（见 `udp/publisher.py`）：
- Header（12 字）：`frame_header=0xA999`、`msg_type`、`frame_length`、`frame_seq`、`system_id`、`radar_id+reserve`、`year/month/day/hour/minute/second(BCD)`、`sub_second(25us)`、`reserve`
- Body（20 字）：`target_count`、`batch_id`、`base_day(2 字)`、`time(2 字, 25us)`、`class_major`、`confidence(0.001)`、`reserve(12 字)`
- Tail（2 字）：`checksum`、`frame_tail=0x55AA`

---

## scripts/build_norm_stats.py

### 作用

从轨迹数据集生成归一化统计文件（mean/std），预测时可直接复用，避免重复扫描训练集。

### 风险提示与回滚方案

- 风险：统计文件与模型训练使用的特征/数据分布不一致会导致推理偏差。
- 回滚：重新使用训练集目录生成统计文件，或改回 `predict_uav.py` 的自动统计模式。

### 用法

```bash
python scripts/build_norm_stats.py \
  --data_dir /path/to/train_dataset \
  --output /path/to/stats.json
```

生成后的 `stats.json` 可在预测时通过 `scripts/predict_uav.py --stats_path` 或 `apps/udp_timesnet_predict.py --stats_path` 直接使用。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | 必填 | 数据目录 |
| `--output` | 必填 | 输出 stats.json 路径 |
| `--pattern` | `*.xls` | 文件匹配模式 |
| `--feature_cols` | 空 | 自定义特征列（逗号分隔） |

### 输出格式

```json
{
  "mean": {"高（目标-滤波后）": 0.0},
  "std": {"高（目标-滤波后）": 1.0}
}
```

## run.py

### 作用

TSLib 的统一入口文件，通过命令行参数控制执行各类时序任务。

主要功能：
1. 根据 `--task_name` 选择任务类型（长期预测、短期预测、插值、异常检测、分类、零样本预测）
2. 根据 `--is_training` 决定训练或评估模式
3. 自动管理 GPU 设备和随机种子

### 用法

```bash
python run.py --task_name <任务类型> --is_training <1或0> --model <模型> --data <数据集> [其他参数]
```

### 核心参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `--task_name` | 是 | 任务类型 |
| `--is_training` | 是 | 1=训练模式，0=评估模式 |
| `--model_id` | 是 | 实验标识符 |
| `--model` | 是 | 模型名称 |
| `--data` | 是 | 数据集类型 |

### 任务类型 (`--task_name`)

- `long_term_forecast`：长期预测
- `short_term_forecast`：短期预测
- `imputation`：插值
- `anomaly_detection`：异常检测
- `classification`：分类
- `zero_shot_forecast`：零样本预测

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--root_path` | `./data/ETT/` | 数据根目录 |
| `--data_path` | `ETTh1.csv` | 数据文件名 |
| `--features` | M | 预测模式：M(多变量预测多变量)、S(单变量预测单变量)、MS(多变量预测单变量) |
| `--seq_len` | 96 | 输入序列长度 |
| `--label_len` | 48 | 解码器起始token长度 |
| `--pred_len` | 96 | 预测序列长度 |

### trajxls 数据集（雷达航迹分类）

`--data trajxls` 是 TSLib 中自定义的轨迹数据集类型，用于**雷达航迹分类**任务。

#### 分类目标
- **鸟 (bird)** vs **无人机 (uav)**

#### 数据格式
- GBK 编码的 tab 分隔 XLS 文件

#### 目录结构要求

```
<root_path>/
├── 鸟/              # 鸟类轨迹样本
│   └── *.xls
├── 无人机/          # 无人机轨迹样本
│   └── **/*.xls
└── 无人机单独航迹/   # 另一批无人机轨迹
    └── **/*.xls
```

#### 特征列（共14维）
高（目标-滤波后）、径向距离、方位、俯仰、点迹距离、点迹方位、点迹俯仰、全速度、径向速度、方位速度、俯仰速度、多普勒展宽、JEM、RCS

#### 相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--track_points` | 20 | 每个样本的点数（截取轨迹尾部） |

#### 数据处理流程

`TrajectoryXLSLoader` 的完整数据处理流程：

```
1. 文件扫描 (_gather_samples)
   ├── <root>/鸟/*.xls              → label=0 (bird)
   ├── <root>/无人机/**/*.xls       → label=1 (uav)
   └── <root>/无人机单独航迹/**/*.xls → label=1 (uav)

2. 数据读取 (_read_single)
   ├── 读取 GBK 编码的 tab 分隔文件
   ├── 提取14个特征列
   └── 插值填充缺失值

3. 训练/测试划分
   ├── 80% 训练集，20% 测试集
   └── 随机打乱后按索引划分

4. 标准化 (_load_and_normalize)
   ├── 仅使用训练集数据拟合标准化器
   └── 对所有数据应用标准化

5. 截断处理
   └── 每个轨迹保留尾部 track_points 个点（默认20）

6. 数据返回 __getitem__
   └── 返回 (seq, label): seq 形状为 (track_points, 14)
```

#### 数据处理详细说明

**文件扫描：**
```python
# _gather_samples() 方法
patterns = [
    os.path.join(root_path, '鸟', '*.xls'),
    os.path.join(root_path, '无人机', '**', '*.xls'),
    os.path.join(root_path, '无人机单独航迹', '**', '*.xls'),
]
```

**特征列：**
| 索引 | 特征名 | 说明 |
|------|--------|------|
| 0 | 高（目标-滤波后） | 目标高度（滤波后） |
| 1 | 径向距离 | 径向距离 |
| 2 | 方位 | 方位角 |
| 3 | 俯仰 | 俯仰角 |
| 4 | 点迹距离 | 点迹距离 |
| 5 | 点迹方位 | 点迹方位角 |
| 6 | 点迹俯仰 | 点迹俯仰角 |
| 7 | 全速度 | 速度 |
| 8 | 径向速度 | 径向速度 |
| 9 | 方位速度 | 方位速度 |
| 10 | 俯仰速度 | 俯仰速度 |
| 11 | 多普勒展宽 | 多普勒展宽 |
| 12 | JEM | JEM特征 |
| 13 | RCS | RCS值 |

#### 示例

```bash
# 轨迹分类任务
python run.py \
  --task_name classification \
  --is_training 1 \
  --model_id trajxls \
  --model TimesNet \
  --data trajxls \
  --root_path ./mydataset/ \
  --features M \
  --seq_len 20 \
  --label_len 10 \
  --pred_len 10 \
  --itr 1
```

### GPU 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_gpu` | True | 是否使用GPU |
| `--gpu` | 0 | GPU编号 |
| `--gpu_type` | cuda | GPU类型：cuda 或 mps |
| `--use_multi_gpu` | False | 是否使用多GPU |
| `--devices` | 0,1,2,3 | 多GPU设备号 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--itr` | 1 | 实验重复次数 |
| `--train_epochs` | 10 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--learning_rate` | 0.0001 | 学习率 |
| `--patience` | 3 | 早停耐心值 |

### 示例

```bash
# 长期预测训练
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id TimesNet_ETTh1 \
  --model TimesNet \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --itr 1

# 评估模式（不训练）
python run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model_id TimesNet_ETTh1 \
  --model TimesNet \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv

# 使用GPU
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model TimesNet \
  --data ETTh1 \
  --use_gpu --gpu 0

# 多GPU训练
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model TimesNet \
  --data ETTh1 \
  --use_multi_gpu --devices 0,1,2,3
```

---

## scripts/predict_uav.py

### 作用

使用训练好的模型对新采集的轨迹数据进行分类预测。支持批量预测，输出每个样本的分类结果和置信度。

### 查找模型

```bash
# 查找所有训练好的模型
find ./checkpoints -name "checkpoint.pth"
```

### 用法

```bash
python scripts/predict_uav.py \
  --model_path ./checkpoints/xxx/checkpoint.pth \
  --data_dir ./mydataset/无人机航迹_aug/无人机 \
  --model TimesNet \
  --seq_len 20 \
  --output predictions.csv
```

### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--model_path` | 是 | 模型 checkpoint 路径 |
| `--data_dir` | 是 | 新数据目录（包含 xls 文件） |
| `--model` | 否 | 模型类型，默认 TimesNet |
| `--seq_len` | 否 | 序列长度，默认 20 |
| `--points_per_sample` | 否 | 每个样本点数，默认 20 |
| `--output` | 否 | 输出结果文件，默认 predictions.csv |
| `--device` | 否 | 设备，默认 cuda |

### 输出格式

预测结果保存为 CSV 文件，包含以下列：

| 列名 | 说明 |
|------|------|
| `file` | 文件名 |
| `prediction` | 预测类别：uav 或 bird |
| `confidence` | 置信度 |
| `prob_uav` | 无人机概率 |
| `prob_bird` | 鸟类概率 |

### 示例

```bash
# 查找模型
find ./checkpoints -name "checkpoint.pth"
# 输出示例: ./checkpoints/classification_TrajGBK_TimesNet_xxx/checkpoint.pth

# 运行预测
python scripts/predict_uav.py \
  --model_path ./checkpoints/classification_TrajGBK_TimesNet_xxx/checkpoint.pth \
  --data_dir ./mydataset/无人机航迹_aug/无人机 \
  --output predictions.csv
```

### 预测统计示例

```
预测统计:
  无人机 (uav): 150
  鸟 (bird): 3
```

---

## scripts/split_trajectory_dataset.py

### 作用

将航迹数据集按指定数量或比例拆分成两份（train/test 或 split1/split2）。

- 支持按比例拆分（如 0.2 = 20%）
- 支持按数量拆分（如 100 = 100条）
- 支持移动或复制两种输出模式

### 目录结构要求

```
root_dir/
├── uav/
│   └── *.xls
└── bird/
    └── *.xls
```

### 用法

```bash
python scripts/split_trajectory_dataset.py \
  --root_dir /path/to/dataset \
  --output_dir /path/to/output \
  --classes uav,bird \
  --split_value 0.2 \
  --mode copy
```

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--root_dir` | 是 | - | 数据集根目录 |
| `--output_dir` | 是 | - | 输出目录 |
| `--classes` | 是 | - | 类别，`all` 或逗号分隔（如 `uav,bird`） |
| `--split_value` | 是 | - | 分割值，`<1` 为比例，`>=1` 为数量 |
| `--mode` | 否 | move | `move`=移动文件，`copy`=复制文件 |
| `--file_pattern` | 否 | `*.xls` | 文件匹配模式 |
| `--recursive` | 否 | False | 递归扫描子目录 |
| `--seed` | 否 | 2021 | 随机种子 |

### 输出结构（mode=copy）

```
output_dir/
├── uav/
│   ├── split1/    # 20% 数据
│   └── split2/    # 80% 数据
└── bird/
    ├── split1/
    └── split2/
```

### 输出结构（mode=move）

```
root_dir/
├── uav/           # split1（指定数量/比例的文件保留在原目录）
└── bird/

output_dir/        # split2（剩余文件直接移动到类别目录）
├── uav/
└── bird/
```

### 示例

```bash
# 按比例拆分 (20% -> split1, 80% -> split2)
python scripts/split_trajectory_dataset.py \
  --root_dir ./mydataset/radar_augv2 \
  --output_dir ./mydataset/radar_split \
  --classes uav,bird \
  --split_value 0.2 \
  --mode copy

# 按数量拆分 (100条 -> split1)
python scripts/split_trajectory_dataset.py \
  --root_dir ./mydataset/radar_augv2 \
  --output_dir ./mydataset/radar_split \
  --classes all \
  --split_value 100 \
  --mode move

# 递归扫描子目录
python scripts/split_trajectory_dataset.py \
  --root_dir ./mydataset/radar_augv2 \
  --output_dir ./mydataset/radar_split \
  --classes uav \
  --split_value 0.3 \
  --recursive
```
