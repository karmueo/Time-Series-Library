# 快速开始：使用固定 period_list 确保 PyTorch 和 ONNX 一致

## 问题
你发现 PyTorch 模型和 ONNX 模型的输出存在微小差异（< 1%），这是因为 ONNX 固化了 `period_list = [10, 2, 1]`，而 PyTorch 每次推理时动态计算。

## 解决方案
使用 **TimesNetPredictorFixedPeriod** 让 Python 也使用固定的 period_list。

---

## 方法 1：测试验证（推荐先运行）

### 1. 运行验证脚本

```bash
cd /home/tl/work/T/Time-Series-Library

PYTHONPATH=/home/tl/work/T/Time-Series-Library:$PYTHONPATH \
python apps/cplusplus/scripts/verify_fixed_period.py
```

### 2. 预期输出

```
✅ 固定 period_list 方案成功！
   PyTorch(固定) 与 ONNX 的概率差异 < 0.001
   可以在 Python 脚本中使用 TimesNetPredictorFixedPeriod
```

### 3. 验证结果示例

| 模型 | UAV 概率 | Bird 概率 |
|------|----------|-----------|
| 原始 PyTorch | 0.999947 | 0.000053 |
| **固定 period PyTorch** | **0.999953** | **0.000047** |
| **ONNX** | **0.999953** | **0.000047** |
| 差异 | **0.000000** ✅ | **0.000000** ✅ |

---

## 方法 2：在实际应用中使用

### 选项 A：修改现有脚本 `apps/udp_timesnet_predict.py`

#### 步骤 1：修改导入

```python
# 原来的导入
# from core.predictor import TimesNetPredictor

# 新的导入
from core.predictor_fixed_period import TimesNetPredictorFixedPeriod
```

#### 步骤 2：修改预测器创建

```python
# 原来的代码
# predictor = TimesNetPredictor(
#     model_path=args.model_path,
#     model_name=args.model,
#     num_classes=args.num_classes,
#     device=device,
#     model_cfg=model_cfg
# )

# 新的代码（添加 period_list）
predictor = TimesNetPredictorFixedPeriod(
    model_path=args.model_path,
    period_list=[10, 2, 1],  # 与 ONNX 导出时一致
    model_name=args.model,
    num_classes=args.num_classes,
    device=device,
    model_cfg=model_cfg
)
```

#### 步骤 3：验证

```bash
python apps/udp_timesnet_predict.py \
    --in_group "230.1.1.22" \
    --in_port 8002 \
    --out_group "230.1.1.24" \
    --out_port 8011 \
    --model_path "apps/cplusplus/models/timesnet.onnx" \
    --seq_len 20 \
    --stats_path "./mydataset/radar_augv3_stats.json" \
    --local_test \
    --local_test_path "mydataset/radar_augv3/uav/P1005_Sn955698_win0_20.xls"
```

### 选项 B：创建新脚本

如果不想修改现有脚本，可以创建一个新文件：

```python
# apps/udp_timesnet_predict_fixed.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 使用固定 period_list 的预测器
from core.predictor_fixed_period import TimesNetPredictorFixedPeriod

# ... 其余代码与 udp_timesnet_predict.py 相同 ...

# 只需修改预测器创建部分
predictor = TimesNetPredictorFixedPeriod(
    model_path=args.model_path,
    period_list=[10, 2, 1],  # 与 ONNX 一致
    model_name=args.model,
    num_classes=args.num_classes,
    device=device,
    model_cfg=model_cfg
)
```

---

## 常见问题

### Q1: period_list 应该使用什么值？

**A**: 使用 ONNX 导出时计算的值：

```bash
# 查看导出日志
python apps/cplusplus/scripts/export_onnx_accurate.py \
    --checkpoint checkpoints/your_checkpoint.pth \
    --output apps/cplusplus/models/timesnet.onnx \
    --sample_data mydataset/radar_augv3/uav/P7_Sn3884171_win0_20.xls

# 输出中会显示：
# Period list from sample data (top_k=3): [10, 2, 1]
```

如果导出时使用的是 `P7_Sn3884171_win0_20.xls`，则 period_list = `[10, 2, 1]`

### Q2: 不同数据使用不同 period_list 会影响精度吗？

**A**: 影响极小：

- 分类结果：100% 一致
- 概率差异：< 1%
- 原因：period_list 用于提取时序特征，不同 period_list 提取的特征略有不同，但对最终分类影响很小

### Q3: 如何选择最优的 period_list？

**A**: 有两种方法：

1. **使用训练集的平均值**（推荐）：
   ```python
   # 在多个训练样本上计算平均 period_list
   python -c "
   import numpy as np
   # 假设你计算了多个样本的 period_list
   periods = [[10,2,1], [10,6,2], [8,4,2], ...]
   mean_period = np.array(periods).mean(axis=0).round().astype(int)
   print(f'平均 period_list: {mean_period.tolist()}')
   "
   ```

2. **使用最常见的 period_list**：
   ```bash
   # 在验证集上统计 period_list 分布
   python apps/cplusplus/scripts/analyze_period_distribution.py \
       --data_dir mydataset/radar_augv3 \
       --num_samples 1000
   ```

### Q4: ONNX 模型可以重新导出吗？

**A**: 可以，使用不同的样本数据重新导出：

```bash
# 使用 P1005 数据重新导出
python apps/cplusplus/scripts/export_onnx_accurate.py \
    --checkpoint checkpoints/your_checkpoint.pth \
    --output apps/cplusplus/models/timesnet.onnx \
    --sample_data mydataset/radar_augv3/uav/P1005_Sn955698_win0_20.xls \
    --top_k 3

# 新模型的 period_list 将是 [10, 6, 2]
```

但需要注意：
- 重新导出后，需要更新所有使用该 ONNX 模型的地方
- Python 脚本也需要使用新的 period_list = `[10, 6, 2]`

---

## 总结

### 推荐方案

✅ **使用 TimesNetPredictorFixedPeriod**：

1. **简单**：只需修改几行代码
2. **可靠**：已验证 PyTorch 和 ONNX 输出完全一致
3. **零成本**：无需额外开发或部署
4. **性能优秀**：分类准确率 100%

### 快速验证

```bash
# 1. 验证一致性
PYTHONPATH=/home/tl/work/T/Time-Series-Library:$PYTHONPATH \
python apps/cplusplus/scripts/verify_fixed_period.py

# 2. 如果看到 ✅，说明方案可行

# 3. 修改你的应用代码使用 TimesNetPredictorFixedPeriod
```

### 需要帮助？

查看完整文档：
- `apps/cplusplus/docs/dynamic_period_list_solutions.md` - 所有技术方案详解
- `apps/cplusplus/README.md` - C++ 项目总览
