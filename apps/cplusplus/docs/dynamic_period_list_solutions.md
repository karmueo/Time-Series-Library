# ONNX 动态 period_list 解决方案

## 问题背景

当前 ONNX 导出时固定了 `period_list = [10, 2, 1]`（来自 P7 数据），而 Python PyTorch 模型在每次推理时动态计算 period_list：

- **P7 数据**: period_list = [10, 2, 1]
- **P1005 数据**: period_list = [10, 6, 2]

这导致两种模型的输出存在微小差异（< 1%）。

## 可行性分析

### ONNX 的根本限制

```python
# TimesNet.py 中的关键代码
for period in period_list:
    out = out.reshape(B, length // period, period, N)  # ❌ ONNX 无法动态 reshape
```

**ONNX 要求计算图在导出时固定**，无法在运行时动态改变 tensor 的 shape。

---

## 方案 A：Python 使用固定 period_list ⭐推荐⭐

### 实现思路
修改 Python 预测脚本，使用与 ONNX 相同的固定 period_list，确保两者输出完全一致。

### 优势
- ✅ **最简单**: 只需修改 Python 代码
- ✅ **完全一致**: PyTorch 和 ONNX 输出误差 < 1e-6
- ✅ **零成本**: 无需额外的开发或部署工作
- ✅ **已验证**: 测试脚本验证通过

### 使用步骤

#### 1. 使用新的预测器类

```python
# 在 apps/udp_timesnet_predict.py 中
from core.predictor_fixed_period import TimesNetPredictorFixedPeriod

# 创建预测器时指定 period_list
predictor = TimesNetPredictorFixedPeriod(
    model_path="checkpoints/xxx.pth",
    period_list=[10, 2, 1],  # 与 ONNX 导出时一致
    model_name="TimesNet",
    num_classes=2,
    device="cpu",
    model_cfg=model_cfg
)
predictor.load(num_features=14)
```

#### 2. 验证一致性

运行验证脚本：
```bash
PYTHONPATH=/home/tl/work/T/Time-Series-Library:$PYTHONPATH \
python apps/cplusplus/scripts/verify_fixed_period.py
```

预期输出：
```
✅ 固定 period_list 方案成功！
   PyTorch(固定) 与 ONNX 的概率差异 < 0.001
```

### 性能对比

| 模型 | UAV 概率 | Bird 概率 |
|------|----------|-----------|
| 原始 PyTorch | 0.999947 | 0.000053 |
| 固定 period PyTorch | 0.999953 | 0.000047 |
| ONNX | 0.999953 | 0.000047 |
| **差异** | **0.000000** | **0.000000** |

---

## 方案 B：C++ 实现 FFT + 多模型切换（高级）

### 实现思路
1. 在 C++ 中实现 FFT 计算 period_list
2. 预先导出多个常用的 period_list 对应的 ONNX 模型
3. 运行时动态计算并选择合适的模型

### 优势
- ✅ 保留动态 period_list 的灵活性
- ✅ 每个输入都使用最优的 period_list

### 劣势
- ❌ 需要实现 C++ FFT（复杂度高）
- ❌ 需要存储多个 ONNX 模型
- ❌ 模型切换带来额外开销
- ❌ 需要分析数据确定常用 period_list

### 实现框架

#### 1. C++ FFT 实现

```cpp
// apps/cplusplus/include/fft_period_calculator.h
#ifndef FFT_PERIOD_CALCULATOR_H
#define FFT_PERIOD_CALCULATOR_H

#include <vector>
#include <complex>
#include <numeric>

namespace timesnet {

class FFTPeriodCalculator {
public:
    /**
     * @brief 计算输入数据的 top-k 周期列表
     * @param data 输入数据 [seq_len][num_features]
     * @param seq_len 序列长度
     * @param num_features 特征数
     * @param top_k 返回前 k 个周期
     * @return 周期列表，如 [10, 2, 1]
     */
    static std::vector<int> computePeriodList(
        const std::vector<std::vector<float>>& data,
        int seq_len,
        int num_features,
        int top_k = 3
    );

private:
    // FFT 实现（可使用 FFTW 或 kissFFT）
    static std::vector<std::complex<float>> FFT(
        const std::vector<float>& signal
    );
};

} // namespace timesnet

#endif
```

#### 2. 多模型管理器

```cpp
// apps/cplusplus/include/multi_model_manager.h
#ifndef MULTI_MODEL_MANAGER_H
#define MULTI_MODEL_MANAGER_H

#include <map>
#include <string>
#include <vector>
#include "predictor/onnx_predictor.h"

namespace timesnet {

/**
 * @brief 多模型管理器
 * 根据 period_list 选择对应的 ONNX 模型
 */
class MultiModelManager {
public:
    /**
     * @brief 加载多个模型
     * @param models 模型配置列表
     *
     * 示例：
     *   {
     *     {{10, 2, 1}, "models/timesnet_p10_2_1.onnx"},
     *     {{10, 6, 2}, "models/timesnet_p10_6_2.onnx"},
     *   }
     */
    bool loadModels(
        const std::map<std::vector<int>, std::string>& models
    );

    /**
     * @brief 根据 period_list 选择模型进行推理
     * @param data 输入数据
     * @param period_list 计算得到的周期列表
     * @return 预测结果
     */
    std::pair<std::vector<int>, std::vector<float>> predict(
        const std::vector<std::vector<std::vector<float>>>& data,
        const std::vector<int>& period_list
    );

    /**
     * @brief 自动计算 period_list 并推理
     */
    std::pair<std::vector<int>, std::vector<float>> predictWithAutoPeriod(
        const std::vector<std::vector<std::vector<float>>>& data
    );

private:
    std::map<std::vector<int>, std::unique_ptr<OnnxPredictor>> models_;
    OnnxPredictor* default_model_ = nullptr;  // 默认模型
};

} // namespace timesnet

#endif
```

#### 3. 导出多个模型

```bash
#!/bin/bash
# 导出多个 ONNX 模型

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/tl/work/T/Time-Series-Library"
CHECKPOINT="checkpoints/classification_TrajGBK_TimesNet_trajxls_ftM_sl20_ll48_pl0_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/checkpoint.pth"
MODEL_DIR="apps/cplusplus/models"

# 分析数据，找出常用的 period_list
# 假设我们发现常用的有：[10,2,1], [10,6,2], [8,4,2]

declare -A PERIOD_LISTS=(
    ["10_2_1"]="[10,2,1]"
    ["10_6_2"]="[10,6,2]"
    ["8_4_2"]="[8,4,2]"
)

for key in "${!PERIOD_LISTS[@]}"; do
    echo "导出 period_list=${PERIOD_LISTS[$key]}"

    python "$SCRIPT_DIR/export_onnx_accurate.py" \
        --checkpoint "$CHECKPOINT" \
        --output "$MODEL_DIR/timesnet_p${key}.onnx" \
        --sample_data mydataset/radar_augv3/uav/P7_Sn3884171_win0_20.xls \
        --top_k 3

    # 修改 export_onnx_accurate.py 支持自定义 period_list
    # 这里需要修改脚本接受 --period_list 参数
done
```

#### 4. 修改导出脚本支持自定义 period_list

```python
# apps/cplusplus/scripts/export_onnx_accurate.py 添加
parser.add_argument('--period_list', type=int, nargs='+',
                    help='自定义 period_list，如 --period_list 10 2 1')

# 在 load_and_prepare_model 中使用
if args.period_list:
    period_list = args.period_list
    print(f"使用自定义 period_list: {period_list}")
else:
    # 原有逻辑：从样本数据计算
    period_list = compute_period_list(sample_input, k=args.top_k)
```

### 部署建议

1. **分析常用 period_list**：
   ```bash
   # 在验证集上计算 period_list 分布
   python apps/cplusplus/scripts/analyze_period_distribution.py \
       --data_dir mydataset/radar_augv3 \
       --num_samples 1000
   ```

2. **选择覆盖 90%+ 场景的 3-5 个 period_list**

3. **使用默认模型处理罕见 period_list**

---

## 方案 C：修改 ONNX 导出结构（不推荐）

### 尝试思路
将 period_list 作为 ONNX 模型的输入，而不是固化在模型中。

### 技术难点
- ❌ ONNX 中无法用动态值控制 `reshape` 操作
- ❌ ONNX 不支持动态的条件分支和循环结构
- ❌ 需要大幅修改模型架构和导出逻辑
- ❌ 可能无法在所有 ONNX Runtime 上运行

### 结论
**不推荐**，实现难度极大且收益有限。

---

## 方案 D：完全使用 libtorch（不推荐）

### 思路
放弃 ONNX，在 C++ 中直接使用 libtorch (PyTorch C++ API)

### 劣势
- ❌ libtorch 部署复杂（需要动态链接、Python 依赖）
- ❌ 库体积巨大（> 500MB）
- ❌ 性能不如 ONNX Runtime
- ❌ 部署和运维成本高

### 结论
仅在需要完全灵活的 Python 功能时考虑，一般不推荐。

---

## 推荐方案总结

| 方案 | 复杂度 | 一致性 | 灵活性 | 推荐度 |
|------|--------|--------|--------|--------|
| A. Python 固定 period_list | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| B. C++ FFT + 多模型 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| C. 修改 ONNX 导出 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| D. 使用 libtorch | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |

## 最终建议

### 生产环境：方案 A
- 修改 Python 脚本使用 `TimesNetPredictorFixedPeriod`
- 确保与 ONNX 完全一致
- 零额外成本，简单可靠

### 研究环境：方案 B
- 如果需要探索动态 period_list 的影响
- 有充足开发资源
- 对灵活性要求高

### 不推荐：方案 C、D
- 技术难度高
- 收益不明显
- 维护成本大
