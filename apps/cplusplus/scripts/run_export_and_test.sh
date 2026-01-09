#!/bin/bash
# 航迹分类模型导出与精度对比脚本
# 用法: ./run_export_and_test.sh [checkpoint_path] [test_data_dir]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}航迹分类模型导出与精度对比${NC}"
echo -e "${GREEN}========================================${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# 项目根目录
PROJECT_ROOT="/home/tl/work/T/Time-Series-Library"

# 默认路径
CHECKPOINT_PATH="${1:-${PROJECT_ROOT}/checkpoints/classification_TrajGBK_TimesNet_trajxls_ftM_sl20_ll48_pl0_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/checkpoint.pth}"
MODEL_DIR="${SCRIPT_DIR}/../models"
TEST_DATA_DIR="${2:-${SCRIPT_DIR}/../data/test_data}"
SAMPLE_DATA="${PROJECT_ROOT}/mydataset/radar_augv3/uav/P7_Sn3884171_win0_20.xls"

echo ""
echo -e "${YELLOW}配置:${NC}"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  输出模型: $MODEL_DIR/timesnet.onnx"
echo "  测试数据: $TEST_DATA_DIR"
echo ""

# 检查 checkpoint 是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}错误: checkpoint 文件不存在: $CHECKPOINT_PATH${NC}"
    exit 1
fi

# 创建模型目录
mkdir -p "$MODEL_DIR"

# 1. 导出 ONNX 模型
echo -e "${GREEN}[1/3] 导出 ONNX 模型...${NC}"
cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/export_onnx_accurate.py" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output "$MODEL_DIR/timesnet.onnx" \
    --verify \
    --sample_data "$SAMPLE_DATA"

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: ONNX 导出失败${NC}"
    exit 1
fi

# 检查 ONNX 模型是否存在
if [ ! -f "$MODEL_DIR/timesnet.onnx" ]; then
    echo -e "${RED}错误: ONNX 模型未生成${NC}"
    exit 1
fi

echo -e "${GREEN}ONNX 模型已导出: $MODEL_DIR/timesnet.onnx${NC}"

# 2. 准备测试数据
echo ""
echo -e "${GREEN}[2/3] 准备测试数据...${NC}"
if [ -d "$TEST_DATA_DIR" ]; then
    NUM_TEST_FILES=$(find "$TEST_DATA_DIR" -name "*.xls" -o -name "*.csv" 2>/dev/null | wc -l)
    echo -e "找到 ${YELLOW}$NUM_TEST_FILES${NC} 个测试文件"
else
    echo -e "${YELLOW}警告: 测试目录不存在，使用随机数据${NC}"
fi

# 3. 精度对比
echo ""
echo -e "${GREEN}[3/3] 精度对比...${NC}"
cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/compare_accuracy.py" \
    --pytorch "$CHECKPOINT_PATH" \
    --onnx "$MODEL_DIR/timesnet.onnx" \
    --sample_data "$SAMPLE_DATA" \
    --accuracy_threshold 0.99 \
    --mae_threshold 0.01 \
    --rmse_threshold 0.02 \
    --output "$MODEL_DIR/accuracy_report.json"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}测试完成!${NC}"
    echo -e "${GREEN}报告保存至: $MODEL_DIR/accuracy_report.json${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}错误: 精度对比失败${NC}"
    exit 1
fi
