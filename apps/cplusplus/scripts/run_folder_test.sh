#!/bin/bash
# 航迹分类模型精度对比脚本（使用真实报文文件夹）
# 用法: ./run_folder_test.sh [checkpoint_path] [onnx_path] [test_data_dir]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}航迹分类模型精度对比（真实报文文件夹）${NC}"
echo -e "${GREEN}========================================${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# 项目根目录
PROJECT_ROOT="/home/tl/work/T/Time-Series-Library"

# 默认路径
CHECKPOINT_PATH="${1:-${PROJECT_ROOT}/checkpoints/classification_TrajGBK_TimesNet_trajxls_ftM_sl20_ll48_pl0_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/checkpoint.pth}"
ONNX_PATH="${2:-${SCRIPT_DIR}/../models/timesnet.onnx}"
TEST_DATA_DIR="${3:-${SCRIPT_DIR}/../data/test_data}"
SAMPLE_DATA="${PROJECT_ROOT}/mydataset/radar_augv3/uav/P7_Sn3884171_win0_20.xls"

echo ""
echo -e "${YELLOW}配置:${NC}"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  ONNX 模型: $ONNX_PATH"
echo "  测试数据: $TEST_DATA_DIR"
echo ""

# 检查 checkpoint 和 ONNX 模型
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}错误: checkpoint 文件不存在: $CHECKPOINT_PATH${NC}"
    exit 1
fi

if [ ! -f "$ONNX_PATH" ]; then
    echo -e "${RED}错误: ONNX 模型不存在: $ONNX_PATH${NC}"
    echo -e "${YELLOW}请先运行 run_export_and_test.sh 导出模型${NC}"
    exit 1
fi

# 检查测试数据目录
if [ ! -d "$TEST_DATA_DIR" ]; then
    echo -e "${RED}错误: 测试数据目录不存在: $TEST_DATA_DIR${NC}"
    exit 1
fi

NUM_TEST_FILES=$(find "$TEST_DATA_DIR" -name "*.xls" -o -name "*.csv" 2>/dev/null | wc -l)
if [ "$NUM_TEST_FILES" -eq 0 ]; then
    echo -e "${RED}错误: 测试目录中没有 .xls 或 .csv 文件${NC}"
    exit 1
fi

echo -e "找到 ${YELLOW}$NUM_TEST_FILES${NC} 个测试文件"
echo ""

# 精度对比（使用真实报文文件夹）
echo -e "${GREEN}[1/1] 精度对比...${NC}"
cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/compare_accuracy.py" \
    --pytorch "$CHECKPOINT_PATH" \
    --onnx "$ONNX_PATH" \
    --sample_data "$SAMPLE_DATA" \
    --use_folder_test \
    --test_data_dir "$TEST_DATA_DIR" \
    --accuracy_threshold 0.99 \
    --mae_threshold 0.01 \
    --rmse_threshold 0.02 \
    --output "$(dirname "$ONNX_PATH")/folder_test_report.json"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}测试完成!${NC}"
    echo -e "${GREEN}报告保存至: $(dirname "$ONNX_PATH")/folder_test_report.json${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}错误: 精度对比失败${NC}"
    exit 1
fi
