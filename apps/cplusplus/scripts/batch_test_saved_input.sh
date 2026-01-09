#!/bin/bash
#
# 批量测试脚本：测试所有保存的输入向量
#
# 使用方法:
#   ./batch_test_saved_input.sh [input_dir] [model.onnx]
#
# 示例:
#   ./batch_test_saved_input.sh
#   ./batch_test_saved_input.sh apps/cplusplus/tmp apps/cplusplus/models/timesnet.onnx

set -e

# 默认参数
INPUT_DIR="${1:-apps/cplusplus/tmp}"
MODEL="${2:-apps/cplusplus/models/timesnet.onnx}"
USE_GPU="${3:-false}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "批量测试保存的输入向量"
echo "=================================================="
echo "输入目录: $INPUT_DIR"
echo "模型文件: $MODEL"
echo "使用 GPU: $USE_GPU"
echo "=================================================="
echo ""

# 检查目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}错误: 目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

# 检查模型是否存在
if [ ! -f "$MODEL" ]; then
    echo -e "${RED}错误: 模型文件不存在: $MODEL${NC}"
    exit 1
fi

# 检查 Python 脚本是否存在
PYTHON_SCRIPT="apps/cplusplus/scripts/test_saved_input.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误: Python 脚本不存在: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# 统计文件数量
FILE_COUNT=$(ls -1 ${INPUT_DIR}/input_*.json 2>/dev/null | wc -l)

if [ "$FILE_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}警告: 在 ${INPUT_DIR} 中没有找到 input_*.json 文件${NC}"
    exit 0
fi

echo -e "${GREEN}找到 ${FILE_COUNT} 个输入文件${NC}"
echo ""

# 创建结果目录
RESULTS_DIR="${INPUT_DIR}/results"
mkdir -p "$RESULTS_DIR"

# 汇总结果文件
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
echo "批量测试结果汇总" > "$SUMMARY_FILE"
echo "测试时间: $(date)" >> "$SUMMARY_FILE"
echo "模型: $MODEL" >> "$SUMMARY_FILE"
echo "文件数量: $FILE_COUNT" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "文件名, 轨迹ID, 预测, 无人机概率, 鸟类概率, 推理时间(ms)" >> "$SUMMARY_FILE"

# �计数器
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=0

# 遍历所有输入文件
for file in ${INPUT_DIR}/input_*.json; do
    if [ ! -f "$file" ]; then
        continue
    fi

    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    BASENAME=$(basename "$file" .json)

    echo -e "${YELLOW}[${TOTAL_COUNT}/${FILE_COUNT}] 测试: ${BASENAME}${NC}"

    # 运行测试并捕获输出
    GPU_FLAG=""
    if [ "$USE_GPU" = "true" ]; then
        GPU_FLAG="--use_gpu"
    fi

    if python "$PYTHON_SCRIPT" "$file" "$MODEL" $GPU_FLAG > "${RESULTS_DIR}/${BASENAME}_output.txt" 2>&1; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo -e "${GREEN}✓ 成功${NC}"

        # 提取关键信息并写入汇总文件
        if command -v jq &> /dev/null; then
            # 如果安装了 jq，使用它提取 JSON 数据
            TRACK_ID=$(jq -r '.results[0].track_id' "${RESULTS_DIR}/${BASENAME}_output.txt" 2>/dev/null || echo "N/A")
            PREDICTION=$(jq -r '.results[0].prediction' "${RESULTS_DIR}/${BASENAME}_output.txt" 2>/dev/null || echo "N/A")
            PROB_UAV=$(jq -r '.results[0].prob_uav' "${RESULTS_DIR}/${BASENAME}_output.txt" 2>/dev/null || echo "N/A")
            PROB_BIRD=$(jq -r '.results[0].prob_bird' "${RESULTS_DIR}/${BASENAME}_output.txt" 2>/dev/null || echo "N/A")
            INFERENCE_TIME=$(jq -r '.inference_time_ms' "${RESULTS_DIR}/${BASENAME}_output.txt" 2>/dev/null || echo "N/A")
        else
            # 否则使用 grep 提取
            TRACK_ID=$(grep -oP 'track_id": \K\d+' "${RESULTS_DIR}/${BASENAME}_output.txt" 2>/dev/null || echo "N/A")
            PREDICTION=$(grep -oP '"prediction": \K\d+' "${RESULTS_DIR}/${BASENAME}_output.txt" 2>/dev/null || echo "N/A")
            PROB_UAV=$(grep -oP '"prob_uav": \K[0-9.]+' "${RESULTS_DIR}/${BASENAME}_output.txt" 2>/dev/null || echo "N/A")
            PROB_BIRD=$(grep -oP '"prob_bird": \K[0-9.]+' "${RESULTS_DIR}/${BASENAME}_output.txt" 2>/dev/null || echo "N/A")
            INFERENCE_TIME=$(grep -oP '"inference_time_ms": \K[0-9.]+' "${RESULTS_DIR}/${BASENAME}_output.txt" 2>/dev/null || echo "N/A")
        fi

        echo "${BASENAME}, ${TRACK_ID}, ${PREDICTION}, ${PROB_UAV}, ${PROB_BIRD}, ${INFERENCE_TIME}" >> "$SUMMARY_FILE"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo -e "${RED}✗ 失败${NC}"
        echo "  错误日志: ${RESULTS_DIR}/${BASENAME}_output.txt"
        echo "${BASENAME}, ERROR, ERROR, ERROR, ERROR, ERROR" >> "$SUMMARY_FILE"
    fi

    echo ""
done

# 打印汇总
echo "=================================================="
echo "测试完成"
echo "=================================================="
echo -e "总计: ${TOTAL_COUNT} | ${GREEN}成功: ${SUCCESS_COUNT}${NC} | ${RED}失败: ${FAIL_COUNT}${NC}"
echo ""
echo "详细结果保存在: ${RESULTS_DIR}/"
echo "汇总报告: ${SUMMARY_FILE}"
echo ""

# 打印汇总表格
if [ -f "$SUMMARY_FILE" ]; then
    echo "=================================================="
    echo "结果汇总"
    echo "=================================================="
    column -t -s ',' "$SUMMARY_FILE"
fi

# 返回状态
if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
else
    exit 0
fi
