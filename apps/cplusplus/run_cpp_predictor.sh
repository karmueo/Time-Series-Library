#!/bin/bash
# TimesNet C++ 航迹预测执行脚本
# 用法: ./run_cpp_predictor.sh [OPTIONS]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}TimesNet C++ 航迹预测${NC}"
echo -e "${GREEN}========================================${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${SCRIPT_DIR}"

# 默认参数 (与 Python 版本一致)
IN_GROUP="${1:-230.1.1.22}"
IN_PORT="${2:-8002}"
IN_IFACE="0.0.0.0"
OUT_GROUP="${3:-230.1.1.24}"
OUT_PORT="${4:-8011}"
OUT_IFACE="0.0.0.0"
MODEL_PATH="${5:-${PROJECT_ROOT}/checkpoints/classification_TrajGBK_TimesNet_trajxls_ftM_sl20_ll48_pl0_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/checkpoint.pth}"
SEQ_LEN=20
MIN_SEQ_LEN=20
NUM_FEATURES=14
STATS_PATH="${6:-${PROJECT_ROOT}/mydataset/radar_augv3_stats.json}"
EMA_ALPHA=0.4
MAX_AGE_S=10.0
USE_GPU=false

# 检查必要文件
if [ ! -f "${MODEL_PATH}" ]; then
    echo -e "${RED}错误: 模型文件不存在: ${MODEL_PATH}${NC}"
    exit 1
fi

if [ ! -f "${STATS_PATH}" ]; then
    echo -e "${YELLOW}警告: 统计文件不存在: ${STATS_PATH}${NC}"
    STATS_PATH=""
fi

# 检查 ONNX 模型 (从 checkpoint 自动导出或使用现有文件)
ONNX_PATH="${SCRIPT_DIR}/models/timesnet.onnx"
if [ ! -f "${ONNX_PATH}" ]; then
    echo -e "${YELLOW}ONNX 模型不存在，正在从 checkpoint 导出...${NC}"
    python3 "${SCRIPT_DIR}/scripts/convert_to_onnx.py" \
        --checkpoint_path "${MODEL_PATH}" \
        --output_path "${ONNX_PATH}" \
        --seq_len ${SEQ_LEN} \
        --d_model 128 \
        --n_heads 4 \
        --d_ff 256 \
        --top_k 3 \
        --num_features ${NUM_FEATURES}
fi

echo ""
echo -e "${YELLOW}配置:${NC}"
echo "  输入组播: ${IN_GROUP}:${IN_PORT}"
echo "  输出组播: ${OUT_GROUP}:${OUT_PORT}"
echo "  模型: ${MODEL_PATH}"
echo "  ONNX: ${ONNX_PATH}"
echo "  序列长度: ${SEQ_LEN}"
echo "  特征数: ${NUM_FEATURES}"
echo "  EMA alpha: ${EMA_ALPHA}"
if [ -n "${STATS_PATH}" ]; then
    echo "  统计文件: ${STATS_PATH}"
fi
echo ""

# 编译 (如果需要)
BUILD_DIR="${SCRIPT_DIR}/build"
EXECUTABLE="${BUILD_DIR}/TimesNetPredictor"
if [ ! -f "${EXECUTABLE}" ]; then
    echo -e "${GREEN}[1/2] 编译项目...${NC}"

    # 下载 ONNX Runtime C++ SDK (如果需要)
    ORT_DIR="${SCRIPT_DIR}/third_party/onnxruntime-linux-x64-gpu"
    if [ ! -d "${ORT_DIR}" ]; then
        echo "下载 ONNX Runtime C++ SDK..."
        mkdir -p "${SCRIPT_DIR}/third_party"
        cd "${SCRIPT_DIR}/third_party"
        wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz
        tar -xzf onnxruntime-linux-x64-gpu-1.22.0.tgz
        rm onnxruntime-linux-x64-gpu-1.22.0.tgz
        cd "${SCRIPT_DIR}"
    fi

    # 创建 build 目录并编译
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_GPU=OFF \
        -DBUILD_TESTS=OFF \
        -DONNXRUNTIME_DIR="${ORT_DIR}"

    cmake --build . --config Release -j4

    echo -e "${GREEN}编译完成!${NC}"
else
    echo -e "${GREEN}[1/2] 使用现有编译结果${NC}"
fi

# 设置环境变量
export LD_LIBRARY_PATH="${SCRIPT_DIR}/third_party/onnxruntime-linux-x64-gpu/lib:${LD_LIBRARY_PATH}"

# 运行
echo ""
echo -e "${GREEN}[2/2] 启动预测服务...${NC}"
echo "----------------------------------------"

"${EXECUTABLE}" \
    --in_group "${IN_GROUP}" \
    --in_port "${IN_PORT}" \
    --in_iface "${IN_IFACE}" \
    --out_group "${OUT_GROUP}" \
    --out_port "${OUT_PORT}" \
    --out_iface "${OUT_IFACE}" \
    --model_path "${ONNX_PATH}" \
    --seq_len "${SEQ_LEN}" \
    --min_seq_len "${MIN_SEQ_LEN}" \
    --num_features "${NUM_FEATURES}" \
    --ema_alpha "${EMA_ALPHA}" \
    --max_age_s "${MAX_AGE_S}" \
    ${STATS_PATH:+--stats_path "${STATS_PATH}"} \
    ${USE_GPU:+--use_gpu}

echo "----------------------------------------"
echo -e "${GREEN}预测服务已停止${NC}"
