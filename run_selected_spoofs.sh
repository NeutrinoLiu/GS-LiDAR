#!/bin/bash

# ====== 可配置项 ======
GPU_ID=$1                          # 第一个参数是 GPU ID
ATTACK_IDS=$2                      # 第二个参数是逗号分隔的 attack id 列表
CONFIG_DIR="all_configs"           # 配置文件所在目录
OUTPUT_BASE="eval_output/opv2v_reconstruction"  # 输出路径前缀
PYTHON_SCRIPT="train.py"          # 执行的 Python 文件
# =======================

# 检查输入参数
if [ -z "$GPU_ID" ] || [ -z "$ATTACK_IDS" ]; then
    echo "用法: ./run_selected_attacks.sh <GPU_ID> <attack_ids (逗号分隔)>"
    echo "示例: ./run_selected_attacks.sh 0 23,56,78,99"
    exit 1
fi

# 分割 attack id 列表
IFS=',' read -ra IDS <<< "$ATTACK_IDS"

# 遍历并运行
for ATTACK_ID in "${IDS[@]}"; do
    CONFIG_FILE="${CONFIG_DIR}/opv2v_spoof_attack_${ATTACK_ID}.yaml"
    OUTPUT_PATH="${OUTPUT_BASE}/spoof_${ATTACK_ID}"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "⚠️  配置文件不存在: $CONFIG_FILE，跳过该任务。"
        continue
    fi

    echo "🚀  正在运行 attack_id=${ATTACK_ID} 使用 GPU ${GPU_ID}..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT --config "$CONFIG_FILE" model_path="$OUTPUT_PATH"
done

echo "✅ 所有任务完成！"
