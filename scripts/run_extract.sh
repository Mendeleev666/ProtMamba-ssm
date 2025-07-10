#!/usr/bin/env bash
# 用于调用 extract_features.py 脚本的示例 Bash 文件
# 在下方变量中填入实际路径即可

CSV_PATH="path/to/sequences.csv"     # CSV 输入文件
MODEL_DIR="data/example_pretrained_model"  # 预训练模型目录
OUTPUT_FILE="embeddings.npz"               # 输出文件名
DEVICE="cuda"                              # 使用的设备，CPU 可改为 cpu

python3 scripts/extract_features.py "$CSV_PATH" \
    --model "$MODEL_DIR" \
    --output "$OUTPUT_FILE" \
    --device "$DEVICE"
