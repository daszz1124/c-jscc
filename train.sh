#!/bin/bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建带时间戳的日志目录
mkdir -p "training_logs/"

for C in 32 64 96 128 192; do
    python main.py \
        --training \
        --trainset DIV2K \
        --testset kodak \
        --distortion-metric MSE \
        --model SwinJSCC_w/_SA \
        --channel-type awgn \
        --C ${C} \
        --multiple-snr "1,4,7,10,13" \
        --model_size base \
        --workdir "training_workdir/C${C}" 2>&1 | \
        tee -a "training_logs/C${C}_awgn_snr10_${TIMESTAMP}.log"
done