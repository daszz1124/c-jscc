#!/bin/bash
# SwinJSCC 训练脚本 - 函数封装版

# 创建时间戳和日志目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "mmeb_training/"
mkdir -p "mmeb_training_logs/"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

run_training() {
    local c=$1
    local model=$2
    local channel_type=$3
    local snr_set=$4
    local metric=$5
    local model_size=$6
    local dataset_name=$7
    local test_snr=$8
    local test_c=$9

    local work_dir="mmeb_training/${TIMESTAMP}_C${c}_${channel_type}_snr${snr_set//,/_}_${model//\//_}_${metric}"
    local log_file="mmeb_training_logs/${TIMESTAMP}_C${c}_${channel_type}_snr${snr_set//,/_}_${model//\//_}_${metric}.log"

    mkdir -p "${work_dir}"

    local start_time=$(date +%s)
    python -W ignore::FutureWarning:timm.models.layers ddp_mmeb_datasets.py \
        --training \
        --testset MMEB \
        --dataset_name "${dataset_name}" \
        --distortion-metric "${metric}" \
        --model "${model}" \
        --channel-type "${channel_type}" \
        --C "${c}" \
        --multiple-snr "${snr_set}" \
        --model_size "${model_size}" \
        --test_snr "${test_snr}" \
        --test_C "${test_c}" \
        --workdir "${work_dir}" 2>&1 | \
        tee -a "${log_file}"

    local end_time=$(date +%s)
    local elapsed_time=$((end_time - start_time))

    printf "=== Training completed: C=%s, elapsed time: %s ===\n" "${c}" "$(date -u -d @${elapsed_time} +'%H:%M:%S')"
}

run_experiment_set1() {
    local c="64,96,128,192"
    local model="SwinJSCC_w/_SAandRA"
    local channel_type="awgn"
    local snr_set="1,4,7,10,13"
    local metric="MSE"
    local model_size="base"
    local dataset_name="NIGHTS"
    local test_snr="10"
    local test_c="8,32,96,192"
    run_training "${c}" "${model}" "${channel_type}" "${snr_set}" "${metric}" "${model_size}" "${dataset_name}" "${test_snr}" "${test_c}"
}



# run_experiment_set2
run_experiment_set1