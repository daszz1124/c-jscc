#!/bin/bash
# SwinJSCC 测试脚本 

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "Swinjscc_evaluation_acc_logs/"
mkdir -p "Swinjscc_evaluation_acc/"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

run_evaluation() {
    local c=$1
    local model=$2
    local channel_type=$3
    local snr_set=$4
    local metric=$5
    local model_size=$6
    local model_path=$7
    local dataset_name=$8

    local work_dir="./Swinjscc_evaluation_acc/${TIMESTAMP}_C${c}_${channel_type}_snr${snr_set//,/_}_${model//\//_}_${metric}"
    local log_file="./Swinjscc_evaluation_acc_logs/${TIMESTAMP}_C${c}_${channel_type}_snr${snr_set//,/_}_${model//\//_}_${metric}.log"

    mkdir -p "${work_dir}"

    local start_time=$(date +%s)
    python -W ignore::FutureWarning:timm.models.layers eval_acc.py \
        --testset MMEB \
        --model_path "${model_path}" \
        --distortion-metric "${metric}" \
        --model "${model}" \
        --channel-type "${channel_type}" \
        --C "${c}" \
        --multiple-snr "${snr_set}" \
        --model_size "${model_size}" \
        --dataset_name "${dataset_name}" \
        --work_dir "${work_dir}" 2>&1 | \
        tee -a "${log_file}"

    local end_time=$(date +%s)
    local elapsed_time=$((end_time - start_time))

    printf "=== evaluation completed: C=%s, elapsed time: %s ===\n" "${c}" "$(date -u -d @${elapsed_time} +'%H:%M:%S')"
}

run_evaluation_set1() {
    local c="32,64,96,128,192"
    local model="SwinJSCC_w/_SAandRA"
    local channel_type="awgn"
    local snr_set="1,4,7,10,13"
    local metric="MSE"
    local model_size="base"
    local model_path="checkpoint/Swinjscc/pretrained_SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr13.model"
    local dataset_name=$1
    run_evaluation "${c}" "${model}" "${channel_type}" "${snr_set}" "${metric}" "${model_size}" "${model_path}" "${dataset_name}"
}


run_evaluation_set2() {
    local c="8,16,32,64,96,128,192"
    local model="SwinJSCC_w/_SAandRA"
    local channel_type="awgn"
    local snr_set="1,4,7,10,13"
    local metric="MSE"
    local model_size="base"
    local model_path="checkpoint/Swinjscc/SwinJSCC_NIGHTS_SNR1-10_Rate96-192_EP260.model"
    local dataset_name=$1
    run_evaluation "${c}" "${model}" "${channel_type}" "${snr_set}" "${metric}" "${model_size}" "${model_path}" "${dataset_name}"
}


# run_evaluation_set2 "CIRR"
run_evaluation_set2 "NIGHTS"
