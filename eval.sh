#!/bin/bash
# SwinJSCC 测试脚本 

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "mmeb_kodak_evaluation_logs/"

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

    local work_dir="./mmeb_kodak_evaluation/${TIMESTAMP}_C${c}_${channel_type}_snr${snr_set//,/_}_${model//\//_}_${metric}"
    local log_file="mmeb_kodak_evaluation_logs/${TIMESTAMP}_C${c}_${channel_type}_snr${snr_set//,/_}_${model//\//_}_${metric}.log"

    mkdir -p "${work_dir}"

    local start_time=$(date +%s)
    python -W ignore::FutureWarning:timm.models.layers eval.py \
        --trainset MMEB_Kodak \
        --model_path "${model_path}" \
        --distortion-metric "${metric}" \
        --model "${model}" \
        --channel-type "${channel_type}" \
        --C "${c}" \
        --multiple-snr "${snr_set}" \
        --model_size "${model_size}" \
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
    local model_path="mmeb_kodak/20250715_134416_C32,64,96,128,192_awgn_snr1_4_7_10_13_SwinJSCC_w__SAandRA_MSE/2025-07-15_13-44-23/models/2025-07-15_13-44-23_EP200.model"
    run_evaluation "${c}" "${model}" "${channel_type}" "${snr_set}" "${metric}" "${model_size}" "${model_path}"
}

run_evaluation_set2() {
    local model="SwinJSCC_w/_SA"
    local c="96"
    local channel_type="awgn"
    local snr_set="1,4,7,10,13"
    local metric="MSE"
    local model_size="base"
    local model_path="mmeb_kodak/20250717_003514_C96_awgn_snr1_4_7_10_13_SwinJSCC_w__SA_MSE/2025-07-17_00-35-20/models/2025-07-17_00-35-20_EP200.model"
    run_evaluation "${c}" "${model}" "${channel_type}" "${snr_set}" "${metric}" "${model_size}" "${model_path}"
}

run_evaluation_set3() {
    # Experiment set 3
    local channel_type="awgn"
    local c="32,64,96,128,192"
    local model="SwinJSCC_w/_RA"
    local snr_set="10"
    local metric="MSE"
    local model_size="base"
    local model_path="mmeb_kodak/20250718_230737_C32,64,96,128,192_awgn_snr10_SwinJSCC_w__RA_MSE/2025-07-18_23-07-43/models/2025-07-18_23-07-43_EP100.model"
    run_evaluation "${c}" "${model}" "${channel_type}" "${snr_set}" "${metric}" "${model_size}" "${model_path}"
}

run_evaluation_set1
run_evaluation_set2
run_evaluation_set3