#!/bin/bash
# SwinJSCC 测试脚本 

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "ConSwinjscc_evaluation_logs/NIGHTS/"
mkdir -p "ConSwinjscc_evaluation/NIGHTS/"

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

    local work_dir="./ConSwinjscc_evaluation/NIGHTS/${TIMESTAMP}_C${c}_${channel_type}_snr${snr_set//,/_}_${model//\//_}_${metric}"
    local log_file="./ConSwinjscc_evaluation_logs/NIGHTS/${TIMESTAMP}_C${c}_${channel_type}_snr${snr_set//,/_}_${model//\//_}_${metric}.log"

    mkdir -p "${work_dir}"

    local start_time=$(date +%s)
    python -W ignore::FutureWarning:timm.models.layers eval_con.py \
        --testset MMEB \
        --dataset_name NIGHTS \
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
    local c="128,192"
    local model="SwinJSCC_w/_SAandRA"
    local channel_type="awgn"
    local snr_set="10,13"
    local metric="MSE"
    local model_size="base"
    local model_path="mmeb_condition_training/NIGHTS/20250803_162345_C192_awgn_snr13_SwinJSCC_w__SAandRA_MSE/2025-08-03_16-23-51/models/2025-08-03_16-23-51_EP200.model"
    run_evaluation "${c}" "${model}" "${channel_type}" "${snr_set}" "${metric}" "${model_size}" "${model_path}"
}


# CIRR 数据集
#  stage 1 pretrained . mmeb_condition_training/20250725_152448_C32,64,96,128,192_awgn_snr1_4_7_10_13_SwinJSCC_w__SAandRA_MSE/2025-07-25_15-24-55/models/2025-07-25_15-24-55_EP100.model
#  stage 2 pretrained *. mmeb_condition_training/20250726_123846_C32,64,96,128,192_awgn_snr1_4_7_10_13_SwinJSCC_w__SAandRA_MSE/2025-07-26_12-38-53/models/2025-07-26_12-38-53_EP65.model
#  stage 3 pretrained *. mmeb_condition_training/20250727_123959_C96,128,192_awgn_snr7_10_13_SwinJSCC_w__SAandRA_MSE/2025-07-27_12-40-06/models/2025-07-27_12-40-06_EP100.model



## VisDig 数据集,预训练参数从 CIRR
# test1 pretrained  Stage1 = mmeb_condition_training/CIRR/20250728_120025_C96,128,192_awgn_snr7_10_13_SwinJSCC_w__SAandRA_MSE/2025-07-28_12-00-32/models/2025-07-28_12-00-32_EP30.model
# test1 pretrained * Stage2 = mmeb_condition_training/VisDial/20250729_224638_C128,192_awgn_snr1_4_7_10_13_SwinJSCC_w__SAandRA_MSE/2025-07-29_22-46-45/models/2025-07-29_22-46-45_EP90.model


## 0-1 VisDial 数据集 训练
# Stage1 Pretrained 
# best_model/best_VisDial_Stage1.model
 
# Stage2 Pretrained mmeb_condition_training/VisDial/20250731_151037_C32,64,96,128,192_awgn_snr1_4_7_10_13_SwinJSCC_w__SAandRA_MSE/2025-07-31_15-10-44/models/2025-07-31_15-10-44_EP150.model


# 0-1 Stage1 测试

#NIGHTS pretrained : mmeb_condition_training/NIGHTS/20250803_162345_C192_awgn_snr13_SwinJSCC_w__SAandRA_MSE/2025-08-03_16-23-51/models/2025-08-03_16-23-51_EP200.model


run_evaluation_set1