#!/bin/bash
# SwinJSCC 训练脚本 - 函数封装版

# 创建时间戳和日志目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "mmeb_condition_training/"
mkdir -p "mmeb_condition_training_logs/"

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
    local epoch=$8

    local work_dir="mmeb_condition_training/${dataset_name}/${TIMESTAMP}_C${c}_${channel_type}_snr${snr_set//,/_}_${model//\//_}_${metric}"
    local log_file="mmeb_condition_training_logs/${dataset_name}/${TIMESTAMP}_C${c}_${channel_type}_snr${snr_set//,/_}_${model//\//_}_${metric}.log"

    mkdir -p "${work_dir}"
    mkdir -p "$(dirname "${log_file}")"

    local start_time=$(date +%s)
    python -W ignore::FutureWarning:timm.models.layers ddp_con_train.py \
        --training \
        --testset MMEB \
        --dataset_name "${dataset_name}" \
        --distortion-metric "${metric}" \
        --model "${model}" \
        --channel-type "${channel_type}" \
        --C "${c}" \
        --multiple-snr "${snr_set}" \
        --model_size "${model_size}" \
        --epoch ${epoch} \
        --workdir "${work_dir}" 2>&1 | \
        tee -a "${log_file}"

    local end_time=$(date +%s)
    local elapsed_time=$((end_time - start_time))

    printf "=== Training completed: C=%s, elapsed time: %s ===\n" "${c}" "$(date -u -d @${elapsed_time} +'%H:%M:%S')"
}

run_experiment_set1() {
    local c="32,64,96,128,192"
    local model="SwinJSCC_w/_SAandRA"
    local channel_type="awgn"
    local snr_set="1,4,7,10,13"
    local metric="MSE"
    local model_size="base"
    local dataset_name="VisDial"
    local epoch=200
    run_training "${c}" "${model}" "${channel_type}" "${snr_set}" "${metric}" "${model_size}" "${dataset_name}" ${epoch}
}


run_experiment_Stage1() {
    local c="192"
    local model="SwinJSCC_w/_SAandRA"
    local channel_type="awgn"
    local snr_set="13"
    local metric="MSE"
    local model_size="base"
    local dataset_name="NIGHTS"
    local epoch=1000
    run_training "${c}" "${model}" "${channel_type}" "${snr_set}" "${metric}" "${model_size}" "${dataset_name}" ${epoch}
}

run_experiment_Stage2() {
    local c="32,64,96,128,192"
    local model="SwinJSCC_w/_SAandRA"
    local channel_type="awgn"
    local snr_set="1,4,7,10,13"
    local metric="MSE"
    local model_size="base"
    local dataset_name="VisDial"
    local epoch=200
    run_training "${c}" "${model}" "${channel_type}" "${snr_set}" "${metric}" "${model_size}" "${dataset_name}" ${epoch}
}

# run_experiment_set1
# run_experiment_set1_stage2


# VisDial Stage1
# run_experiment_Stage1  * 从0 开始： mmeb_condition_training/VisDial/20250730_120021_C128,192_awgn_snr1_4_7_10_13_SwinJSCC_w__SAandRA_MSE/2025-07-30_12-00-27/models/2025-07-30_12-00-27_EP50.model
# run_experiment_Stage1  * 最好的模型是 mmeb_condition_training/VisDial/20250730_230745_C128,192_awgn_snr1_4_7_10_13_SwinJSCC_w__SAandRA_MSE/2025-07-30_23-07-51/models/2025-07-30_23-07-51_EP80.model



# 
run_experiment_Stage1

# pretrained channel 192
# mmeb_condition_training/NIGHTS/20250806_214619_C192_awgn_snr1_4_7_10_13_SwinJSCC_w__SAandRA_MSE/2025-08-06_21-46-25/models/checkpoint_ep130_snr_13_rate_192_best_psnr_33.2674.pth

# pretrained channel 128,192
#  mmeb_condition_training/NIGHTS/20250807_142000_C128,192_awgn_snr1_4_7_10_13_SwinJSCC_w__SAandRA_MSE/2025-08-07_14-20-06/models/checkpoint_ep0_snr_13_rate_192_best_psnr_33.1316.pth

# pretrained channel 128,192
# mmeb_condition_training/NIGHTS/20250811_234122_C128,192_awgn_snr1_4_7_10_13_SwinJSCC_w__SAandRA_MSE/2025-08-11_23-41-29/models/checkpoint_ep190_snr_1_rate_192_best_psnr_28.3732.pth


# WebQA

# mmeb_condition_training/WebQA/20250821_225958_C192_awgn_snr13_SwinJSCC_w__SAandRA_MSE/2025-08-21_23-00-04/models/checkpoint_ep280_snr_13_rate_192_best_psnr_34.5535.pth

# channel 192，snr 13
# mmeb_condition_training/WebQA/20250824_195700_C192_awgn_snr13_SwinJSCC_w__SAandRA_MSE/2025-08-24_19-57-06/models/checkpoint_ep460_snr_13_rate_192_best_psnr_34.8860.pth



# run_experiment_set1