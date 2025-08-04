#!/bin/bash

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 设置保存路径
SAVE_PATH="vector_reconstruction"
mkdir -p "${SAVE_PATH}"

LOG_FILE="${SAVE_PATH}/training_${TIMESTAMP}.log"

DATASET_NAME="VisDial"
SPLIT="original"

touch "${LOG_FILE}"

echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "Training started at ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

train_model() {
    local latent_dim=$1
    local epochs=$2
    local batch_size=$3
    local noise_std=$4
    local lambda_nce=$5
    local alpha=$6
    local beta=$7

    echo "----------------------------------------" | tee -a "${LOG_FILE}"
    echo "Training with parameters:" | tee -a "${LOG_FILE}"
    echo "Latent dimension: ${latent_dim}" | tee -a "${LOG_FILE}"
    echo "Epochs: ${epochs}" | tee -a "${LOG_FILE}"
    echo "Batch size: ${batch_size}" | tee -a "${LOG_FILE}"
    echo "Lambda_NCE: ${lambda_nce}" | tee -a "${LOG_FILE}"
    echo "Noise STD: ${noise_std}" | tee -a "${LOG_FILE}"
    echo "----------------------------------------" | tee -a "${LOG_FILE}"

    python vector_reconstruction.py \
        --is_train True \
        --dataset_name "${DATASET_NAME}" \
        --split "${SPLIT}" \
        --latent_dim "${latent_dim}" \
        --epoch "${epochs}" \
        --batch_size "${batch_size}" \
        --learning_rate 1e-3 \
        --noise_std "${noise_std}" \
        --lambda_nce "${lambda_nce}" \
        --alpha "${alpha}" \
        --beta "${beta}" \
        --save_dir "${SAVE_PATH}" 2>&1 | tee -a "${LOG_FILE}"

    echo "✅ Training completed for latent_dim=${latent_dim}, lambda_nce=${lambda_nce}" | tee -a "${LOG_FILE}"
}

# latent_dim

# infonce loss 和 recon loss
train_model 256 500 256 0.1 1.00 0.5 1.00
train_model 512 500 512 0.1 1.00 0.5 1.00
train_model 1024 500 1024 0.1 1.00 0.5 1.00

# 只用infonce loss 监督
train_model 256 500 256 0.1 1.00 0.5 0.0
train_model 512 500 512 0.1 1.00 0.5 0.0
train_model 1024 500 1024 0.1 1.00 0.5 0.0



# 缩小 regon loss
train_model 256 500 256 0.1 1.00 0.5 0.2
train_model 512 500 512 0.1 1.00 0.5 0.2
train_model 1024 500 1024 0.1 1.00 0.5 0.2


# batch_size
train_model 512 500 256 0.1 1.00 0.5 1.00
train_model 512 500 512 0.1 1.00 0.5 1.00
train_model 512 500 1024 0.1 1.00 0.5 1.00


train_model 256 500 256 0.1 1.00 0.5 1.00
train_model 512 500 512 0.1 1.00 0.5 1.00
train_model 1024 500 1024 0.1 1.00 0.5 1.00




