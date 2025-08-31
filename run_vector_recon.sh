#!/bin/bash

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 设置保存路径
SAVE_PATH="vector_channel_compress"

# NIGHTS
mkdir -p "${SAVE_PATH}"



# DATASET_NAME="VisualNews_t2i"
SPLIT="original"



train_model() {
    local latent_dim=$1
    local epochs=$2
    local batch_size=$3
    local noise_std=$4
    local lambda_nce=$5
    local alpha=$6
    local beta=$7
    local dataset_name=$8

    LOG_FILE="${SAVE_PATH}/snr_${noise_std}/training_${TIMESTAMP}.log"

    mkdir -p "${SAVE_PATH}/snr_${noise_std}"
    local save_path="${SAVE_PATH}/snr_${noise_std}"

    touch "${LOG_FILE}"
    echo "----------------------------------------" | tee -a "${LOG_FILE}"
    echo "Training started at ${TIMESTAMP}" | tee -a "${LOG_FILE}"
    echo "----------------------------------------" | tee -a "${LOG_FILE}"
    echo "Training with parameters:" | tee -a "${LOG_FILE}"
    echo "Latent dimension: ${latent_dim}" | tee -a "${LOG_FILE}"
    echo "Epochs: ${epochs}" | tee -a "${LOG_FILE}"
    echo "Batch size: ${batch_size}" | tee -a "${LOG_FILE}"
    echo "Lambda_NCE: ${lambda_nce}" | tee -a "${LOG_FILE}"
    echo "Noise STD: ${noise_std}" | tee -a "${LOG_FILE}"
    echo "----------------------------------------" | tee -a "${LOG_FILE}"

    # # 
    python vector_reconstruction.py \
        --is_train True \
        --dataset_name "${dataset_name}" \
        --split "${SPLIT}" \
        --latent_dim "${latent_dim}" \
        --epoch "${epochs}" \
        --batch_size "${batch_size}" \
        --learning_rate 1e-3 \
        --noise_std "${noise_std}" \
        --lambda_nce "${lambda_nce}" \
        --alpha "${alpha}" \
        --beta "${beta}" \
        --save_dir "${save_path}" 2>&1 | tee -a "${LOG_FILE}"

    echo "✅ Training completed for latent_dim=${latent_dim}, lambda_nce=${lambda_nce}" | tee -a "${LOG_FILE}"
}

run_batch(){
    local dataset_name=$1
    local noise_std=$2
    # train_model 1536 100 512 "-1" 1.00 0.5 0.0 $dataset_name
    train_model 1024 50 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
    train_model 768 50 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
    train_model 512 50 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
    train_model 384 50 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
    train_model 256 50 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
    train_model 192 50 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
    train_model 128 50 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
    train_model 96 50 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
    train_model 64 50 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
    train_model 32 50 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
    train_model 16 50 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
    train_model 8 50 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
}

run_simmple_test() {
    local dataset_name=$1
    train_model 1024 100 1024 ${noise_std} 1.00 0.5 0.0 $dataset_name
}


# run_simmple_test "CIRR"

# run_batch_no_model "VisDial"
# run_batch_no_model "CIRR"
# run_batch_no_model "NIGHTS"
# run_batch_no_model "VisualNews_t2i"
# run_batch_no_model "WebQA"

 
run_all_snr(){
    local noise_std=$1
    # run_batch  "NIGHTS" ${noise_std}
    # run_batch  "VisualNews_t2i" ${noise_std}
    # run_batch  "WebQA" ${noise_std} 
    run_batch  "CIRR" ${noise_std} 
    run_batch  "VisDial" ${noise_std}
}


# run_all_snr 10
# run_all_snr 7
# run_all_snr 4
# run_all_snr 1

# run_all_snr 0
# run_all_snr "-1"
# run_all_snr "-4"
run_all_snr "-7"
# run_all_snr "-10"


