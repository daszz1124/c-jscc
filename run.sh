# Create logs directory
mkdir -p logs

mkdir -p logs/C

# Loop through different C values
for C in 32 64 96 128 192; do
    python main.py \
        --trainset DIV2K \
        --testset kodak \
        --distortion-metric MSE \
        --model SwinJSCC_w/_SA \
        --channel-type awgn \
        --C 96 \
        --multiple-snr 10 \
        --model_size base \
        --model_path "checkpoint/SwinJSCC w-RA/SwinJSCC_w_RA_AWGN_HRimage_cbr_psnr_snr10.model" \
        > logs/C/C${C}_awgn_snr10.log 2>&1
done


