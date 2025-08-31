import os
import numpy as np
import cv2
import zlib
from scipy.special import erfc
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class MMEBTestDataset(Dataset):
    def __init__(self, base_dir, dataset_name, image_dir, split='test'):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.split = split
        self.image_dir = image_dir
        self.file_map = {
            'test': 'test-00000-of-00001.parquet'
        }
        if split not in self.file_map:
            raise ValueError(f"Unsupported split: {split}")
        self.data_dir = os.path.join(base_dir, dataset_name)
        parquet_file = self.file_map[split]
        parquet_path = os.path.join(self.data_dir, parquet_file)
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        self.df = pd.read_parquet(parquet_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["tgt_img_path"][0]
        pos_image_path = os.path.join(self.image_dir, name)
        image_name = os.path.basename(pos_image_path)
        image = Image.open(pos_image_path).convert('RGB')
        im_height, im_width = image.size
        if im_height % 128 != 0 or im_width % 128 != 0:
            im_height = im_height - im_height % 128
            im_width = im_width - im_width % 128
        transform = transforms.Compose([
            transforms.CenterCrop((im_width, im_height)),
            transforms.ToTensor()])
        img = transform(image)
        return img, image_name

    def __len__(self):
        return len(self.df)

class LDPC:
    def __init__(self, code_rate=0.5):
        self.code_rate = code_rate
        
    def encode(self, data):
        n = int(len(data) / self.code_rate)
        encoded = np.zeros(n, dtype=np.uint8)
        encoded[:len(data)] = data
        encoded[len(data):] = np.random.randint(0, 2, size=int(n-len(data)))
        return encoded
        
    def decode(self, received_data, snr_db):
        snr = 10 ** (snr_db / 10.0)
        sigma = 1 / np.sqrt(2 * snr)
        ber = 0.5 * erfc(np.sqrt(snr))
        errors = np.random.binomial(1, ber, size=len(received_data))
        decoded = received_data ^ errors
        if self.code_rate < 1.0:
            data_length = int(len(decoded) * self.code_rate)
            return decoded[:data_length]
        return decoded

def png_compress(image):
    _, buffer = cv2.imencode('.png', image)
    return buffer.tobytes()

def png_decompress(compressed_data, shape, dtype):
    try:
        nparr = np.frombuffer(compressed_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 确保为 RGB/BGR
        if image.shape != shape:
            return None
        return image.astype(dtype)
    except:
        return None

def bytes_to_bits(byte_data):
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    bits = np.unpackbits(byte_array)
    return bits

def bits_to_bytes(bit_data):
    if len(bit_data) % 8 != 0:
        pad = 8 - (len(bit_data) % 8)
        bit_data = np.pad(bit_data, (0, pad), mode='constant')
    return np.packbits(bit_data).tobytes()

def simulate_transmission(image, cbr, snr_db, ldpc):
    original_shape = image.shape
    original_dtype = image.dtype
    compressed_data = png_compress(image)
    target_bits = int(cbr * image.size)
    compressed_bits = bytes_to_bits(compressed_data)
    if len(compressed_bits) > target_bits:
        compressed_bits = compressed_bits[:target_bits]
    elif len(compressed_bits) < target_bits:
        compressed_bits = np.pad(compressed_bits, (0, target_bits - len(compressed_bits)), mode='constant')
    encoded_bits = ldpc.encode(compressed_bits)
    received_bits = encoded_bits
    decoded_bits = ldpc.decode(received_bits, snr_db)
    recovered_image = png_decompress(bits_to_bytes(decoded_bits), original_shape, original_dtype)
    if recovered_image is None:
        return None, 0.0, 0.0, 0.0
    image_torch = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    recovered_torch = torch.from_numpy(recovered_image).permute(2, 0, 1).float() / 255.0
    image_torch = image_torch.unsqueeze(0)
    recovered_torch = recovered_torch.unsqueeze(0)
    current_msssim = ms_ssim(image_torch, recovered_torch, data_range=1.0).item()
    if len(original_shape) == 3:
        original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        recovered_gray = cv2.cvtColor(recovered_image, cv2.COLOR_BGR2GRAY)
        current_psnr = psnr(image, recovered_image, data_range=255)
        current_ssim = ssim(original_gray, recovered_gray, data_range=255)
    else:
        current_psnr = psnr(image, recovered_image, data_range=255)
        current_ssim = ssim(image, recovered_image, data_range=255)
    return recovered_image, current_psnr, current_ssim, current_msssim

def main():
    base_dir = "/home/iisc/zsd/project/VG2SC/MMEB-Datasets/MMEB-Datasets/MMEB-eval"
    dataset_name = 'CIRR'
    image_dir = "/home/iisc/zsd/project/VG2SC/MMEB-Datasets/eval_images"
    batch_size = 1
    max_images = 10 

    try:
        test_dataset = MMEBTestDataset(base_dir, dataset_name, image_dir, split='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"成功加载数据集，包含 {len(test_dataset)} 张图像")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("生成单张测试图像...")
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (400, 400), (0, 255, 0), -1)
        cv2.circle(image, (256, 256), 100, (0, 0, 255), -1)
        test_dataset = [(torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0, "synthetic.png")]
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    snr_values = [1, 4, 7, 10, 13]
    cbr_values = [0.0208, 0.0416, 0.0625, 0.0833, 0.125, 0.2, 0.5]  # 添加更高 CBR
    code_rates = [0.3, 0.5, 0.7]

    results = []
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    print("开始仿真...")
    for code_rate in code_rates:
        ldpc = LDPC(code_rate)
        for cbr in cbr_values:
            for snr in snr_values:
                psnr_values = []
                msssim_values = []
                print(f"处理 Code Rate={code_rate:.1f}, CBR={cbr:.4f} bpp, SNR={snr} dB...")
                for img_idx, (img_tensor, img_name) in enumerate(test_loader):
                    if img_idx >= max_images:  # 限制图像数量
                        break
                    img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    recovered_img, psnr_val, ssim_val, msssim_val = simulate_transmission(
                        img_np, cbr, snr, ldpc)
                    if recovered_img is not None:
                        psnr_values.append(psnr_val)
                        msssim_values.append(msssim_val)
                    print(f"  图像 {img_name[0]}: PSNR={psnr_val:.2f} dB, MS-SSIM={msssim_val:.4f}")
                
                avg_psnr = np.mean(psnr_values) if psnr_values else 0.0
                avg_msssim = np.mean(msssim_values) if msssim_values else 0.0
                results.append({
                    'code_rate': code_rate,
                    'cbr': cbr,
                    'snr': snr,
                    'avg_psnr': avg_psnr,
                    'avg_msssim': avg_msssim
                })
                # 保存中间结果
                pd.DataFrame(results).to_csv(os.path.join(output_dir, "compression_results.csv"), index=False)

    print("\n测试集平均性能结果汇总:")
    print(f"{'Code Rate':<12} {'CBR (bpp)':<12} {'SNR (dB)':<10} {'平均PSNR (dB)':<15} {'平均MS-SSIM':<12}")
    print("-" * 62)
    for item in results:
        print(f"{item['code_rate']:<12.1f} {item['cbr']:<12.4f} {item['snr']:<10d} {item['avg_psnr']:<15.2f} {item['avg_msssim']:<12.4f}")

    # 可视化结果
    for code_rate in code_rates:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for cbr in cbr_values:
            cbr_results = [item for item in results if item['cbr'] == cbr and item['code_rate'] == code_rate]
            snrs = [item['snr'] for item in cbr_results]
            psnrs = [item['avg_psnr'] for item in cbr_results]
            axes[0].plot(snrs, psnrs, marker='o', label=f'CBR={cbr:.4f} bpp')
        axes[0].set_title(f'平均PSNR随SNR变化 (Code Rate={code_rate:.1f})')
        axes[0].set_xlabel('SNR (dB)')
        axes[0].set_ylabel('平均PSNR (dB)')
        axes[0].grid(True)
        axes[0].legend()
        for cbr in cbr_values:
            cbr_results = [item for item in results if item['cbr'] == cbr and item['code_rate'] == code_rate]
            snrs = [item['snr'] for item in cbr_results]
            msssims = [item['avg_msssim'] for item in cbr_results]
            axes[1].plot(snrs, msssims, marker='s', label=f'CBR={cbr:.4f} bpp')
        axes[1].set_title(f'平均MS-SSIM随SNR变化 (Code Rate={code_rate:.1f})')
        axes[1].set_xlabel('SNR (dB)')
        axes[1].set_ylabel('平均MS-SSIM')
        axes[1].grid(True)
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'testset_compression_results_code_rate_{code_rate:.1f}.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    main()