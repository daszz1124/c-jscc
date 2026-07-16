import os
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import torchvision
import numpy as np
from pytorch_msssim import ms_ssim as ms_ssim_func
from loss.distortion import MS_SSIM
from model.network import SwinJSCC
from data.mmeb_datasets import *

from utils import logger_configuration, seed_torch
import warnings
import time
import csv

from vlm2vec_service import GenerateImageIntentEvaluator

warnings.filterwarnings("ignore", category=FutureWarning,
                        module="timm.models.layers")


def parse_args():
    parser = argparse.ArgumentParser(description='SwinJSCC Test')
    parser.add_argument('--testset', type=str, default='MMEB',
                        choices=['Kodak', 'MMEB'], help='Train dataset name')
    parser.add_argument('--distortion-metric', type=str, default='MSE',
                        choices=['MSE', 'MS-SSIM'], help='trainset metric')
    parser.add_argument('--dataset_name', type=str, default='NIGHTS',
                        choices=['CIRR', 'VisDial', 'NIGHTS','WebQA','VisualNews_t2i'],
                        help='Dataset name for MMEB')
    parser.add_argument('--model', type=str, default='SwinJSCC_w/_SAandRA',
                        choices=['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA',
                                 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'],
                        help='SwinJSCC model variant')
    parser.add_argument('--channel-type', type=str, default='awgn',
                        choices=['awgn', 'rayleigh'], help='Wireless channel model')
    parser.add_argument('--C', type=str, default='96',
                        help='Bottleneck dimension (comma-separated for test)')
    parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13',
                        help='SNR values (comma-separated for test)')
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'], help='SwinJSCC model size')
    parser.add_argument('--model_path', type=str,
                        default="checkpoint/Swinjscc/SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model")
    parser.add_argument('--work_dir', type=str, default='./eval_results',
                        help='Path to save test images')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--print_step', type=int, default=100,
                        help='Frequency of logging during testing')
    return parser.parse_args()


class Config:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.print_step = args.print_step
        self.pass_channel = True
        self.filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.workdir = f"{args.work_dir}/{self.filename}/model_{args.model}_C{args.C}_channel_{args.channel_type}_snr{args.multiple_snr}_size_{args.model_size}_{args.distortion_metric}"
        os.makedirs(self.workdir, exist_ok=True)
        self.models = f"{self.workdir}/models"
        os.makedirs(self.models, exist_ok=True)
        self.log = f"{self.workdir}/Log_{self.filename}.log"
        self.samples = f"{self.workdir}/samples"
        os.makedirs(self.samples, exist_ok=True)

        self.image_dims = (3, 256, 256)
        self.train_data_dir = "/mnt/d/zsd/project/c-jscc/datasets/vlm2vec/MMEB-Datasets/MMEB-train"
        self.image_dir = "/mnt/d/zsd/project/c-jscc/datasets/vlm2vec/MMEB-Datasets/images"
        self.dataset_name = args.dataset_name

        if args.testset == "MMEB":
            self.test_data_dir = "/mnt/d/zsd/project/c-jscc/datasets/vlm2vec/MMEB-Test/MMEB-eval"
            self.test_image_dir = "/mnt/d/zsd/project/c-jscc/datasets/vlm2vec/MMEB-Test/eval_image"
        elif args.testset == "Kodak":
            self.test_data_dir = [
                "/home/iisc/zsd/project/VG2SC/SwinJSCC/datasets/kodak/"]

        self.batch_size = 16
        self.downsample = 4
        self.channel_number = int(args.C) if args.model in [
            'SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA'] else None

        model_configs = {
            'small': {
                'encoder_kwargs': dict(img_size=self.image_dims[1:], patch_size=2, in_chans=3,
                                       embed_dims=[128, 192, 256, 320], depths=[2, 2, 2, 2], num_heads=[4, 6, 8, 10],
                                       C=self.channel_number, window_size=8, mlp_ratio=4., qkv_bias=True,
                                       norm_layer=nn.LayerNorm, patch_norm=True),
                'decoder_kwargs': dict(img_size=self.image_dims[1:], embed_dims=[320, 256, 192, 128],
                                       depths=[2, 2, 2, 2], num_heads=[10, 8, 6, 4], C=self.channel_number,
                                       window_size=8, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm, patch_norm=True)
            },
            'base': {
                'encoder_kwargs': dict(img_size=self.image_dims[1:], patch_size=2, in_chans=3,
                                       embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
                                       C=self.channel_number, window_size=8, mlp_ratio=4., qkv_bias=True,
                                       norm_layer=nn.LayerNorm, patch_norm=True),
                'decoder_kwargs': dict(img_size=self.image_dims[1:], embed_dims=[320, 256, 192, 128],
                                       depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], C=self.channel_number,
                                       window_size=8, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm, patch_norm=True)
            },
            'large': {
                'encoder_kwargs': dict(img_size=self.image_dims[1:], patch_size=2, in_chans=3,
                                       embed_dims=[128, 192, 256, 320], depths=[2, 2, 18, 2], num_heads=[4, 6, 8, 10],
                                       C=self.channel_number, window_size=8, mlp_ratio=4., qkv_bias=True,
                                       norm_layer=nn.LayerNorm, patch_norm=True),
                'decoder_kwargs': dict(img_size=self.image_dims[1:], embed_dims=[320, 256, 192, 128],
                                       depths=[2, 18, 2, 2], num_heads=[10, 8, 6, 4], C=self.channel_number,
                                       window_size=8, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm, patch_norm=True)
            }
        }
        self.encoder_kwargs = model_configs[args.model_size]['encoder_kwargs']
        self.decoder_kwargs = model_configs[args.model_size]['decoder_kwargs']


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.reset()


def calculate_psnr(mse):
    if mse > 0:
        return 10 * (torch.log10(255. * 255. / mse))
    return 100


def load_weights(net, model_path):
    pretrained = torch.load(model_path)
    new_state_dict = {}
    for key, value in pretrained.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    net.load_state_dict(new_state_dict, strict=False)
    return net


def save_2d_array_to_csv(array, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(array)

def test(net, test_loader, config, logger, evaluator, args):
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs,similaritys,qry_similaritys = [AverageMeter() for _ in range(7)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs,similaritys,qry_similaritys]
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to(config.device)
    multiple_snr = [int(snr) for snr in args.multiple_snr.split(",")]
    channel_number = [int(c) for c in args.C.split(",")]
    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))
    results_similarity = np.zeros((len(multiple_snr), len(channel_number)))
    results_qry_similarity = np.zeros((len(multiple_snr), len(channel_number)))

    result_save_dir = os.path.join(config.samples, 'results')
    os.makedirs(result_save_dir, exist_ok=True)

    for i, test_snr in enumerate(multiple_snr):
        for j, rate in enumerate(channel_number):
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    test_samples_dir = os.path.join(
                        config.samples, f"test_SNR{test_snr}_Rate{rate}")  # 使用test_snr
                    os.makedirs(test_samples_dir, exist_ok=True)

                    start_time = time.time()
                    input, names = batch[0], batch[2]
                    qry_text, qry_image_path, tgt_text, tgt_image_path = batch[
                        3], batch[4], batch[5], batch[6]
                    input = input.to(config.device)
                    recon_image, CBR, actual_snr, mse, loss_G = net(
                        input, test_snr, rate)

                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(actual_snr)  # 使用actual_snr更新指标
                    psnr = calculate_psnr(mse)
                    psnrs.update(psnr)
                    msssim = 1 - \
                        CalcuSSIM(input, recon_image.clamp(
                            0., 1.)).mean().item()
                    msssims.update(msssim)

                    dataset_name = args.dataset_name
                    os.makedirs(os.path.join(test_samples_dir,
                                dataset_name), exist_ok=True)

                    recon_image_paths = []
                    for k in range(recon_image.shape[0]):
                        recon_image_path = os.path.join(
                            test_samples_dir, dataset_name,
                            names[k][:-4] +
                            f"_regon_psnr{psnr:.5f}_msssim{msssim:.5f}.png"
                        )
                        torchvision.utils.save_image(
                            recon_image[k],
                            recon_image_path
                        )
                        recon_image_paths.append(recon_image_path)

                    # similarity = evaluator.evaluate_similarity(
                    #     qry_text, qry_image_path, tgt_text, tgt_image_path
                    # )

                    similarity = evaluator.evaluate_similarity(
                        tgt_text, tgt_image_path, tgt_text, recon_image_paths
                    )

                    qry_similarity = evaluator.evaluate_similarity(
                        qry_text, qry_image_path, tgt_text, recon_image_paths
                    )

                    similaritys.update(similarity)
                    qry_similaritys.update(qry_similarity)

                    log = (' | '.join([
                        f'Step [{batch_idx + 1}/{len(test_loader)}]',
                        f'Time {elapsed.val:.3f}s',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Similarity {similaritys.val:.3f} ({similaritys.avg:.3f})',
                        f'Qry Similarity {qry_similaritys.val:.3f} ({qry_similaritys.avg:.3f})'
                    ]))
                    logger.info(f'Test SNR={test_snr}, Rate={rate} with image  {names[0]}: {log}')

                results_snr[i, j] = snrs.avg
                results_cbr[i, j] = cbrs.avg
                results_psnr[i, j] = psnrs.avg
                results_msssim[i, j] = msssims.avg
                results_similarity[i,j] = similaritys.avg
                results_qry_similarity[i,j] = qry_similaritys.avg

                logger.info(f"Test SNR={test_snr}, Rate={rate} Summary:")
                log = (' | '.join([
                    f'SNR {snrs.avg:.1f}',
                    f'CBR {cbrs.avg:.4f}',
                    f'PSNR {psnrs.avg:.3f}',
                    f'MSSSIM {msssims.avg:.3f}',
                    f'similarity {similaritys.avg:.3f}',
                    f'qry_similaritys {qry_similaritys.avg:.3f}'
                ]))

                logger.info(f'Test Summary SNR={test_snr}, Rate={rate}: {log}')

                for meter in metrics:
                    meter.clear()

    if config.device.type == 'cuda':
        logger.info(f"SNR: {results_snr.tolist()}")
        logger.info(f"CBR: {results_cbr.tolist()}")
        logger.info(f"PSNR: {results_psnr.tolist()}")
        logger.info(f"MS-SSIM: {results_msssim.tolist()}")
        logger.info(f"Similarity: {results_similarity.tolist()}")
        logger.info(f"Qry Similarity: {results_qry_similarity.tolist()}")
        logger.info("Finish Test!")

        save_2d_array_to_csv(results_psnr.tolist(),
                             os.path.join(result_save_dir, f'psnr.csv'))
        save_2d_array_to_csv(results_msssim.tolist(),
                             os.path.join(result_save_dir, f'msssim.csv'))
        save_2d_array_to_csv(results_similarity.tolist(),
                             os.path.join(result_save_dir, f'similarity.csv'))
        save_2d_array_to_csv(results_qry_similarity.tolist(),
                             os.path.join(result_save_dir, f'Qry_similarity.csv'))

    return results_psnr.mean(), results_msssim.mean(), 0.0, results_cbr.mean()


def main(opts):
    config = Config(opts)
    logger = logger_configuration(config, save_log=True)
    if config.device.type == 'cuda':
        logger.info(config.__dict__)

    _, test_dataset = load_for_eval_dataset_mmeb(opts, config)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=opts.num_workers, pin_memory=True
    )

    net = SwinJSCC(opts, config)
    net = load_weights(net, opts.model_path)
    net = net.to(config.device)

    evaluator = GenerateImageIntentEvaluator()

    mean_psnr, mean_ms_ssim, _, mean_cbr = test(
        net, test_loader, config, logger, evaluator, opts)

    print(f"Mean PSNR: {mean_psnr:.4f}")
    print(f"Mean MS-SSIM: {mean_ms_ssim:.4f}")
    print(f"Mean CBR: {mean_cbr:.4f}")


if __name__ == "__main__":
    opts = parse_args()
    seed_torch(seed=opts.seed)
    torch.backends.cudnn.benchmark = True
    main(opts)
