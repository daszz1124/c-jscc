import os
import random
import sys
import logging
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from datetime import datetime
import torchvision
import time
import numpy as np
from pytorch_msssim import ms_ssim as ms_ssim_func
from torch.utils.tensorboard import SummaryWriter
from loss.distortion import MS_SSIM
from model.network import SwinJSCC
from data.swindatasets import *

from utils import logger_configuration, save_model, seed_torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="timm.models.layers")


def parse_args():
    parser = argparse.ArgumentParser(description='SwinJSCC')
    parser.add_argument('--training', action='store_true',
                        default=False, help='Training or testing')
    parser.add_argument('--trainset', type=str, default='MMEB_Kodak',
                        choices=['MMEB_Kodak', 'MMEB'], help='Train dataset name')
    parser.add_argument('--dataset_name', type=str, default='CIRR',
                        choices=['CIRR'],
                        help='Dataset name for MMEB')
    parser.add_argument('--distortion-metric', type=str, default='MSE',
                        choices=['MSE', 'MS-SSIM'], help='Evaluation metric')
    parser.add_argument('--model', type=str, default='SwinJSCC_w/_SAandRA',
                        choices=['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA',
                                 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'],
                        help='SwinJSCC model variant')
    parser.add_argument('--channel-type', type=str, default='awgn',
                        choices=['awgn', 'rayleigh'], help='Wireless channel model')
    parser.add_argument('--C', type=str, default='96',
                        help='Bottleneck dimension (comma-separated for test)')
    parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,14',
                        help='SNR values (comma-separated for test)')
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'], help='SwinJSCC model size')
    parser.add_argument('--model_path', type=str,
                        default="/home/iisc/zsd/project/VG2SC/SwinJSCC/checkpoint/SwinJSCC_w_SAandRA.model")
    parser.add_argument('--workdir', type=str, default='./workdir')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Total training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--dist_port', type=int, default=12345,
                        help='Port for distributed training')
    parser.add_argument('--print_step', type=int, default=100,
                        help='Frequency of logging during training')
    return parser.parse_args()


class Config:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.pass_channel = True
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.norm = False
        self.print_step = args.print_step
        self.plot_step = 100
        self.filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.workdir = f"{args.workdir}/{self.filename}"
        os.makedirs(self.workdir, exist_ok=True)
        self.log = f"{self.workdir}/Log_{self.filename}.log"
        self.samples = f"{self.workdir}/samples"
        self.models = f"{self.workdir}/models"
        self.tensorboard_dir = f"{self.workdir}/tensorboard"
        os.makedirs(self.samples, exist_ok=True)
        os.makedirs(self.models, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        self.normalize = False
        self.learning_rate = args.lr
        self.tot_epoch = args.epochs
        self.save_model_freq = 20
        self.image_dims = (3, 256, 256)

        self.train_data_dir = "/home/iisc/zsd/project/VG2SC/MMEB-Datasets/MMEB-Datasets/MMEB-train"
        self.image_dir = "/home/iisc/zsd/project/VG2SC/MMEB-Datasets/trainning_images"
        self.dataset_name = args.dataset_name

        if args.trainset == "MMEB":
            self.test_data_dir = "/home/iisc/zsd/project/VG2SC/MMEB-Datasets/MMEB-Datasets/MMEB-eval"
            self.test_image_dir = "/home/iisc/zsd/project/VG2SC/MMEB-Datasets/eval_images"

        elif args.trainset == "MMEB_Kodak":
            self.test_data_dir = [
                "/home/iisc/zsd/project/VG2SC/SwinJSCC/datasets/kodak/"]

        self.batch_size = args.batch_size
        self.downsample = 4
        self.isTrain = args.training
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


def load_weights(net, model_path, device):
    pretrained = torch.load(model_path, map_location=device)
    model_dict = net.state_dict()
    pretrained = {k: v for k, v in pretrained.items(
    ) if k in model_dict and model_dict[k].size() == v.size()}
    net.load_state_dict(pretrained, strict=False)


def train_one_epoch(net, train_loader, optimizer, epoch, config, logger, writer, node_rank, global_step):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [
        AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to(net.device)

    for batch_idx, input in enumerate(train_loader):
        start_time = time.time()
        global_step[0] += 1
        input = input[0].to(net.device) if isinstance(
            input, (list, tuple)) else input.to(net.device)
        recon_image, CBR, SNR, mse, loss_G = net(input)
        loss = loss_G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        cbrs.update(CBR)
        snrs.update(SNR)
        psnr = calculate_psnr(mse)
        psnrs.update(psnr)
        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
        msssims.update(msssim)

        if node_rank == 0 and (global_step[0] % config.print_step) == 0:
            process = (batch_idx + 1) / len(train_loader) * 100.0
            log = (' | '.join([
                f'Epoch {epoch}',
                f'Step [{batch_idx + 1}/{len(train_loader)}={process:.2f}%]',
                f'Time {elapsed.val:.3f}',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                f'Lr {optimizer.param_groups[0]["lr"]}',
            ]))
            logger.info(log)
            writer.add_scalar('Loss/Train', losses.avg, global_step[0])
            writer.add_scalar('Metrics/PSNR_Train', psnrs.avg, global_step[0])
            writer.add_scalar('Metrics/MS-SSIM_Train',
                              msssims.avg, global_step[0])
            writer.add_scalar('Metrics/CBR_Train', cbrs.avg, global_step[0])
            writer.add_scalar('Metrics/SNR_Train', snrs.avg, global_step[0])

    if node_rank == 0:
        writer.add_scalar('Epoch_Metrics/Train_Avg', losses.avg, epoch)
        writer.add_scalar('Epoch_Metrics/PSNR_Train', psnrs.avg, epoch)
        writer.add_scalar('Epoch_Metrics/MS-SSIM_Train', msssims.avg, epoch)
        writer.add_scalar('Epoch_Metrics/CBR_Train', cbrs.avg, epoch)
        writer.add_scalar('Epoch_Metrics/SNR_Train', snrs.avg, epoch)

    for meter in metrics:
        meter.clear()

    return global_step[0]


def test_epoch(net, test_loader, config, logger, writer, epoch, node_rank, args):
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to(net.device)
    multiple_snr = [int(snr) for snr in args.multiple_snr.split(",")]
    channel_number = [int(c) for c in args.C.split(",")]
    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))

    for i, SNR in enumerate(multiple_snr):
        for j, rate in enumerate(channel_number):
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    start_time = time.time()
                    if args.trainset == 'CIFAR10':
                        input, _ = batch
                        input = input.to(net.device)
                    else:
                        input, names = batch
                        input = input.to(net.device)

                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR, rate)

                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    psnr = calculate_psnr(mse)
                    psnrs.update(psnr)
                    msssim = 1 - \
                        CalcuSSIM(input, recon_image.clamp(
                            0., 1.)).mean().item()
                    msssims.update(msssim)

                    if node_rank == 0:
                        test_samples_dir = os.path.join(
                            config.samples, f"test_SNR{SNR}_Rate{rate}_epoch{epoch}/input")
                        os.makedirs(test_samples_dir, exist_ok=True)
                        torchvision.utils.save_image(
                            recon_image, os.path.join(test_samples_dir, names[0]))

                results_snr[i, j] = snrs.avg
                results_cbr[i, j] = cbrs.avg
                results_psnr[i, j] = psnrs.avg
                results_msssim[i, j] = msssims.avg

                if node_rank == 0:
                    log = (' | '.join([
                        f'SNR {snrs.avg:.1f}',
                        f'CBR {cbrs.avg:.4f}',
                        f'PSNR {psnrs.avg:.3f}',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    ]))

                    logger.info(f'Test SNR={SNR}, Rate={rate}: {log}')
                    writer.add_scalar(
                        f'Metrics/PSNR_Test_SNR{SNR}_Rate{rate}', psnrs.avg, epoch)
                    writer.add_scalar(
                        f'Metrics/MS-SSIM_Test_SNR{SNR}_Rate{rate}', msssims.avg, epoch)
                    writer.add_scalar(
                        f'Metrics/CBR_Test_SNR{SNR}_Rate{rate}', cbrs.avg, epoch)
                    writer.add_scalar(
                        f'Metrics/SNR_Test_SNR{SNR}_Rate{rate}', snrs.avg, epoch)

                for meter in metrics:
                    meter.clear()

    if node_rank == 0:
        logger.info(f"SNR: {results_snr.tolist()}")
        logger.info(f"CBR: {results_cbr.tolist()}")
        logger.info(f"PSNR: {results_psnr.tolist()}")
        logger.info(f"MS-SSIM: {results_msssim.tolist()}")
        logger.info("Finish Test!")

    return results_psnr.mean(), results_msssim.mean(), 0.0, results_cbr.mean()


def main(opts):
    config = Config(opts)
    logger = logger_configuration(config, save_log=True)
    node_rank = getattr(opts, "ddp.rank", 0)
    device_id = getattr(opts, "dev.device_id", config.device)

    writer = SummaryWriter(config.tensorboard_dir) if node_rank == 0 else None
    if node_rank == 0:
        logger.info(config.__dict__)

    train_dataset, test_dataset = select_dataset_mmeb(opts, config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=opts.num_workers, pin_memory=True
    )

    # if node_rank == 0:
    #     test_loader = DataLoader(
    #         test_dataset, batch_size=1, shuffle=False, num_workers=opts.num_workers, pin_memory=True
    #     )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=opts.num_workers, pin_memory=True
    )

    net = SwinJSCC(opts, config)
    net = net.to(device_id)
    # Fix 模型中存在未被用于计算损失的参数
    net = DDP(net, device_ids=[device_id],
              output_device=device_id, find_unused_parameters=True)

    # Optimizer
    model_params = [{'params': net.parameters(), 'lr': config.learning_rate}]
    optimizer = optim.Adam(model_params, lr=config.learning_rate)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[1000, 2000, 3000], gamma=0.5)

    global_step = [0]
    for epoch in range(config.tot_epoch):
        train_loader.sampler.set_epoch(epoch)
        global_step[0] = train_one_epoch(
            net, train_loader, optimizer, epoch, config, logger, writer, node_rank, global_step)
        lr_scheduler.step()

        if (epoch + 1) % config.save_model_freq == 0:
            if node_rank == 0:
                save_model(
                    net, save_path=f"{config.models}/{config.filename}_EP{epoch + 1}.model")
                mean_psnr, mean_ms_ssim, _, mean_cbr = test_epoch(
                    net, test_loader, config, logger, writer, epoch, node_rank, opts)
                logger.info(
                    f'Epoch {epoch + 1}: PSNR: {mean_psnr:.4f}, MS-SSIM: {mean_ms_ssim:.4f}, CBR: {mean_cbr:.4f}')
            else:
                mean_psnr, mean_ms_ssim, _, mean_cbr = test_epoch(
                    net, test_loader, config, logger, writer, epoch, node_rank, opts)

        dist.barrier()
        torch.cuda.empty_cache()

    if node_rank == 0:
        writer.close()


def distributed_init(opts):
    ddp_url = getattr(opts, "ddp.dist_url",
                      f"tcp://localhost:{opts.dist_port}")
    node_rank = getattr(opts, "ddp.rank", 0)
    world_size = torch.cuda.device_count()

    dist.init_process_group(
        backend="nccl", init_method=ddp_url, world_size=world_size, rank=node_rank)
    dist.all_reduce(torch.zeros(1).cuda())
    setattr(opts, "ddp.rank", node_rank)
    return node_rank


def distributed_worker(i, main, opts):
    setattr(opts, "dev.device_id", i)
    torch.cuda.set_device(i)
    setattr(opts, "dev.device", torch.device(f"cuda:{i}"))
    setattr(opts, "ddp.rank", i)
    distributed_init(opts)
    main(opts)


if __name__ == "__main__":
    opts = parse_args()
    seed_torch(seed=opts.seed)
    torch.backends.cudnn.benchmark = True
    num_gpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        fn=distributed_worker, args=(main, opts), nprocs=num_gpus)
