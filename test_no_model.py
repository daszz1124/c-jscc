import argparse
import os
import pickle
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from torch.utils.data import ConcatDataset

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def argument_parser():
    """Argument parser for command line arguments."""
    parser = argparse.ArgumentParser(
        description="Robust Embedding Compressor Training and Testing"
    )
    parser.add_argument('--dataset_name', type=str, default='VisDial',
                        help='Dataset name folder under base_dir',
                        choices=['CIRR', 'VisDial','NIGHTS','VisualNews_t2i','WebQA'])
    parser.add_argument('--split', type=str, default='original',
                        choices=['original', 'diverse_instruction', 'test'],
                        help='Dataset split to load')
    parser.add_argument('--latent_dim', type=int, default=192,
                        help='Latent bottleneck dimension for the compressor model')
    parser.add_argument('--save_dir', type=str, default="vector_reconstruction",
                        help='Path to save the trained model')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Epochs for training')
    parser.add_argument('--batch_size', type=int,
                        default=1024, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--noise_std', type=float, default=None,
                        help='Stddev of Gaussian noise added during training')
    parser.add_argument('--is_train', type=bool, default=False,
                        help='True for training, False for evaluation')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lambda_nce', type=float, default=1.0)
    parser.add_argument('--scheduler', type=str, default="val_acc",
                        choices=["infonce_loss", "step","val_acc"],)
    return parser.parse_args()


class MMEBEmbeddingOnlyDataset(Dataset):
    def __init__(self, base_dir, dataset_name, split='original', sample_size=None):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.split = split
        self.sample_size = sample_size

        self.file_map = {
            'original': 'original-00000-of-00001.parquet',
            'diverse_instruction': 'diverse_instruction-00000-of-00001.parquet',
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

        if self.sample_size is not None:
            indices = self.sample_indices(len(self.df), sample_size)
            self.df = self.df.iloc[indices].reset_index(drop=True)

        pkl_base_dir = "mmeb_traindatasets_cp" if split != 'test' else "mmeb_testdatasets_cp"

        self.key_embedding_map = self.load_embedding_dict_pkl(
            os.path.join("datasets", pkl_base_dir, dataset_name,
                         f"{dataset_name}_tgt.pkl")
        )
        self.query_embedding_map = self.load_embedding_dict_pkl(
            os.path.join("datasets", pkl_base_dir, dataset_name,
                         f"{dataset_name}_qry.pkl")
        )

    def sample_indices(self, total_size: int, sample_size: int):
        if total_size <= sample_size:
            return list(range(total_size))
        step = (total_size - 1) / (sample_size - 1)
        indices = [int(round(i * step)) for i in range(sample_size)]
        indices[-1] = min(indices[-1], total_size - 1)
        return indices

    def load_embedding_dict_pkl(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.split == 'test':
            query_text = row["qry_text"]
            query_image_path = row["qry_img_path"]
            pos_text = row["tgt_text"]
            tgt_image_path = row["tgt_img_path"]
        else:
            query_text = row["qry"]
            query_image_path = row["qry_image_path"]
            pos_text = row["pos_text"]
            tgt_image_path = row["pos_image_path"]

        query_vector = self.query_embedding_map[(query_text, query_image_path)]
        query_vector = torch.from_numpy(query_vector).float()

        if self.split != 'test':
            key_vector = self.key_embedding_map[(pos_text, tgt_image_path)]
            key_vector = torch.from_numpy(key_vector).float()
            return query_vector, key_vector

        else:
            target_vectors = []
            for text, img_path in zip(pos_text, tgt_image_path):
                target_vector = self.key_embedding_map[(text, img_path)]
                target_vector = torch.from_numpy(target_vector).float()
                target_vectors.append(target_vector)
            target_tensor = torch.stack(target_vectors)
            return query_vector, target_tensor


class RobustReconstructionNet(nn.Module):
    def __init__(self, embed_dim=1536, bottleneck_dim=256, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std

        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 384),
            nn.ReLU(),
            nn.Linear(384, 768),
            nn.ReLU(),
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim)
           
        )

    def forward(self, x):

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        z = self.encoder(x)

        if self.training and self.noise_std > 0:
            noise_z = torch.randn_like(z) * self.noise_std
            z = z + noise_z

        x_recon = self.decoder(z)
        return x_recon

class RobustLatentNet(nn.Module):
    def __init__(self, embed_dim=1536, bottleneck_dim=256, noise_std=0.1):
        super().__init__()
        self.snr_db = noise_std

        dims = [1536,1024, 768, 512, 384, 256, 192, 128, 96, 64, 32,16,8]

        if bottleneck_dim not in dims:
            raise ValueError(f"Unsupported bottleneck_dim: {bottleneck_dim}. Must be one of {dims}")

        start_idx = dims.index(embed_dim)
        end_idx = dims.index(bottleneck_dim)
        layer_dims = dims[start_idx:end_idx + 1] 

        layers = []
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers, nn.BatchNorm1d(bottleneck_dim))
        
    def add_awgn(self, x):
        if self.snr_db is None:
            return x  
        signal_power = x.pow(2).mean()
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = noise_power.sqrt()
        noise = torch.randn_like(x) * noise_std
        return x + noise

    def forward(self, x,is_qry = True):
        if self.training and (self.snr_db is not None and is_qry is not None):
            x = self.add_awgn(x)

        # z = self.encoder(x)
        # return z
        return x

class ResidualBlock(nn.Module):
    """残差块，用于扩展模型同时同时不丢失信息"""

    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class RobustReconstructionNet1(nn.Module):
    def __init__(self, embed_dim=1536, bottleneck_dim=None, noise_std=0.01, num_blocks=6, dropout_rate=0.1):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else embed_dim
        self.noise_std = noise_std

        self.initial_proj = nn.Sequential(
            nn.Linear(embed_dim, self.bottleneck_dim),
            nn.BatchNorm1d(self.bottleneck_dim),
            nn.SiLU(),
        )

        self.encoder_blocks = nn.Sequential(
            *[ResidualBlock(self.bottleneck_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.BatchNorm1d(self.bottleneck_dim),
        )

        self.decoder_blocks = nn.Sequential(
            *[ResidualBlock(self.bottleneck_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.final_proj = nn.Linear(self.bottleneck_dim, embed_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """正交初始化帮助保留更多信息"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        x = self.initial_proj(x)
        z = self.encoder_blocks(x)
        z = self.bottleneck(z)

        x_recon = self.decoder_blocks(z)
        x_recon = self.final_proj(x_recon)

        return x_recon


def combined_recon_loss(original, reconstructed, alpha=0.5):
    mse_loss = F.mse_loss(reconstructed, original)
    original_norm = F.normalize(original, dim=-1)
    recon_norm = F.normalize(reconstructed, dim=-1)
    cos_sim = F.cosine_similarity(original_norm, recon_norm, dim=-1).mean()
    cos_loss = 1 - cos_sim
    return alpha * mse_loss + (1 - alpha) * cos_loss


def info_nce_loss(q_recon, target, temperature=0.07):
    q_recon = F.normalize(q_recon.squeeze(1), dim=-1)  # [B, embed_dim]
    target = F.normalize(target.squeeze(1), dim=-1)    # [B, embed_dim]

    logits = torch.matmul(q_recon, target.T)  # [B, B]
    logits = logits / temperature
    labels = torch.arange(q_recon.size(0), device=q_recon.device)

    loss = F.cross_entropy(logits, labels)
    return loss


def train_epoch(model, dataloader, optimizer, device, args):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_nce = 0.0

    for query_vecs, target_vecs in dataloader:
        query_vecs = query_vecs.to(device)
        target_vecs = target_vecs.to(device)

        optimizer.zero_grad()
        q_recon = model(query_vecs)
        t_recon = model(target_vecs,is_qry=False)
        
        # loss_recon = args.beta * \
        #     combined_recon_loss(q_recon, query_vecs, args.alpha)
            
        loss_nce = args.lambda_nce * info_nce_loss(q_recon, t_recon)
        loss = loss_nce
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * query_vecs.size(0)
        total_nce += loss_nce.item() * query_vecs.size(0)

    size = len(dataloader.dataset)
    return total_loss / size, total_recon / size, total_nce / size


def get_pred(reconstructed_query: torch.Tensor, target_vectors: torch.Tensor, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cosine similarity scores between reconstructed query vector and multiple target vectors.
    """
    if normalize:
        reconstructed_query = F.normalize(
            reconstructed_query, dim=-1)  # [B, 1, D]
        target_vectors = F.normalize(
            target_vectors, dim=-1)            # [B, M, D]

    # Compute cosine similarity: [B, M]
    scores = torch.bmm(reconstructed_query,
                       target_vectors.transpose(1, 2)).squeeze(1)  # [B, M]
    predictions = torch.argmax(scores, dim=1)  # [B]

    return scores, predictions


@torch.no_grad()
def test_epoch(model, dataloader, device, args, normalization=True):
    model.eval()
    correct = 0
    total = 0

    real_correct = 0

    for batch in dataloader:
        qry_vec = batch[0].to(device)
        tgt_vec = batch[1].to(device)

        recon_qry = model(qry_vec)
        
        recon_tgts = []
        for tgt in tgt_vec:
            recon_tgt = model(tgt,is_qry=False)
            recon_tgts.append(recon_tgt)
            
        recon_tgts = torch.stack(recon_tgts, dim=0) 
            
        B, M, D = tgt_vec.shape
        
        recon_qry = recon_qry.unsqueeze(1)
        qry_vec = qry_vec.unsqueeze(1)

        _, preds = get_pred(recon_qry, recon_tgts)
        correct += torch.sum(preds == 0).item()  

        _, real_preds = get_pred(qry_vec, tgt_vec)
        real_correct += torch.sum(real_preds == 0).item()  # grounding accuracy

        total += B

    acc = correct / total
    real_acc = real_correct / total

    size = len(dataloader.dataset)
    return acc, real_acc


def add_awgn(snr_db,x):
    signal_power = x.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = noise_power.sqrt()
    noise = torch.randn_like(x) * noise_std
    return x + noise

def test_no_model(dataloader, device, args, normalization=True):
    correct = 0
    total = 0

    real_correct = 0

    for batch in dataloader:
        qry_vec = batch[0].to(device)
        tgt_vec = batch[1].to(device)
        recon_qry = add_awgn(args.noise_std, qry_vec)
        
        recon_tgts = []
        for tgt in tgt_vec:
            recon_tgts.append(tgt)
            
        recon_tgts = torch.stack(recon_tgts, dim=0) 
            
        B, M, D = tgt_vec.shape
        
        recon_qry = recon_qry.unsqueeze(1)
        qry_vec = qry_vec.unsqueeze(1)

        _, preds = get_pred(recon_qry, recon_tgts)
        correct += torch.sum(preds == 0).item()  

        _, real_preds = get_pred(qry_vec, tgt_vec)
        real_correct += torch.sum(real_preds == 0).item()  # grounding accuracy

        total += B

    acc = correct / total
    real_acc = real_correct / total

    size = len(dataloader.dataset)
    return acc, real_acc


def main():
    args = argument_parser()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    train_base_dir = "/home/iisc/zsd/project/VG2SC/MMEB-Datasets/MMEB-Datasets/MMEB-train"
    test_base_dir = "/home/iisc/zsd/project/VG2SC/MMEB-Datasets/MMEB-Datasets/MMEB-eval"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(args.save_dir, f"run_{args.dataset_name}_latent{args.latent_dim}_epoch{args.epoch}_split_{args.split}_noise_std{args.noise_std}_bszie{args.batch_size}_lambda_nce{args.lambda_nce}_alpha{args.alpha}_beta{args.beta}")
    work_dir = os.path.join(work_dir, timestamp)
    os.makedirs(work_dir, exist_ok=True)
    tensorboard_dir = os.path.join(work_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_dir)
    model_save_path = os.path.join(work_dir, "model")
    os.makedirs(model_save_path,exist_ok=True)

    print(
        f"Loading dataset {args.dataset_name} split={args.split} from base_dir={train_base_dir}")
    train_dataset = MMEBEmbeddingOnlyDataset(
        base_dir=train_base_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        sample_size=None
    )

    test_datasets = MMEBEmbeddingOnlyDataset(
        base_dir=test_base_dir,
        dataset_name=args.dataset_name,
        split='test',
        sample_size=None
    )


    testdataloader = DataLoader(
        test_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    recon_acc, real_acc = test_no_model(testdataloader, device, args)
    print(
        f"Recon Acc: {recon_acc}  | "
        f"Real Acc: {real_acc} | "
        )

if __name__ == '__main__':
    main()
