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


def argument_parser():
    """Argument parser for command line arguments."""
    parser = argparse.ArgumentParser(
        description="Robust Embedding Compressor Training and Testing"
    )
    parser.add_argument('--dataset_name', type=str, default='CIRR',
                        help='Dataset name folder under base_dir',
                        choices=['CIRR', 'VisDial'])
    parser.add_argument('--split', type=str, default='original',
                        choices=['original', 'diverse_instruction', 'test'],
                        help='Dataset split to load')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent bottleneck dimension for the compressor model')
    parser.add_argument('--save_path', type=str, default="vector_reconstruction_save_path",
                        help='Path to save the trained model')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Epochs for training')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--noise_std', type=float, default=0.1,
                        help='Stddev of Gaussian noise added during training')
    parser.add_argument('--is_train', type=bool, default=True,
                        help='True for training, False for evaluation')
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

        pkl_base_dir = "mmeb_traindatasets" if split != 'test' else "mmeb_testdatasets"

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
            nn.Linear(embed_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, embed_dim),
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


def info_nce_loss(q_recon, target, temperature=0.07):
    q_recon = F.normalize(q_recon.squeeze(1), dim=-1)  # [B, embed_dim]
    target = F.normalize(target.squeeze(1), dim=-1)    # [B, embed_dim]

    logits = torch.matmul(q_recon, target.T)  # [B, B]
    logits = logits / temperature
    labels = torch.arange(q_recon.size(0), device=q_recon.device)

    loss = F.cross_entropy(logits, labels)
    return loss


def train_epoch(model, dataloader, optimizer, mse_criterion, device):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_nce = 0.0

    for query_vecs, target_vecs in dataloader:
        query_vecs = query_vecs.to(device)
        target_vecs = target_vecs.to(device)

        optimizer.zero_grad()
        q_recon = model(query_vecs)

        loss_recon = mse_criterion(q_recon, query_vecs)
        loss_nce = info_nce_loss(q_recon, target_vecs)
        loss = loss_recon + loss_nce

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * query_vecs.size(0)
        total_recon += loss_recon.item() * query_vecs.size(0)
        total_nce += loss_nce.item() * query_vecs.size(0)

    size = len(dataloader.dataset)
    return total_loss / size, total_recon / size, total_nce / size



import torch
import torch.nn.functional as F

def get_pred(reconstructed_query: torch.Tensor, target_vectors: torch.Tensor, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cosine similarity scores between reconstructed query vector and multiple target vectors.
    """
    if normalize:
        reconstructed_query = F.normalize(reconstructed_query, dim=-1)  # [B, 1, D]
        target_vectors = F.normalize(target_vectors, dim=-1)            # [B, M, D]

    # Compute cosine similarity: [B, M]
    scores = torch.bmm(reconstructed_query, target_vectors.transpose(1, 2)).squeeze(1)  # [B, M]
    predictions = torch.argmax(scores, dim=1)  # [B]

    return scores, predictions


@torch.no_grad()
def test_epoch(model, dataloader, device, normalization=True):
    model.eval()
    correct = 0
    total = 0
    
    real_correct = 0

    for batch in dataloader:
        qry_vec = batch[0].to(device)       # [B, 1536]
        tgt_vec = batch[1].to(device)         # [B, M, 1536]

        recon_qry = model(qry_vec)                                 # [B, 1536]

        B, M, D = tgt_vec.shape
        recon_qry = recon_qry.unsqueeze(1)                         # [B, 1, 1536]
        qry_vec = qry_vec.unsqueeze(1)                             # [B, 1, 1536]

        _, preds = get_pred(recon_qry,tgt_vec)
        correct += torch.sum(preds == 0).item()  # reconstruction accuracy
        
        _, real_preds = get_pred(qry_vec,tgt_vec)
        real_correct += torch.sum(real_preds == 0).item()  # grounding accuracy
        
        total += B

    acc = correct / total
    real_acc = real_correct / total
    return acc , real_acc


def main():
    args = argument_parser()
    device = torch.device('cpu')

    train_base_dir = "/home/iisc/zsd/project/VG2SC/MMEB-Datasets/MMEB-Datasets/MMEB-train"
    test_base_dir = "/home/iisc/zsd/project/VG2SC/MMEB-Datasets/MMEB-Datasets/MMEB-eval"

    print(f"Loading dataset {args.dataset_name} split={args.split} from base_dir={train_base_dir}")

    train_dataset = MMEBEmbeddingOnlyDataset(
        base_dir=train_base_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        sample_size=None if args.split == 'test' else 20000
    )

    test_datasets = MMEBEmbeddingOnlyDataset(
        base_dir=test_base_dir,
        dataset_name=args.dataset_name,
        split='test',
        sample_size=None
    )

    traindataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    testdataloader = DataLoader(
        test_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    model = RobustReconstructionNet(
        embed_dim=1536,
        bottleneck_dim=args.latent_dim, 
        noise_std=args.noise_std
    )
    
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    mse_criterion = nn.MSELoss()

    if args.is_train:
        print(f"Start training for {args.epoch} epochs")
        for epoch in tqdm(range(args.epoch)):
            loss, recon_loss, nce_loss = train_epoch(
                model,
                traindataloader,
                optimizer,
                mse_criterion,
                device
            )
            
            recon_acc, real_acc = test_epoch(model, testdataloader, device=device)
            
            print(f"Epoch {epoch+1}/{args.epoch} | Total Loss: {loss:.4f} | Recon Loss: {recon_loss:.4f} | InfoNCE Loss: {nce_loss:.4f} | Recon Acc: {recon_acc}  | Real Acc: {real_acc}")
            
        os.makedirs(args.save_path, exist_ok=True)
        save_file = os.path.join(args.save_path, f"model_{args.dataset_name}_latent{args.latent_dim}_epoch{args.epoch}.pt")
        torch.save(model.state_dict(), save_file)
        print(f"Model saved to {save_file}")
    else:
        print("Evaluation mode not implemented yet")


if __name__ == '__main__':
    main()
