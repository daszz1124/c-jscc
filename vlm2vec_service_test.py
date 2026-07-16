

import os
import pandas as pd
import pickle
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from vlm2vec_service import GenerateImageIntentEvaluator

import argparse


class MMEBEmbeddingForAccuracy(Dataset):
    def __init__(self, base_dir, image_dir,dataset_name, split='test'):
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

        pkl_base_dir = "mmeb_traindatasets" if split != 'test' else "mmeb_testdatasets"

        self.key_embedding_map = self.load_embedding_dict_pkl(
            os.path.join("datasets", pkl_base_dir, dataset_name,
                         f"{dataset_name}_tgt.pkl")
        )
        self.query_embedding_map = self.load_embedding_dict_pkl(
            os.path.join("datasets", pkl_base_dir, dataset_name,
                         f"{dataset_name}_qry.pkl")
        )

    def load_embedding_dict_pkl(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        query_text = row["qry_text"]
        query_image_path = row["qry_img_path"]
        pos_text = row["tgt_text"]
        tgt_image_path = row["tgt_img_path"]

        qry_image_path = os.path.join(self.image_dir, query_image_path) 

        query_vector = self.query_embedding_map[(query_text, query_image_path)]
        query_vector = torch.from_numpy(query_vector).float()

        target_vectors = []
        for text, img_path in zip(pos_text, tgt_image_path):
            target_vector = self.key_embedding_map[(text, img_path)]
            target_vector = torch.from_numpy(target_vector).float()
            target_vectors.append(target_vector)
        target_tensor = torch.stack(target_vectors)

        image_name = os.path.basename(qry_image_path)
        image = Image.open(qry_image_path).convert('RGB')
        
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        img = self.transform(image)
        return img,image_name,query_vector, target_tensor, query_text,qry_image_path

class MMEBConditionTestDataset(Dataset):
    def __init__(self, base_dir, dataset_name, image_dir, split='test', transform=None):
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
        self.embedding_map = self.load_embedding_dict_pkl(
            f"datasets/mmeb_testdatasets/{self.dataset_name}/{self.dataset_name}_tgt.pkl")

    def load_embedding_dict_pkl(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        qry_text = row["qry_text"]
        qry_image_path = row["qry_img_path"]

        tgt_text = row["tgt_text"][0]
        tgt_image_path = row["tgt_img_path"][0]

        embedding_vector = self.embedding_map[(tgt_text, tgt_image_path)]
        qry_image_path = os.path.join(self.image_dir, qry_image_path) if qry_image_path != "" else ""
        tgt_image_path = os.path.join(self.image_dir, tgt_image_path) 
        image_name = os.path.basename(tgt_image_path)

        image = Image.open(tgt_image_path).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        img = self.transform(image)
        embedding_vector = torch.from_numpy(
            embedding_vector).unsqueeze(0).float()

        return img, embedding_vector, image_name, qry_text, qry_image_path, tgt_text, tgt_image_path

    def __len__(self):
        return len(self.df)


def argument_parser():
    """Argument parser for command line arguments."""
    parser = argparse.ArgumentParser(
        description="Robust Embedding Compressor Training and Testing"
    )
    parser.add_argument('--dataset_name', type=str, default='VisDial',
                        help='Dataset name folder under base_dir',
                        choices=['CIRR', 'VisDial', 'NIGHTS', 'VisualNews_t2i', 'WebQA'])
    parser.add_argument('--split', type=str, default='original',
                        choices=['original', 'diverse_instruction', 'test'],
                        help='Dataset split to load')
    return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()

    evaluator = GenerateImageIntentEvaluator()
    test_data_dir = "/mnt/d/zsd/project/c-jscc/datasets/vlm2vec/MMEB-Test/MMEB-eval"
    test_image_dir = "/mnt/d/zsd/project/c-jscc/datasets/vlm2vec/MMEB-Test/eval_image"

    test_dataset = MMEBConditionTestDataset(
        base_dir=test_data_dir,
        image_dir=test_image_dir,
        dataset_name=args.dataset_name,
    )

    test_dataset = MMEBEmbeddingForAccuracy

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True
    )

    similarities = []

    for idx, batch in enumerate(test_loader):
        qry_text, qry_image_path, tgt_text, tgt_image_path = batch[3], batch[4], batch[5], batch[6]
        avg_sim = evaluator.evaluate_similarity(
            qry_text, qry_image_path, tgt_text, tgt_image_path)

        log = (' | '.join([
            f'Index {idx}',
            f'Query Text: {batch[3]}',
            f'Target Text: {batch[5]}',
            f'avg_similarity: {avg_sim}'
        ]))

        similarities.append({
            'Index': idx,
            'Query Text': batch[3],
            'Target Text': batch[5],
            'avg_similarity': avg_sim
        })

        print(log)

    avg_sim = sum(item['avg_similarity'] for item in similarities) / \
        len(similarities) if similarities else 0.0

    similarities.append({
        'Index': 'Average',
        'Query Text': '',
        'Target Text': '',
        'avg_similarity': avg_sim
    })

    df = pd.DataFrame(similarities)
    df.to_excel(f'similarity_results/{args.dataset_name}_similarity_results.xlsx', index=False)
