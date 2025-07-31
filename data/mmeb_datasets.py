from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import pandas as pd

from .utils import load_embeddings
import pickle
import json
import torch


class MMEBTrainDataset(Dataset):
    def __init__(self, base_dir, dataset_name, image_dir, split='original', transform=None):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.split = split
        self.image_dir = image_dir
        self.transform = self._transforms() if transform is None else transform
        self.file_map = {
            'train': 'train-00000-of-00001.parquet',
            'original': 'original-00000-of-00001.parquet',
            'diverse_instruction': 'diverse_instruction-00000-of-00001.parquet'
        }

        if split not in self.file_map:
            raise ValueError(f"Unsupported split: {split}")

        self.data_dir = os.path.join(base_dir, dataset_name)
        parquet_file = self.file_map[split]
        parquet_path = os.path.join(self.data_dir, parquet_file)
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        self.df = pd.read_parquet(parquet_path)

    def __len__(self):
        return len(self.df)

    def _transforms(self,):
        transforms_list = [
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor()
        ]

        return transforms.Compose(transforms_list)

    def load_image(self, image_path):
        full_path = os.path.join(self.image_dir, image_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        img = Image.open(full_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = row.to_dict()
        pos_image_path = os.path.join(self.image_dir, row["pos_image_path"])
        sample["pos_image"] = self.load_image(pos_image_path)
        return sample["pos_image"]


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
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        img = self.transform(image)
        return img, image_name

    def __len__(self):
        return len(self.df)


class MMEBConditionTrainDataset(Dataset):
    def __init__(self, base_dir, dataset_name, image_dir, split='original', transform=None, sample_size=20000):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.split = split
        self.image_dir = image_dir
        self.sample_size = sample_size
        self.transform = self._transforms() if transform is None else transform
        self.file_map = {
            'original': 'original-00000-of-00001.parquet',
            'diverse_instruction': 'diverse_instruction-00000-of-00001.parquet'
        }

        if split not in self.file_map:
            raise ValueError(f"Unsupported split: {split}")

        self.data_dir = os.path.join(base_dir, dataset_name)
        parquet_file = self.file_map[split]
        parquet_path = os.path.join(self.data_dir, parquet_file)
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self.df = pd.read_parquet(parquet_path)
        self.df = self.filter_valid_images(self.df, 'pos_image_path')

        if self.sample_size is not None:
            indices = self.sample_indices(len(self.df), sample_size)
            self.df = self.df.iloc[indices].reset_index(drop=True)

        self.embedding_map = self.load_embedding_dict_pkl(
            f"datasets/mmeb_traindatasets/{self.dataset_name}/{self.dataset_name}_tgt.pkl")

    def sample_indices(self, total_size: int, sample_size: int):
        """
        Sample indices with equal intervals.
        """
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

    def _transforms(self,):
        transforms_list = [
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor()
        ]

        return transforms.Compose(transforms_list)

    def load_image(self, image_path):
        full_path = os.path.join(self.image_dir, image_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        img = Image.open(full_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def filter_valid_images(self, df: pd.DataFrame, image_path_col: str) -> pd.DataFrame:
        """Filter out samples with images smaller than 256x256."""
        valid_indices = []
        for idx, row in df.iterrows():
            image_path = os.path.join(self.image_dir, row[image_path_col])
            if not os.path.exists(image_path):
                continue  # Skip non-existent images

            try:
                with Image.open(image_path) as img:
                    # PIL size returns (width, height)
                    width, height = img.size
                    # Check if the smaller side is >= 256 (to ensure 256x256 crops)
                    if min(width, height) >= 256:
                        valid_indices.append(idx)
            except Exception:
                continue  # Skip invalid images
        return df.loc[valid_indices].reset_index(drop=True)  # Reset index

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pos_text = row["pos_text"]
        tgt_image_path = row["pos_image_path"]

        image_name = os.path.basename(tgt_image_path)
        embedding_vector = self.embedding_map[(pos_text, tgt_image_path)]
        post_image = self.load_image(tgt_image_path)

        embedding_vector = torch.from_numpy(
            embedding_vector).unsqueeze(0).float()

        return post_image, embedding_vector, image_name


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
        pos_text = row["tgt_text"][0]
        tgt_image_path = row["tgt_img_path"][0]

        embedding_vector = self.embedding_map[(pos_text, tgt_image_path)]
        pos_image_path = os.path.join(self.image_dir, tgt_image_path)
        image_name = os.path.basename(pos_image_path)

        image = Image.open(pos_image_path).convert('RGB')
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

        return img, embedding_vector, image_name

    def __len__(self):
        return len(self.df)


class CondiationKodak(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()

        with open("datasets/kodak/kodak_val_qwen2vl_embeddings.pickle", 'rb') as f:
            self.embedding_map = pickle.load(f)

        with open("datasets/kodak/kodak_ofa.json", 'rb') as f:
            self.captions = json.load(f)

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        name = os.path.basename(image_ori)

        caption = self.captions[name]
        embedding_vector = self.embedding_map[(caption, name)]

        image = Image.open(image_ori).convert('RGB')
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

        return img, embedding_vector, name

    def __len__(self):
        return len(self.imgs)


class TESTDatasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        name = os.path.basename(image_ori)
        image = Image.open(image_ori).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        img = self.transform(image)
        return img, name

    def __len__(self):
        return len(self.imgs)


def select_dataset_mmeb(args, config):
    if args.testset == 'MMEB':
        train_dataset = MMEBConditionTrainDataset(
            base_dir=config.train_data_dir,
            dataset_name=args.dataset_name,
            image_dir=config.image_dir,
            split='original',
            transform=None
        )

        test_dataset = MMEBConditionTestDataset(
            base_dir=config.test_data_dir,
            dataset_name=args.dataset_name,
            image_dir=config.test_image_dir,
            split='test',
        )
        return train_dataset, test_dataset
    if args.testset == 'Kodak':
        train_dataset = MMEBConditionTrainDataset(
            base_dir=config.train_data_dir,
            dataset_name=args.dataset_name,
            image_dir=config.image_dir,
            split='original',
            transform=None
        )
        test_dataset = CondiationKodak(config.test_data_dir)
        return train_dataset, test_dataset
    else:
        raise ValueError(f"Unsupported dataset: {args.trainset}")
