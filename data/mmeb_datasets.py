from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import pandas as pd

from .utils import load_embeddings


class MMEBTrainDataset(Dataset):
    def __init__(self, base_dir, dataset_name, image_dir, split='train', transform=None):
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
        pos_image_path = os.path.join(self.image_dir,name)
        image = Image.open(pos_image_path).convert('RGB')
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
        return len(self.df)


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
        train_dataset = MMEBTrainDataset(
            base_dir=config.train_data_dir,
            dataset_name=args.dataset_name,
            image_dir=config.image_dir,
            split='train',
            transform=None
        )
        test_dataset = MMEBTestDataset(
            base_dir=config.test_data_dir,
            dataset_name=args.dataset_name,
            image_dir=config.test_image_dir,
            split='test',
        )
        return train_dataset, test_dataset
    if args.testset == 'Kodak':
        train_dataset = MMEBTrainDataset(
            base_dir=config.train_data_dir,
            dataset_name=args.dataset_name,
            image_dir=config.image_dir,
            split='train',
            transform=None
        )
        test_dataset = TESTDatasets(config.test_data_dir)
        return train_dataset, test_dataset
    else:
        raise ValueError(f"Unsupported dataset: {args.trainset}")
