import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

datasets_list = [
    'CIRR',
    'NIGHTS',
    'VisDial',
    'VisualNews_t2i',
    'WebQA'
]


class MMEBDataset(Dataset):
    def __init__(self, base_dir, dataset_name, image_dir, split='train', transform=None):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.split = split
        self.image_dir = image_dir
        self.transform = transform
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
        qyr_image_path = os.path.join(self.image_dir, row["qry_image_path"])
        pos_image_path = os.path.join(self.image_dir, row["pos_image_path"])
        if row["qry_image_path"]:
            sample["qry_image"] = self.load_image(qyr_image_path)
        else:
            sample["qry_image"] = None
        sample["pos_image"] = self.load_image(pos_image_path)
        return sample


def save_dataset_images(base_dir, dataset_name, image_dir, split='train', save_root='./saved_images', num_samples=100):
    dataset = MMEBDataset(base_dir, dataset_name, image_dir, split)
    save_path = os.path.join(save_root, f"{dataset_name}_{split}")
    os.makedirs(save_path, exist_ok=True)
    count = min(num_samples, len(dataset))
    for idx in range(count):
        sample = dataset[idx]
        qry_fname = os.path.join(save_path, f"{idx:04d}_qry.jpg")
        pos_fname = os.path.join(save_path, f"{idx:04d}_pos.jpg")
        if sample["qry_image"] is not None:
            sample["qry_image"].save(qry_fname)
        else:
            print(f"Warning: qry_image for index {idx} is None, skipping save.")
        sample["pos_image"].save(pos_fname)

        if (idx + 1) % 10 == 0 or idx == count - 1:
            print(f"Saved {idx + 1}/{count} pairs")

    print(f"✅ All images saved to {save_path}")


save_dataset_images(
    base_dir='/home/iisc/zsd/project/VG2SC/MMEB-Datasets/MMEB-Datasets/MMEB-train',
    dataset_name='VisDial',
    image_dir='/home/iisc/zsd/project/VG2SC/MMEB-Datasets/trainning_images',
    split='train',
    save_root='./saved_images',
    num_samples=100
)


for dataset_name in datasets_list:
    print(f"Dataset: {dataset_name}")
    dataset = MMEBDataset(
        base_dir='/home/iisc/zsd/project/VG2SC/MMEB-Datasets/MMEB-Datasets/MMEB-train',
        dataset_name=dataset_name,
        image_dir='/home/iisc/zsd/project/VG2SC/MMEB-Datasets/trainning_images',
        split='train'
    )
    print(f"Number of samples: {len(dataset)}")
    print(f"First sample: {dataset[0]}")
    print("-" * 40)
