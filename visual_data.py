import glob
import pandas as pd


def load_datasets(task_name='CIRR'):
    task_path = f"/home/iisc/zsd/project/VG2SC/MMEB-Datasets/MMEB-Datasets/MMEB-train/{task_name}"
    task_path_list = glob.glob(task_path + '/*.parquet')
    all_datasets = []
    for task_path in task_path_list:
        df = pd.read_parquet(task_path)
        x = df[df["qry_image_path"] == "images/CIRR/Train/train-6381-0-img0.jpg"]
        print("task_path",task_path)


load_datasets()