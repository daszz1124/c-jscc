import glob
import pandas as pd


# WebQA : text --> image
# VisDial : text --> image
# NIGHTS : image --> image
# CIRR: image --> image
# VisualNews_t2i: text --> image

def load_datasets(task_name='CIRR'):
    task_path = f"/home/iisc/zsd/project/VG2SC/MMEB-Datasets/MMEB-Datasets/MMEB-eval/{task_name}"
    task_path_list = glob.glob(task_path + '/*.parquet')
    all_datasets = []
    for task_path in task_path_list:
        all_datasets.append(pd.read_parquet(task_path))
    df = pd.concat(all_datasets, ignore_index=True)
    return df[:100]
df = load_datasets()