import glob
import pandas as pd
import numpy as np



def load_datasets(task_name='NIGHTS',file_name = "distort_008_566_0"):
    task_path = f"/mnt/d/zsd/project/c-jscc/datasets/vlm2vec/MMEB-Test/MMEB-eval/{task_name}"
    task_path_list = glob.glob(task_path + '/*.parquet')
    all_datasets = []
    for task_path in task_path_list:
        all_datasets.append(pd.read_parquet(task_path))
    df = pd.concat(all_datasets, ignore_index=True)
    return df



target_name = "distort_004_541_1.jpg"
df = load_datasets()


def filter_list_column(path_list):
    if isinstance(path_list, np.ndarray):
        if target_name in path_list[0]:
            return True
    return False

filtered_df = df[df['tgt_img_path'].apply(filter_list_column)]

for index, text in enumerate(filtered_df["qry_text"]):
    print(f"=== 第 {index+1} 条样本 ===")
    print(text)
    print("-" * 50)

# 打印结果
print(f"筛选出 {len(filtered_df)} 条数据")
print(filtered_df["qry_text"])
print(filtered_df["qry_img_path"])
print(filtered_df["tgt_img_path"])
print(filtered_df["tgt_text"])