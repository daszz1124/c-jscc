import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict


def load_embeddings(embed_path: str, subset: str) -> Tuple[np.ndarray, List[dict]]:
    """Load query and target embedding vectors and index information"""
    encode_qry_path = os.path.join(f"{embed_path}", f"{subset}_qry")
    encode_tgt_path = os.path.join(f"{embed_path}", f"{subset}_tgt")

    with open(encode_qry_path, 'rb') as f:
        qry_tensor, qry_index = pickle.load(f)
    with open(encode_tgt_path, 'rb') as f:
        tgt_tensor, tgt_index = pickle.load(f)

    return qry_tensor, qry_index, tgt_tensor, tgt_index


def save_embedding_dict_pkl(path, embeddings: np.ndarray, indices: List[dict]):
    embedding_dict = {
        (item["text"], item["img_path"]): embed
        for embed, item in zip(embeddings, indices)
    }
    with open(path, 'wb') as f:
        pickle.dump(embedding_dict, f)


def change_embedding_dict_pkl(path: str) -> Dict[Tuple[str, str], np.ndarray]:
    """Load embedding dictionary and convert img_path to filename"""
    with open(path, 'rb') as f:
        original_dict = pickle.load(f)

    modified_dict = {
        (text, os.path.basename(img_path)): embedding
        for (text, img_path), embedding in original_dict.items()
    }

    with open(path, 'wb') as f:
        pickle.dump(modified_dict, f)

    print(f"Modified {path} successfully")

    return modified_dict


if __name__ == "__main__":

    datasets = ["NIGHTS", "VisDial", "VisualNews_t2i", "WebQA","CIRR","MSCOCO_t2i"]

    base_path = "datasets/diverse_instruction"

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        dataset_path = os.path.join(base_path, dataset)
        print("dataset_path", dataset_path)
        if not os.path.exists(dataset_path):
            print(f"Skipping {dataset}: Path not found")
            continue

        try:
            qry_tensor, qry_index, tgt_tensor, tgt_index = load_embeddings(
                dataset_path, dataset)

            qry_pkl_path = os.path.join(dataset_path, f"{dataset}_qry.pkl")
            tgt_pkl_path = os.path.join(dataset_path, f"{dataset}_tgt.pkl")

            save_embedding_dict_pkl(qry_pkl_path, qry_tensor, qry_index)
            save_embedding_dict_pkl(tgt_pkl_path, tgt_tensor, tgt_index)

            print(f"qry_tensor shape: {qry_tensor.shape}")
            print(f"qry_index length: {len(qry_index)}")
            print(f"tgt_tensor shape: {tgt_tensor.shape}")
            print(f"tgt_index length: {len(tgt_index)}")
            if len(tgt_index) > 0:
                print(f"tgt[0]: {tgt_index[0]}")

            print(f"Finished processing {dataset}")

        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue
