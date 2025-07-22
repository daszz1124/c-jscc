
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
    encode_qry_path = os.path.join(embed_path, f"{subset}_qry")
    encode_tgt_path = os.path.join(embed_path, f"{subset}_tgt")

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
        
    print("modified_dict", modified_dict)

    return modified_dict


if __name__ == "__main__":
    # qry_tensor, qry_index, tgt_tensor, tgt_index= load_embeddings("datasets/query_datasets/CIRR", "CIRR")
    # save_embedding_dict_pkl("datasets/query_datasets/CIRR_qry.pkl", qry_tensor, qry_index)
    # save_embedding_dict_pkl("datasets/query_datasets/CIRR_tgt.pkl", tgt_tensor, tgt_index)

    # print(qry_index[0])
    # print(qry_tensor[0])
    # print(tgt_index[0])
    # print(tgt_tensor[0])

    # embedding = change_embedding_dict_pkl(
    #     "/home/iisc/zsd/project/VG2SC/SwinJSCC/datasets/kodak/kodak_val_qwen2vl_embeddings.pickle")

    with open("/home/iisc/zsd/project/VG2SC/SwinJSCC/datasets/kodak/kodak_val_qwen2vl_embeddings.pickle", 'rb') as f:
        original_dict = pickle.load(f)
        
    print(original_dict)