
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import List, Tuple


def load_embeddings(embed_path: str, subset: str) -> Tuple[np.ndarray, List[dict]]:
    """Load query and target embedding vectors and index information"""
    encode_qry_path = os.path.join(embed_path, f"{subset}_qry")
    encode_tgt_path = os.path.join(embed_path, f"{subset}_tgt")

    with open(encode_qry_path, 'rb') as f:
        qry_tensor, qry_index = pickle.load(f)
    with open(encode_tgt_path, 'rb') as f:
        tgt_tensor, tgt_index = pickle.load(f)

    return qry_tensor, qry_index, tgt_tensor, tgt_index