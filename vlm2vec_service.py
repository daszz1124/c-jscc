import os
import pandas as pd
import pickle
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F

from VLM2Vec.src.model import MMEBModel
from VLM2Vec.src.arguments import ModelArguments
from VLM2Vec.src.model_utils import load_processor, QWEN2_VL, vlm_image_tokens


class GenerateImageIntentEvaluator:
    def __init__(self, model_args=None):
        if model_args is None:
            self.model_args = ModelArguments(
                model_name='/mnt/d/zsd/project/model/Qwen2-VL-2B-Instruct',
                checkpoint_path='/mnt/d/zsd/project/model/VLM2Vec-Qwen2VL-2B',
                pooling='last',
                normalize=True,
                model_backbone='qwen2_vl',
                lora=True
            )
        else:
            self.model_args = model_args

        self.processor = load_processor(self.model_args)
        self.model = MMEBModel.load(self.model_args)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device, dtype=torch.bfloat16)
        self.model.eval()
        self.image_token = vlm_image_tokens[QWEN2_VL]

    def _process_image(self, qry_text, qry_image):
        """处理单张图像并获取特征表示"""
        if len(qry_image) == 0 or qry_image is None or qry_image[0] == '':
            image = None
        elif isinstance(qry_image, str):
            image = Image.open(qry_image).convert('RGB')

        inputs = self.processor(
            text=f'{self.image_token}{qry_text}',
            images=image,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        if 'image_grid_thw' in inputs:
            inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)

        with torch.no_grad():
            qry_rep = self.model(qry=inputs)["qry_reps"]

        return qry_rep

    def _process_images_batch(self, qry_texts, images):
        if not isinstance(images, list):
            images = [images]
        image_reps = []
        for qry, img in zip(qry_texts, images):
            rep = self._process_image(qry, img)
            image_reps.append(rep)
        return torch.cat(image_reps, dim=0)

    def evaluate_similarity(self, qry_text, images, tgt_text, tgt_images):
        with torch.no_grad():
            if isinstance(images, list) or (isinstance(images, torch.Tensor) and images.dim() > 0):
                qry_reps = self._process_images_batch(qry_text, images)
            else:
                qry_reps = self._process_image(qry_text, images).unsqueeze(0)
            if isinstance(tgt_images, list) or (isinstance(tgt_images, torch.Tensor) and tgt_images.dim() > 0):
                tgt_reps = self._process_images_batch(tgt_text, tgt_images)
            else:
                tgt_reps = self._process_image(
                    tgt_text, tgt_images).unsqueeze(0)

            sims = self.model.compute_similarity(qry_reps, tgt_reps)
            sims = sims.float().cpu().numpy().flatten().tolist()
            avg_sim = sum(sims) / len(sims)
            return avg_sim
        
    
    def compress_query(self, qry_text, qry_image):
        with torch.no_grad():
            qry_rep = self._process_images_batch(qry_text, qry_image)
            return qry_rep
        
def get_pred(reconstructed_query: torch.Tensor, target_vectors: torch.Tensor, normalize: bool = True):
    """
    Compute cosine similarity scores between reconstructed query vector and multiple target vectors.
    """
    if normalize:
        reconstructed_query = F.normalize(
            reconstructed_query, dim=-1)  # [B, 1, D]
        target_vectors = F.normalize(
            target_vectors, dim=-1)            # [B, M, D]

    # Compute cosine similarity: [B, M]
    scores = torch.bmm(reconstructed_query,
                    target_vectors.transpose(1, 2)).squeeze(1)  # [B, M]
    predictions = torch.argmax(scores, dim=1)  # [B]

    return scores, predictions

