"""
Team classification using SigLIP embeddings → UMAP → KMeans.

Fits on player crops sampled from the full video, then predicts
a team label (0 or 1) for any new crop.
"""

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from more_itertools import chunked

from transformers import AutoProcessor, SiglipVisionModel
import umap
from sklearn.cluster import KMeans

from config import SIGLIP_MODEL_PATH, BATCH_SIZE


class TeamClassifier:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)
        self._fitted = False

    def _embed(self, crops: list) -> np.ndarray:
        """Convert a list of numpy BGR crops to SigLIP embeddings."""
        pil_crops = [
            Image.fromarray(crop[:, :, ::-1])  # BGR → RGB
            for crop in crops
        ]
        batches = list(chunked(pil_crops, BATCH_SIZE))
        embeddings = []

        with torch.no_grad():
            for batch in tqdm(batches, desc="Embedding crops", leave=False):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                emb = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                embeddings.append(emb)

        return np.concatenate(embeddings)

    def fit(self, crops: list) -> "TeamClassifier":
        """Fit the classifier on a list of numpy BGR player crops."""
        embeddings = self._embed(crops)
        projections = self.reducer.fit_transform(embeddings)
        self.cluster_model.fit(projections)
        self._fitted = True
        return self

    def predict(self, crops: list) -> np.ndarray:
        """Predict team label (0 or 1) for each crop."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        embeddings = self._embed(crops)
        projections = self.reducer.transform(embeddings)
        return self.cluster_model.predict(projections)

    def fit_predict(self, crops: list) -> np.ndarray:
        self.fit(crops)
        return self.predict(crops)
