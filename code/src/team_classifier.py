"""
Team classification using SigLIP embeddings → UMAP → KMeans.

Fits on player crops sampled from the full video, then predicts
a team label (0 or 1) for any new crop.
"""

import cv2
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
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(self.device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3, random_state=42)
        self.cluster_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        self._fitted = False
        self._swap = False

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

    def _orange_score(self, crop: np.ndarray) -> float:
        """Return fraction of pixels in the orange jersey hue range (BGR input)."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (5, 100, 100), (25, 255, 255))
        return mask.mean()

    def fit(self, crops: list) -> "TeamClassifier":
        """Fit the classifier on a list of numpy BGR player crops."""
        embeddings = self._embed(crops)
        projections = self.reducer.fit_transform(embeddings)
        labels = self.cluster_model.fit_predict(projections)
        self._fitted = True

        # ensure cluster 0 = Macalester (orange)
        scores = {0: 0.0, 1: 0.0}
        for crop, label in zip(crops, labels):
            scores[label] += self._orange_score(crop)
        self._swap = scores[1] > scores[0]
        print(f"Orange scores per cluster: {scores} | swap={self._swap}")

        unique, counts = np.unique(labels, return_counts=True)
        print(f"Team clusters: {dict(zip(unique.tolist(), counts.tolist()))}")
        return self

    def predict(self, crops: list) -> np.ndarray:
        """Predict team label (0 or 1) for each crop."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        embeddings = self._embed(crops)
        projections = self.reducer.transform(embeddings)
        labels = self.cluster_model.predict(projections)
        return 1 - labels if self._swap else labels

    def fit_predict(self, crops: list) -> np.ndarray:
        self.fit(crops)
        return self.predict(crops)
