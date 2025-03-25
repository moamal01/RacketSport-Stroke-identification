import numpy as np
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def _load_dataset(self, split="train"):
        file_path = f"data/splits/oversampled/{split}.npz"
        data = np.load(file_path)
        return data["X"], data["y"]

    def __init__(self, split):
        self.X, self.y = self._load_dataset(split)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.y[idx]