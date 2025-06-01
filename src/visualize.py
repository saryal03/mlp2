#!/Users/sa/Documents/SCU/2025/ML/mlproject2/.venv/bin/python

# src/visualize.py

import os
# Allow duplicate OpenMP runtimes on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from torch.utils.data import Dataset
import faiss

class CUBTripletDataset(Dataset):
    """
    Loads a split file listing image paths and labels, returning only the anchor path & label.
    This is used to collect raw file paths for plotting.
    """
    def __init__(self, cub_root: str, split_txt: str, transform=None):
        self.cub_root = cub_root
        self.transform = transform
        self.image_paths = []
        self.labels = []

        with open(split_txt, "r") as f:
            for line in f:
                rel_path, label = line.strip().split()
                full_path = os.path.join(cub_root, rel_path)
                self.image_paths.append(full_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # We return the raw path and label; transform is unused here.
        return self.image_paths[idx], self.labels[idx]

def load_embeddings(cfg, split="val"):
    """
    Load saved embeddings and labels from disk.
    """
    emb = torch.load(os.path.join(cfg.embed.save_dir, f"{split}_embeddings.pt"))
    lbl = torch.load(os.path.join(cfg.embed.save_dir, f"{split}_labels.pt"))
    return emb.numpy().astype("float32"), lbl.numpy()

def show_retrieval(cfg, k=5, num_examples=5):
    """
    Randomly pick num_examples queries from 'test' split, retrieve top-k from 'val' gallery,
    and display query + retrieved images side by side.
    """
    # 1) Load embeddings and labels
    gallery_emb, gallery_lbl = load_embeddings(cfg, split="val")
    query_emb, query_lbl = load_embeddings(cfg, split="test")

    # 2) Build FAISS index on val (gallery) embeddings
    D = gallery_emb.shape[1]
    index = faiss.IndexFlatIP(D)
    index.add(gallery_emb)

    # 3) Load raw image paths for gallery and test
    val_dataset = CUBTripletDataset(
        cub_root=cfg.data.cub_root,
        split_txt=cfg.data.val_split,
        transform=None
    )
    test_dataset = CUBTripletDataset(
        cub_root=cfg.data.cub_root,
        split_txt=cfg.data.test_split,
        transform=None
    )
    gallery_paths = val_dataset.image_paths
    query_paths = test_dataset.image_paths

    # 4) Randomly sample queries and show top-k
    sample_indices = random.sample(range(len(query_emb)), num_examples)
    for qi in sample_indices:
        q_path = query_paths[qi]
        q_label = query_lbl[qi]
        q_embedding = query_emb[qi : qi+1]  # shape (1, D)
        _, I = index.search(q_embedding, k)

        # Plot query + ranked results
        plt.figure(figsize=(3*(k+1), 3))
        query_img = Image.open(q_path).convert("RGB").resize((224, 224))
        plt.subplot(1, k+1, 1)
        plt.imshow(query_img)
        plt.title(f"Query ({q_label})")
        plt.axis("off")

        for j in range(k):
            g_idx = I[0, j]
            g_path = gallery_paths[g_idx]
            g_lbl = gallery_lbl[g_idx]
            g_img = Image.open(g_path).convert("RGB").resize((224, 224))
            plt.subplot(1, k+1, j+2)
            plt.imshow(g_img)
            plt.title(f"Rank {j+1}\nLabel {g_lbl}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

def main():
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "../configs/cub.yaml"))
    show_retrieval(cfg, k=5, num_examples=5)

if __name__ == "__main__":
    main()