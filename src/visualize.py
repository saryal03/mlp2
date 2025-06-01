import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf

import faiss
from torchvision import transforms

from datasets.cub_triplet import CUBTripletDataset
from models.resnet_triplet import ResNet50TripletHead

def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def load_embeddings(cfg, split="val"):
    emb = torch.load(os.path.join(cfg.embed.save_dir, f"{split}_embeddings.pt"))
    lbl = torch.load(os.path.join(cfg.embed.save_dir, f"{split}_labels.pt"))
    return emb.numpy().astype("float32"), lbl.numpy()

def show_retrieval(cfg, k=5, num_examples=5):
    """
    Randomly pick num_examples queries from 'test' split, retrieve top-k from 'val' gallery,
    and display query + retrieved images.
    """
    # 1) Load embeddings and labels
    gallery_emb, gallery_lbl = load_embeddings(cfg, split="val")
    query_emb, query_lbl = load_embeddings(cfg, split="test")

    # 2) Create FAISS index
    D = gallery_emb.shape[1]
    index = faiss.IndexFlatIP(D)
    index.add(gallery_emb)

    # 3) Load raw image paths for gallery/test (to display them)
    val_dataset = CUBTripletDataset(
        cub_root=cfg.data.cub_root,
        split_txt=cfg.data.val_split,
        transform=None,  # we only need image paths
    )
    test_dataset = CUBTripletDataset(
        cub_root=cfg.data.cub_root,
        split_txt=cfg.data.test_split,
        transform=None,
    )
    # Build lists of gallery images (using the “anchor” from each triple)
    gallery_paths = [val_dataset.image_paths[i] for i in range(len(val_dataset))]
    query_paths = [test_dataset.image_paths[i] for i in range(len(test_dataset))]

    # 4) For a few random queries, retrieve top-k and plot
    sample_indices = random.sample(range(len(query_emb)), num_examples)
    for qi in sample_indices:
        q_path = query_paths[qi]
        q_label = query_lbl[qi]
        q_embedding = query_emb[qi : qi+1]  # shape (1, D)
        _, I = index.search(q_embedding, k)

        # Plot query image
        plt.figure(figsize=(3*(k+1), 3))
        query_img = Image.open(q_path).convert("RGB").resize((224,224))
        plt.subplot(1, k+1, 1)
        plt.imshow(query_img)
        plt.title(f"Query ({q_label})")
        plt.axis("off")

        # Plot top-k
        for j in range(k):
            g_idx = I[0, j]
            g_path = gallery_paths[g_idx]
            g_lbl = gallery_lbl[g_idx]
            g_img = Image.open(g_path).convert("RGB").resize((224,224))
            plt.subplot(1, k+1, j+2)
            plt.imshow(g_img)
            plt.title(f"Rank {j+1}\nLabel {g_lbl}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "../configs/cub.yaml"))
    show_retrieval(cfg, k=5, num_examples=5)