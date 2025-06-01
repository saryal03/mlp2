#!/Users/sa/Documents/SCU/2025/ML/mlproject2/.venv/bin/python

# src/eval.py

import os
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

def compute_recall_at_k_numpy(gallery_emb, gallery_labels, query_emb, query_labels, ks=[1,5,10]):
    """
    Compute Recall@k by brute‐force cosine similarity in numpy.
    gallery_emb: (Ng, D) numpy array, L2‐normalized
    gallery_labels: (Ng,) numpy array of ints
    query_emb: (Nq, D) numpy array, L2‐normalized
    query_labels: (Nq,) numpy array of ints
    Returns: dict {k: recall@k}
    """
    # Compute full similarity matrix: (Nq, Ng)
    sim_matrix = query_emb.dot(gallery_emb.T)  # cosine similarity since normalized
    Nq = query_emb.shape[0]
    recalls = {}
    for k in ks:
        correct = 0
        # For each query, get top‐k indices by similarity
        # np.argpartition for efficiency, then sort the top‐k slice
        topk_indices = np.argpartition(-sim_matrix, k-1, axis=1)[:, :k]
        for i in range(Nq):
            retrieved = gallery_labels[topk_indices[i]]
            if query_labels[i] in retrieved:
                correct += 1
        recalls[k] = correct / Nq
    return recalls

def compute_map_numpy(gallery_emb, gallery_labels, query_emb, query_labels):
    """
    Compute mean Average Precision (mAP) with numpy, showing a tqdm progress bar.
    """
    # Precompute full similarity matrix
    sim_matrix = query_emb.dot(gallery_emb.T)  # (Nq, Ng)
    Nq, Ng = sim_matrix.shape
    APs = []
    for i in tqdm(range(Nq), desc="Computing mAP", unit="query"):
        true_label = query_labels[i]
        # Sort gallery by descending similarity
        ranking = np.argsort(-sim_matrix[i])  # length Ng
        retrieved_labels = gallery_labels[ranking]
        # Create hits array
        hits = (retrieved_labels == true_label).astype(int)
        if hits.sum() == 0:
            APs.append(0.0)
            continue
        precisions = []
        num_hits = 0
        for idx, val in enumerate(hits, start=1):
            if val == 1:
                num_hits += 1
                precisions.append(num_hits / idx)
        APs.append(np.mean(precisions))
    return float(np.mean(APs))

def evaluate(cfg):
    """
    Loads val/test embeddings and labels, computes Recall@1/5/10 and mAP using numpy,
    prints results, then exits automatically.
    """
    # 1) Load embeddings and labels (torch tensors)
    val_emb = torch.load(os.path.join(cfg.embed.save_dir, "val_embeddings.pt"))
    val_lbl = torch.load(os.path.join(cfg.embed.save_dir, "val_labels.pt"))
    test_emb = torch.load(os.path.join(cfg.embed.save_dir, "test_embeddings.pt"))
    test_lbl = torch.load(os.path.join(cfg.embed.save_dir, "test_labels.pt"))

    # Convert to numpy arrays
    gallery_emb = val_emb.numpy().astype("float32")
    gallery_lbl = val_lbl.numpy()
    query_emb = test_emb.numpy().astype("float32")
    query_lbl = test_lbl.numpy()

    # Ensure embeddings are L2‐normalized (they should be already)
    gallery_emb /= np.linalg.norm(gallery_emb, axis=1, keepdims=True)
    query_emb   /= np.linalg.norm(query_emb, axis=1, keepdims=True)

    # 2) Compute Recall@K
    recalls = compute_recall_at_k_numpy(
        gallery_emb, gallery_lbl, query_emb, query_lbl, ks=[1,5,10]
    )
    print(f"Recall@1:  {recalls[1]:.4f}")
    print(f"Recall@5:  {recalls[5]:.4f}")
    print(f"Recall@10: {recalls[10]:.4f}")

    # 3) Compute mAP (with progress bar)
    mAP = compute_map_numpy(gallery_emb, gallery_lbl, query_emb, query_lbl)
    print(f"mAP:       {mAP:.4f}")

    # 4) Auto-exit
    print("Evaluation complete. Exiting.")

if __name__ == "__main__":
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "../configs/cub.yaml"))
    evaluate(cfg)