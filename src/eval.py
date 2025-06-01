import os
import torch
import numpy as np
import faiss
from omegaconf import OmegaConf

def compute_recall_at_k(gallery_emb, gallery_labels, query_emb, query_labels, ks=[1,5,10]):
    """
    gallery_emb: numpy (Ng, D)
    gallery_labels: numpy (Ng,)
    query_emb: numpy (Nq, D)
    query_labels: numpy (Nq,)
    Return: dict {k: recall@k}
    """
    Ng, D = gallery_emb.shape
    index = faiss.IndexFlatIP(D)  # Inner product for cosine (embeddings are L2-normalized)
    index.add(gallery_emb)
    _, I = index.search(query_emb, max(ks))  # (Nq, max_k)

    recalls = {}
    for k in ks:
        correct = 0
        for i in range(query_emb.shape[0]):
            retrieved_idxs = I[i, :k]
            retrieved_labels = gallery_labels[retrieved_idxs]
            if query_labels[i] in retrieved_labels:
                correct += 1
        recalls[k] = correct / query_emb.shape[0]
    return recalls

def compute_map(gallery_emb, gallery_labels, query_emb, query_labels):
    """
    Compute mean Average Precision (mAP) over all queries.
    """
    Nq = query_emb.shape[0]
    index = faiss.IndexFlatIP(gallery_emb.shape[1])
    index.add(gallery_emb)
    D, I = index.search(query_emb, gallery_emb.shape[0])  # (Nq, Ng)

    APs = []
    for i in range(Nq):
        true_label = query_labels[i]
        retrieved_labels = gallery_labels[I[i]]  # length Ng
        hits = (retrieved_labels == true_label).astype(int)
        if hits.sum() == 0:
            APs.append(0.0)
            continue
        # Precision at each hit position
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
    Loads val/test embeddings and labels, computes Recall@1/5/10 and mAP, prints results.
    """
    # 1) Load embeddings and labels
    val_emb = torch.load(os.path.join(cfg.embed.save_dir, "val_embeddings.pt"))
    val_lbl = torch.load(os.path.join(cfg.embed.save_dir, "val_labels.pt"))
    test_emb = torch.load(os.path.join(cfg.embed.save_dir, "test_embeddings.pt"))
    test_lbl = torch.load(os.path.join(cfg.embed.save_dir, "test_labels.pt"))

    # Convert to numpy
    val_emb_np = val_emb.numpy().astype("float32")
    val_lbl_np = val_lbl.numpy()
    test_emb_np = test_emb.numpy().astype("float32")
    test_lbl_np = test_lbl.numpy()

    # Compute gallery & query. Use val as gallery, test as query (or vice versa)
    gallery_emb, gallery_lbl = val_emb_np, val_lbl_np
    query_emb, query_lbl = test_emb_np, test_lbl_np

    # 2) Recall@k
    recalls = compute_recall_at_k(gallery_emb, gallery_lbl, query_emb, query_lbl, ks=[1,5,10])
    print("Recall@1:", recalls[1])
    print("Recall@5:", recalls[5])
    print("Recall@10:", recalls[10])

    # 3) mAP
    mAP = compute_map(gallery_emb, gallery_lbl, query_emb, query_lbl)
    print("mAP:", mAP)

if __name__ == "__main__":
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "../configs/cub.yaml"))
    evaluate(cfg)