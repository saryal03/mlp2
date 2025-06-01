#!/Users/sa/Documents/SCU/2025/ML/mlproject2/.venv/bin/python

# src/text_query.py

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from omegaconf import OmegaConf

def load_image_paths(split_txt, cub_root):
    """
    Given a splits file (train/val/test), return a list of full image paths.
    """
    paths = []
    with open(split_txt, "r") as f:
        for line in f:
            rel_path, _ = line.strip().split()
            full_path = os.path.join(cub_root, rel_path)
            paths.append(full_path)
    return paths

def compute_clip_image_embeddings(model, processor, image_paths, device, batch_size=64):
    """
    Given a list of image file paths, returns a numpy array of CLIP image embeddings.
    """
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Computing CLIP image embeddings", unit="batch"):
            batch_paths = image_paths[i:i+batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs)  # (B, D)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            all_embs.append(image_features.cpu())
    return torch.cat(all_embs, dim=0).numpy()  # shape (N, D)

def embed_text(model, processor, text, device):
    """
    Encode a single text query into a CLIP text embedding (numpy, L2-normalized).
    """
    model.eval()
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt").to(device)
        text_features = model.get_text_features(**inputs)  # (1, D)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    return text_features.cpu().numpy()[0]  # shape (D,)

def main():
    # 1) Load config
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "../configs/cub.yaml"))
    cub_root = cfg.data.cub_root
    val_split = cfg.data.val_split

    # 2) Set device (MPS if available, else CPU)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # 3) Load CLIP model & processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 4) Load gallery (validation) image paths
    gallery_paths = load_image_paths(val_split, cub_root)
    if not gallery_paths:
        raise FileNotFoundError(f"No gallery images found in {val_split}")

    # 5) Compute (or load) CLIP image embeddings for gallery
    gallery_emb_path = os.path.join(cfg.embed.save_dir, "clip_val_image_embs.npy")
    if os.path.exists(gallery_emb_path):
        print("Loading precomputed CLIP image embeddings for gallery...")
        gallery_emb = np.load(gallery_emb_path)
    else:
        print("Computing CLIP image embeddings for gallery (val set)...")
        gallery_emb = compute_clip_image_embeddings(model, processor, gallery_paths, device, batch_size=cfg.embed.batch_size)
        os.makedirs(cfg.embed.save_dir, exist_ok=True)
        np.save(gallery_emb_path, gallery_emb)
        print(f"Saved gallery embeddings to {gallery_emb_path}")

    # 6) Prompt user for a text query
    query_text = input("Enter your text query (e.g., 'red bird with white head'): ").strip()
    if not query_text:
        print("No query entered. Exiting.")
        return

    # 7) Embed the text query
    print("Encoding text query...")
    text_emb = embed_text(model, processor, query_text, device)  # shape (D,)

    # 8) Compute cosine similarity between text embedding and each image embedding
    #    Since both are L2-normalized, similarity = dot product
    sims = gallery_emb.dot(text_emb)  # shape (N_gallery,)

    # 9) Retrieve top-K images
    K = 5
    topk_idxs = np.argpartition(-sims, K-1)[:K]
    topk_sorted = topk_idxs[np.argsort(-sims[topk_idxs])]  # sort these by descending sim

    # 10) Display the query and top-K retrieved images
    print(f"Top {K} results for query: \"{query_text}\"")
    plt.figure(figsize=(3*(K+1), 3))
    # Show a blank subplot with query text
    plt.subplot(1, K+1, 1)
    plt.text(0.5, 0.5, query_text, fontsize=14, ha="center", va="center")
    plt.axis("off")

    for i, idx in enumerate(topk_sorted):
        img_path = gallery_paths[idx]
        title = os.path.basename(os.path.dirname(img_path))  # class folder name as proxy label
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        plt.subplot(1, K+1, i+2)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()