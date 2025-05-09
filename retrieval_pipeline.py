import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

try:
    import peft  # LoRA & PEFT utilities
except ImportError:
    peft = None

try:
    import faiss  # Facebook AI Similarity Search
except ImportError:
    faiss = None

###############################
# CONFIGURATION
###############################
class Config:
    # Model & prompts
    clip_model_name: str = "openai/clip-vit-base-patch16"
    train_prompts: List[str] = [
        "a photo of a {label}",
        "a close-up photo of a {label}",
        "a detailed image of a {label}",
    ]
    prompt_learnable_tokens: int = 8  # for CoOp‑style prompt tuning
    lora_rank: int = 8  # LoRA adapter rank
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Dataset paths (edit to match local layout)
    cub_root: Path = Path("./data/CUB_200_2011")
    cars_root: Path = Path("./data/stanford_cars")

    # Training hyper‑parameters
    batch_size: int = 64
    epochs: int = 1  # quick demo; increase for better performance
    lr: float = 1e-4

    # Retrieval parameters
    faiss_nlist: int = 100  # IVF list count
    top_k: Tuple[int, ...] = (1, 5, 10)

###############################
# DATASET
###############################
class ImageLabelDataset(torch.utils.data.Dataset):
    """Minimal image‑path/label dataset; expects list[(path,label)]."""

    def __init__(self, samples: List[Tuple[str, int]], processor: CLIPProcessor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.processor(images=path, return_tensors="pt").pixel_values.squeeze(0)
        return image, label


def build_samples_list(root: Path, split_file: str) -> List[Tuple[str, int]]:
    """Reads split info and returns a list of (image_path, class_idx)."""
    samples = []
    with open(split_file, "r") as f:
        for line in f:
            img_rel_path, cls = line.strip().split()
            samples.append((str(root / img_rel_path), int(cls)))
    return samples

###############################
# PROMPT ENGINEERING
###############################
class PromptLayer(nn.Module):
    """Learnable context tokens à la CoOp."""

    def __init__(self, hidden_dim: int, num_tokens: int, init_range: float = 0.02):
        super().__init__()
        self.context = nn.Parameter(torch.empty(num_tokens, hidden_dim).uniform_(-init_range, init_range))

    def forward(self, text_embeddings):
        # prepend learnable tokens: [ctx_1] … [ctx_n] + original embeddings (w/o SOS)
        sos = text_embeddings[:, :1, :]
        rest = text_embeddings[:, 1:, :]
        ctx = self.context.unsqueeze(0).expand(text_embeddings.size(0), -1, -1)
        return torch.cat((sos, ctx, rest), dim=1)

###############################
# MODEL SETUP
###############################

def get_clip_with_lora(cfg: Config):
    processor = CLIPProcessor.from_pretrained(cfg.clip_model_name)
    model = CLIPModel.from_pretrained(cfg.clip_model_name)

    if peft is None:
        print("⚠️ PEFT not installed; LoRA disabled.")
        return model, processor

    lora_config = peft.LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=["attn"],  # LoRA on attention projections
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=peft.TaskType.FEATURE_EXTRACTION,
    )
    model = peft.get_peft_model(model, lora_config)
    return model, processor

###############################
# TRAINING LOOP (LoRA fine‑tuning)
###############################

def train_clip_lora(model: CLIPModel, dataloader: DataLoader, cfg: Config):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(cfg.epochs):
        for images, labels in dataloader:
            images = images.to(device)
            text_inputs = [cfg.train_prompts[0].format(label="") for _ in labels]  # dummy base prompt
            tokenized = model.processor(text=text_inputs, return_tensors="pt", padding=True).to(device)

            outputs = model(pixel_values=images, **tokenized)
            logits_per_image = outputs.logits_per_image  # (batch, text_batch)
            # contrastive loss: image i matches text i
            targets = torch.arange(images.size(0), device=device)
            loss = nn.functional.cross_entropy(logits_per_image, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

###############################
# EMBEDDING & INDEXING
###############################

def extract_embeddings(model: CLIPModel, dataloader: DataLoader) -> Tuple[torch.Tensor, List[int]]:
    model.eval()
    device = next(model.parameters()).device
    all_embeds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeds = model.get_image_features(images)
            embeds = nn.functional.normalize(embeds, p=2, dim=1)
            all_embeds.append(embeds.cpu())
            all_labels.extend(labels)
    return torch.cat(all_embeds, dim=0), all_labels


def build_faiss_index(vectors: torch.Tensor, cfg: Config):
    if faiss is None:
        raise ImportError("faiss not installed")
    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, cfg.faiss_nlist, 16, 8)
    index.train(vectors.numpy())
    index.add(vectors.numpy())
    return index

###############################
# RETRIEVAL & EVALUATION
###############################

def recall_at_k(results: List[List[int]], labels: List[int], k: int) -> float:
    correct = sum(labels[i] in results[i][:k] for i in range(len(labels)))
    return correct / len(labels)


def main():
    cfg = Config()

    # 1. Model & processor
    model, processor = get_clip_with_lora(cfg)

    # 2. Data: Only CUB (train split). Replace with proper loader.
    train_samples = build_samples_list(cfg.cub_root, "splits/train.txt")
    train_ds = ImageLabelDataset(train_samples, processor)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    # 3. Optional: Fine‑tune LoRA
    if peft is not None:
        train_clip_lora(model, train_dl, cfg)
        model.save_pretrained("./checkpoints/clip_lora")

    # 4. Extract embeddings and build index
    vectors, labels = extract_embeddings(model, train_dl)
    index = build_faiss_index(vectors, cfg)

    # 5. Evaluate retrieval on train set (demo)
    _, I = index.search(vectors.numpy(), max(cfg.top_k))  # indices of nearest neighbours
    recalls = {k: recall_at_k(I.tolist(), labels, k) for k in cfg.top_k}
    print("Recall:", recalls)


if __name__ == "__main__":
    main()
