# src/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets.cub_triplet import CUBTripletDataset
from models.resnet_triplet import ResNet50TripletHead

def get_transform(image_size=224):
    """
    ImageNet‐style transform: resize, random horizontal flip,
    convert to tensor, normalize to ImageNet stats.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def train_triplet(cfg):
    # Use MPS if available on M3 Pro Max, otherwise fall back to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1) Prepare dataset & DataLoader
    transform = get_transform(cfg.data.image_size)
    train_dataset = CUBTripletDataset(
        cub_root=cfg.data.cub_root,
        split_txt=cfg.data.train_split,
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    print(f"Loaded {len(train_dataset)} triplet samples for training.")

    # 2) Instantiate model
    model = ResNet50TripletHead(
        projection_dim=cfg.model.projection_dim,
        freeze_backbone=cfg.train.backbone_freeze,
    ).to(device)

    # 3) Loss, optimizer, and scheduler
    triplet_loss_fn = nn.TripletMarginLoss(
        margin=cfg.train.triplet_margin, p=2, reduction="mean"
    )
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs * len(train_loader),
        eta_min=cfg.train.lr * 0.01,
    )

    # 4) Training loop with tqdm for progress
    for epoch in range(cfg.train.epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{cfg.train.epochs}",
            unit="batch",
        )
        for step, (anchor_imgs, pos_imgs, neg_imgs, _) in loop:
            anchor_imgs = anchor_imgs.to(device)
            pos_imgs = pos_imgs.to(device)
            neg_imgs = neg_imgs.to(device)

            # Forward pass → get embeddings
            emb_a = model(anchor_imgs)  # (B, 128)
            emb_p = model(pos_imgs)
            emb_n = model(neg_imgs)

            # Compute triplet loss
            loss = triplet_loss_fn(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if (step + 1) % cfg.train.log_interval == 0:
                avg_loss = running_loss / cfg.train.log_interval
                loop.set_postfix({"loss": f"{avg_loss:.4f}"})
                running_loss = 0.0

        # 5) Save checkpoint after each epoch
        checkpoint_dir = cfg.train.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "../configs/cub.yaml"))
    train_triplet(cfg)