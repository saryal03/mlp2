# src/embed.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from models.resnet_triplet import ResNet50TripletHead

class SingleImageDataset(Dataset):
    """
    Loads a split file listing image paths and labels, returning (image_tensor, label).
    """
    def __init__(self, cub_root: str, split_txt: str, transform=None):
        self.cub_root = cub_root
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Parse split file
        with open(split_txt, "r") as f:
            for line in f:
                rel_path, label = line.strip().split()
                full_path = os.path.join(cub_root, rel_path)
                self.image_paths.append(full_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def get_transform(image_size=224):
    """
    ImageNet‐style transform: resize, convert to tensor, normalize.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def load_model_checkpoint(cfg, device):
    """
    Find the latest checkpoint in cfg.train.checkpoint_dir and load it into the model.
    """
    ckpt_dir = cfg.train.checkpoint_dir
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    ckpt_files.sort()
    latest_ckpt = os.path.join(ckpt_dir, ckpt_files[-1])

    model = ResNet50TripletHead(
        projection_dim=cfg.model.projection_dim,
        freeze_backbone=False
    ).to(device)
    state = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model

def dump_embeddings_for_split(cfg, split: str):
    """
    Given 'val' or 'test', computes embeddings and labels and saves them.
    """
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = load_model_checkpoint(cfg, device)

    # Select split paths
    if split == "val":
        split_txt = cfg.data.val_split
    elif split == "test":
        split_txt = cfg.data.test_split
    else:
        raise ValueError("split must be 'val' or 'test'")

    # Build dataset & dataloader
    transform = get_transform(cfg.data.image_size)
    dataset = SingleImageDataset(
        cub_root=cfg.data.cub_root,
        split_txt=split_txt,
        transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.embed.batch_size,
        shuffle=False,
        num_workers=cfg.eval.num_workers,
        pin_memory=True
    )

    all_embeddings = []
    all_labels = []

    # Progress bar for embedding
    with torch.no_grad():
        for batch_imgs, batch_labels in tqdm(loader, desc=f"Embedding {split}", unit="batch"):
            batch_imgs = batch_imgs.to(device)
            emb = model(batch_imgs)  # (B, 128)
            all_embeddings.append(emb.cpu())
            all_labels.append(batch_labels)

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, 128)
    labels = torch.cat(all_labels, dim=0)          # (N,)

    # Save to disk
    os.makedirs(cfg.embed.save_dir, exist_ok=True)
    emb_path = os.path.join(cfg.embed.save_dir, f"{split}_embeddings.pt")
    lbl_path = os.path.join(cfg.embed.save_dir, f"{split}_labels.pt")
    torch.save(embeddings, emb_path)
    torch.save(labels, lbl_path)

    print(f"Saved {split} embeddings → {emb_path}")
    print(f"Saved {split} labels     → {lbl_path}")

def main():
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "../configs/cub.yaml"))

    # Dump val and test embeddings in sequence, with progress bars
    dump_embeddings_for_split(cfg, split="val")
    dump_embeddings_for_split(cfg, split="test")

    # Auto-exit when done
    print("Embedding extraction complete. Exiting.")

if __name__ == "__main__":
    main()