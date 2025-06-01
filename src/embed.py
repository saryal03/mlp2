import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import OmegaConf

from datasets.cub_triplet import CUBTripletDataset
from models.resnet_triplet import ResNet50TripletHead


def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def dump_embeddings(cfg, split: str = "val"):
    """
    Loads the trained model checkpoint, computes 128-D embeddings for
    either 'val' or 'test' split, and saves embeddings and labels to .pt files.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load config‐selected split
    split_txt = cfg.data.val_split if split == "val" else cfg.data.test_split

    # 2) Build dataset & dataloader
    transform = get_transform(cfg.data.image_size)
    ds = CUBTripletDataset(
        cub_root=cfg.data.cub_root,
        split_txt=split_txt,
        transform=transform,
    )
    # We only need anchor images (ignore positive/negative), so make a simple wrapper:
    class SingleDataset(torch.utils.data.Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = torch.load  # placeholder

    # Actually, reuse CUBTripletDataset, but only take anchor images:
    ds_single = []
    for idx in range(len(ds)):
        img, _, _, lbl = ds[idx]
        ds_single.append((img, lbl))
    images = [pair[0] for pair in ds_single]
    labels = torch.tensor([pair[1] for pair in ds_single], dtype=torch.long)

    # Batch pass: since CUBTripletDataset returns triples, we rebuild a simple DataLoader
    # For simplicity, just stack in chunks of batch_size:
    all_embeddings = []
    model = ResNet50TripletHead(
        projection_dim=cfg.model.projection_dim,
        freeze_backbone=False  # we only need forward pass
    ).to(device)

    # Load the last checkpoint
    ckpt_dir = cfg.train.checkpoint_dir
    ckpt_files = sorted(os.listdir(ckpt_dir))
    last_ckpt = os.path.join(ckpt_dir, ckpt_files[-1])
    state = torch.load(last_ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    with torch.no_grad():
        for i in range(0, len(images), cfg.embed.batch_size):
            batch_imgs = images[i:i+cfg.embed.batch_size]
            batch_tensor = torch.stack(batch_imgs).to(device)
            batch_emb = model(batch_tensor)  # (B, 128)
            all_embeddings.append(batch_emb.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, 128)

    # 3) Save embeddings and labels
    os.makedirs(cfg.embed.save_dir, exist_ok=True)
    torch.save(all_embeddings, os.path.join(cfg.embed.save_dir, f"{split}_embeddings.pt"))
    torch.save(labels, os.path.join(cfg.embed.save_dir, f"{split}_labels.pt"))
    print(f"Saved {split} embeddings to {cfg.embed.save_dir}")


if __name__ == "__main__":
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "../configs/cub.yaml"))
    dump_embeddings(cfg, split="val")
    dump_embeddings(cfg, split="test")