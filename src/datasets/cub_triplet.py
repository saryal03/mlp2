import os
import random
from PIL import Image
from torch.utils.data import Dataset

class CUBTripletDataset(Dataset):
    """
    Expects:
      - cub_root/images/<class_folder>/<image_name>.jpg
      - split_txt: each line "images/<class_folder>/<image_name>.jpg <class_id>"
    Returns: (anchor_img, positive_img, negative_img, anchor_label)
    """

    def __init__(self, cub_root: str, split_txt: str, transform=None):
        super().__init__()
        self.cub_root = cub_root
        self.transform = transform

        # Parse split_txt to build list of (full_image_path, label)
        self.image_paths = []
        self.labels = []
        with open(split_txt, "r") as f:
            for line in f:
                rel_path, label = line.strip().split()
                full_path = os.path.join(cub_root, rel_path)
                self.image_paths.append(full_path)
                self.labels.append(int(label))

        # Build a mapping from label -> list of indices
        self.label_to_indices = {}
        for idx, lbl in enumerate(self.labels):
            self.label_to_indices.setdefault(lbl, []).append(idx)

        # All unique labels
        self.labels_set = sorted(self.label_to_indices.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]

        # Positive: random sample from same class (but not idx itself if possible)
        pos_indices = self.label_to_indices[anchor_label]
        if len(pos_indices) > 1:
            pos_idx = idx
            while pos_idx == idx:
                pos_idx = random.choice(pos_indices)
        else:
            pos_idx = idx
        positive_path = self.image_paths[pos_idx]

        # Negative: random sample from a different class
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = random.choice(self.labels_set)
        neg_idx = random.choice(self.label_to_indices[neg_label])
        negative_path = self.image_paths[neg_idx]

        # Load images
        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, anchor_label