
> **Course context**  Project for SCU *CSEN 240 — Machine Learning* (Spring 2025).
> Authors: Justin Wang, Sudip Aryal, Harshvardhan Garude.

MLProject2: Fine-Grained Image Retrieval with Triplet Loss

This repository implements a supervised image retrieval pipeline on CUB-200-2011 using a ResNet-50 backbone and triplet loss. The goal is to learn a 128-dimensional embedding space where semantically similar bird images (same class) are close and dissimilar images (different classes) are far apart. Once trained, we index gallery embeddings with FAISS and evaluate retrieval performance on held-out splits.
---



MLProject2/
├── configs/
│   └── cub.yaml
├── data/                              # CUB-200-2011 dataset downloaded separately
│   └── CUB_200_2011/
│       ├── images/                    # All CUB images organized by class folder
│       └── splits/                    # Text files listing train/val/test image paths + labels
│           ├── train.txt
│           ├── val.txt
│           └── test.txt
├── outputs/
│   └── checkpoints/                   # Saved model checkpoints during training
├── features/
│   └── embeddings/                    # Saved 128-D embeddings for val/test splits
│       ├── val_embeddings.pt
│       ├── val_labels.pt
│       ├── test_embeddings.pt
│       └── test_labels.pt
├── src/
│   ├── datasets/
│   │   └── cub_triplet.py             # CUBTripletDataset: yields (anchor, positive, negative, label)
│   ├── models/
│   │   └── resnet_triplet.py          # ResNet50TripletHead: ResNet-50 backbone + projection → 128-D
│   ├── train.py                       # Training loop with TripletMarginLoss
│   ├── embed.py                       # Dump 128-D embeddings for val/test splits
│   ├── eval.py                        # Compute Recall@K and mAP using FAISS
│   └── visualize.py                   # Visualize top-K retrieval examples
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
>>>>>>> dbef334 (initial commit completed train.py 30 epoch)
