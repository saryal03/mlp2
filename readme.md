
> **Course context**  Project for SCU *CSEN 240 — Machine Learning* (Spring 2025).
> Authors: Justin Wang, Sudip Aryal, Harshvardhan Garude.

# MLProject2: Triplet‐Loss Retrieval & CLIP‐Based Text Query on CUB-200-2011

This repository implements a fine-grained image retrieval pipeline on the CUB-200-2011 dataset. It covers:

1. **Triplet-loss training** with a ResNet-50 backbone → 128-D embedding head.
2. **Embedding extraction** for validation and test splits.
3. **Retrieval evaluation** (Recall@1/5/10, mAP) using NumPy.
4. **Qualitative visualization** of top-K retrievals (image → image).
5. **Text-to-image retrieval** using CLIP (text → image).

All code is organized under `src/`. Configuration resides in `configs/cub.yaml`. A Python 3 virtual environment is used for dependency isolation.

