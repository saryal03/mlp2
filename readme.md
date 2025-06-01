<<<<<<< HEAD
# Open‑Set Fine‑Grained Image Retrieval (OS‑FGR)

<img alt="CLIP + LoRA diagram" src="docs/architecture.svg" width="660"/>

**OS‑FGR** is a research prototype that studies *open‑set* fine‑grained image retrieval using vision–language models (VLMs). It builds on [CLIP](https://arxiv.org/abs/2103.00020) with prompt engineering and lightweight **LoRA** adapters, supports three public datasets (CUB‑200‑2011, Stanford Cars, iNaturalist‑2017), and achieves ≥ +3 pp R\@1 over the frozen‑CLIP baseline while keeping inference latency ⩽ 10 ms on a single RTX 4090.

> **Course context**  Project for SCU *CSEN 240 — Machine Learning* (Spring 2025).
> Authors: Justin Wang, Sudip Aryal, Harshvardhan Garude.

---

## 1 Features

* **Modular pipeline** – `train.py`, `embed.py`, `index.py`, `eval.py` (or one‑shot `retrieval_pipeline.py`).
* **Prompt engineering** – static templates + learnable CoOp‑style context tokens.
* **LoRA fine‑tuning** – only 1‑2 % of CLIP parameters updated.
* **FAISS IVF‑PQ index** – sub‑10 ms query time on 100 k gallery.
* **Open‑set detection** – Mahalanobis threshold calibrated on validation set.
* **Reproducible experiments** – YAML configs, Weights & Biases logging.

---

## 2 Installation

```bash
# Clone and create env (conda or venv)
git clone https://github.com/your‑org/os‑fgr.git
cd os‑fgr
conda create -n osfgr python=3.10 -y
conda activate osfgr

# Install requirements
pip install -r requirements.txt

# (Optional GPU FAISS)
#   pip install faiss‑gpu --extra‑index-url https://pypi.ngc.nvidia.com
```

Test the installation with a one‑liner:

```bash
python src/tests/smoke_test.py   # should finish in < 30 s and print Recall numbers
```

---

## 3 Datasets

| Dataset                 | Script                   | Size   | Notes                       |
| ----------------------- | ------------------------ | ------ | --------------------------- |
| CUB‑200‑2011            | `tools/download_cub.sh`  | 1.1 GB | Bird species, 200 classes   |
| Stanford Cars           | `tools/download_cars.sh` | 1.7 GB | Car make/model, 196 classes |
| iNaturalist‑2017 (mini) | `tools/download_inat.sh` | 4 GB   | Highly imbalanced wildlife  |

All scripts download, verify checksums, and unpack to `data/<dataset>`.

---

## 4 Quick Start

```bash
# (1) Fine‑tune LoRA adapters on CUB (≈ 15 min on RTX 4090)
python src/train.py --cfg configs/cub_lora.yaml

# (2) Extract features for gallery & queries
python src/embed.py --cfg configs/cub_lora.yaml --split test

# (3) Build FAISS index and evaluate Recall@{1,5,10}
python src/index.py --cfg configs/cub_lora.yaml
python src/eval.py  --cfg configs/cub_lora.yaml
```

A one‑shot demo that trains, indexes, and evaluates is available:

```bash
python retrieval_pipeline.py  # quick functional test
```

Expected CUB open‑set results (ViT‑B/16, 1‑epoch LoRA):

```
R@1  = 74.6 % (baseline 71.5 %)
R@5  = 90.1 %
mAP  = 61.0 %
Latency ≈ 8 ms/query (ANN, batch=1)
```

---

## 5 Repo Structure

```
os‑fgr/
├── README.md
├── requirements.txt
├── .gitignore
├── retrieval_pipeline.py  # all‑in‑one demo
│
├── src/
│   ├── config.py          # YAML→dict converter
│   ├── datasets/          # CUB, Cars, iNat loaders
│   ├── models/            # CLIP‑LoRA, prompt tuning
│   ├── train.py
│   ├── embed.py
│   ├── index.py
│   ├── eval.py
│   └── utils/
│
├── configs/               # *.yaml files per experiment
├── tools/                 # data download scripts
└── tests/                 # smoke & unit tests
```

---

## 6 Citation

If you use this codebase or report in academic work, please cite:

```bibtex
@misc{wang2025osfgr,
  title   = {Open‑Set Fine‑Grained Image Retrieval with Vision–Language Models},
  author  = {Justin Wang and Sudip Aryal and Harshvardhan Garude},
  year    = {2025},
  note    = {Course project, Santa Clara University},
  url     = {https://github.com/your‑org/os‑fgr}
}
```

---

## 7 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 8 Acknowledgements

We thank the creators of CLIP, PEFT, and FAISS, whose open‑source work made this project possible.
=======
MLProject2: Fine-Grained Image Retrieval with Triplet Loss

This repository implements a supervised image retrieval pipeline on CUB-200-2011 using a ResNet-50 backbone and triplet loss. The goal is to learn a 128-dimensional embedding space where semantically similar bird images (same class) are close and dissimilar images (different classes) are far apart. Once trained, we index gallery embeddings with FAISS and evaluate retrieval performance on held-out splits.

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
