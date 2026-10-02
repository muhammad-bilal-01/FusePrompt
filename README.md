# FusePrompt: A Prompt Fusion Framework for Vision-Language Adaptation

[![Paper](https://img.shields.io/badge/Paper-Image%20and%20Vision%20Computing-blue)](#citation)
[![Python](https://img.shields.io/badge/Python-3.8-green)](#installation)

> **FusePrompt: A Prompt Fusion Framework for Vision-Language Adaptation**
> Muhammad Bilal, Muhammad Shehzad Hanif, Muhammad Bilal, Wenxiong Kang, Arif Mahmood
> *Image and Vision Computing* <!-- TODO: add volume/year/DOI/arXiv link when available -->

Official implementation of **FusePrompt**, a prompt-driven fusion framework that jointly trains a shallow independent prompt learner (**IPL**, PromptSRC-style) and a deep coupled multi-modal prompt learner (**MMPL**, MaPLe-style) on top of a single frozen CLIP backbone.

<p align="center">
  <img src="docs/fuseprompt.jpg" alt="FusePrompt architecture" width="100%">
</p>

---

## Table of Contents

- [Highlights](#highlights)
- [Method Overview](#method-overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Reproducing the Paper Results](#reproducing-the-paper-results)
- [Hyperparameters](#hyperparameters)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Highlights

> **Abstract:** Prompt learning has become a practical way to adapt vision-language models (VLMs) such as CLIP for few-shot and cross-domain tasks without updating the backbone. However, existing methods often emphasize domain-specific optimization and typically rely on a single prompting strategy, limiting their representational diversity. FusePrompt ensembles shallow and deep prompting strategies through Independent Vision-Language Prompting (IPL) and Multi-Modal Prompt Learning (MMPL), jointly trained within a single shared backbone. To prevent the two branches from converging to the same representation, we introduce a feature-level diversity loss and regularize each branch against a different set of frozen CLIP text features. Across eleven datasets, FusePrompt outperforms each individual branch and matches or exceeds prior prompt learning methods on the majority of benchmarks under both base-to-novel and cross-dataset settings.

**Main contributions**

1. **Dual-branch joint training.** IPL (shallow, independent prompts) and MMPL (deep, coupled prompts) are co-optimized in one shared frozen CLIP backbone, not trained separately and merged afterwards.
2. **Cosine-based diversity loss** on both image and text features penalizes overlap between the two branches, so they do not learn redundant representations.
3. **50 paraphrased text templates** for the MMPL branch (standard ImageNet templates for IPL) add linguistic diversity, and each branch is regularized against its own frozen text features (asymmetric frozen design).

**Lightweight:** only ~0.06M (64,512) trainable parameters per branch on top of frozen CLIP ViT-B/16.

---

## Method Overview

FusePrompt runs two prompt learners over the same frozen CLIP:

| | IPL branch | MMPL branch |
|---|---|---|
| Inspired by | PromptSRC | MaPLe |
| Prompting | Shallow, independent vision and text prompts | Deep, coupled prompts across multiple layers (text prompts projected into vision prompts) |
| Frozen text reference | Standard ImageNet templates (`g'^1`) | 50 paraphrased templates (`g'^2`) |
| Frozen image reference | Frozen CLIP image features (`f'_CLIP`) shared by both branches | Frozen CLIP image features (`f'_CLIP`) shared by both branches |

**Training objective**

```
L_final = L_CE + L_SCL + λ4 · L_diversity
L_SCL   = (λ1/2) · L_SCL-image + (λ2/2) · L_SCL-text + (1/2) · L_SCL-logits
L_diversity = L_image-diversity + λ3 · L_text-diversity
```

- `L_CE`: cross-entropy on the **averaged logits** of both branches.
- `L_SCL`: self-consistency (L1 between prompted and frozen features, plus KL on logits), summed over both branches.
- `L_diversity`: squared dot product between the two branches' prompted features (image and text).

**Inference:** logits from IPL and MMPL are averaged with equal weight (0.5 / 0.5). Gaussian Weighted Prompt Aggregation (GPA) is applied to each branch over all training epochs.

---

## Results

### Base-to-Novel Generalization (average over 11 datasets, ViT-B/16)

| Method | Base | Novel | HM |
|---|:---:|:---:|:---:|
| CLIP | 69.34 | 74.22 | 71.70 |
| CoOp | 82.69 | 63.22 | 71.66 |
| CoCoOp | 80.47 | 71.69 | 75.83 |
| MaPLe | 82.28 | 75.14 | 78.55 |
| PromptSRC | 84.26 | 76.10 | 79.97 |
| **FusePrompt** | **85.99** | **76.25** | **80.83** |

<details>
<summary><b>Per-dataset base-to-novel results (click to expand)</b></summary>

| Dataset | Base | Novel | HM |
|---|:---:|:---:|:---:|
| ImageNet | 78.16 | 70.08 | 73.90 |
| Caltech101 | 98.71 | 95.09 | 96.87 |
| OxfordPets | 96.28 | 97.82 | 97.04 |
| StanfordCars | 84.01 | 72.07 | 77.58 |
| Flowers102 | 98.29 | 77.16 | 86.45 |
| Food101 | 90.54 | 91.06 | 90.80 |
| FGVCAircraft | 48.74 | 33.29 | 39.56 |
| SUN397 | 82.82 | 78.55 | 80.63 |
| DTD | 85.30 | 63.29 | 72.66 |
| EuroSAT | 94.24 | 79.38 | 86.17 |
| UCF101 | 88.83 | 80.91 | 84.69 |

</details>

### Individual branches vs. FusePrompt (average over 11 datasets)

| | IPL | MMPL | **FusePrompt** |
|---|:---:|:---:|:---:|
| Base | 83.30 | 80.35 | **85.99** |
| Novel | 71.04 | 74.77 | **76.25** |
| HM | 76.68 | 77.46 | **80.83** |

### Cross-Dataset Evaluation (train on ImageNet, test on 10 unseen datasets)

| | CoOp | CoCoOp | MaPLe | PromptSRC | **FusePrompt** |
|---|:---:|:---:|:---:|:---:|:---:|
| ImageNet (source) | 71.51 | 71.02 | 70.72 | 71.27 | **71.88** |
| Average (10 targets) | 63.88 | 65.74 | 66.30 | 65.81 | 65.28 |

FusePrompt improves over PromptSRC on ImageNet, Aircraft, Caltech101 and EuroSAT while staying competitive on the rest. See the paper for the full table.

---

## Repository Structure

```
FusePrompt/
├── Dassl.pytorch/      # Dassl training framework (bundled)
├── clip/               # CLIP model code
├── configs/            # dataset and trainer YAML configs
├── datasets/           # dataset loaders
├── docs/               # extra documentation (dataset preparation, figures)
├── interpret_prompts/  # prompt interpretation utilities
├── lpclip/             # linear-probe CLIP utilities
├── scripts/            # training / evaluation shell scripts
├── trainers/           # FusePrompt trainer (IPL + MMPL + losses)
├── clip_words.csv
├── parse_test_res.py   # averages results over seeds
├── requirements.txt
└── train.py            # main entry point
```

---

## Installation

The code is built on [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) and the codebases of [PromptSRC](https://github.com/muzairkhattak/PromptSRC) and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning). Tested with Python 3.8.

```bash
# 1. Clone the repository
git clone https://github.com/muhammad-bilal-01/FusePrompt.git
cd FusePrompt

# 2. Create and activate a conda environment
conda create -y -n fuseprompt python=3.8
conda activate fuseprompt

# 3. Install PyTorch and torchvision (>= 1.8.1; pick the build for your CUDA version)
#    See https://pytorch.org/get-started/previous-versions/
pip install torch torchvision

# 4. Install the remaining requirements
pip install -r requirements.txt

# 5. Install Dassl (already included in this repo, no need to clone)
cd Dassl.pytorch/
pip install -r requirements.txt
python setup.py develop
cd ..
```

---

## Data Preparation

FusePrompt uses the same 11 datasets and the same folder layout as CoOp / CoCoOp / PromptSRC:

ImageNet, Caltech101, OxfordPets, StanfordCars, Flowers102, Food101, FGVCAircraft, SUN397, DTD, EuroSAT, UCF101.

Follow the dataset instructions in [`docs/DATASETS.md`](docs/DATASETS.md) to download and arrange the data (the same steps as [CoOp's `DATASETS.md`](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)). After this, your dataset root should look like:

```
$DATA/
├── imagenet/
├── caltech-101/
├── oxford_pets/
├── stanford_cars/
├── oxford_flowers/
├── food-101/
├── fgvc_aircraft/
├── sun397/
├── dtd/
├── eurosat/
└── ucf101/
```

Set `DATA` to this folder inside the shell scripts (see below).

---

## Training and Evaluation

> **Before running:** open the script you want in `scripts/` and set `DATA=/path/to/your/datasets`.
> All commands are run from the repository root. The seed (`1`, `2`, `3`) is the last argument.

The default config is in `configs/trainers/FusePrompt/` (ViT-B/16, 4 vision + 4 text prompt tokens, 50 epochs, 16 shots).

### 1) Base-to-Novel Generalization

Train on the base classes, then evaluate on base and novel classes.

```bash
# Syntax: bash scripts/fuseprompt/base2new_train.sh <dataset> <seed>
#         bash scripts/fuseprompt/base2new_test.sh  <dataset> <seed>

# Example: EuroSAT, seed 1
bash scripts/fuseprompt/base2new_train.sh eurosat 1     # train on base classes
bash scripts/fuseprompt/base2new_test.sh  eurosat 1     # evaluate on novel classes
```

Possible `<dataset>` values: `imagenet`, `caltech101`, `oxford_pets`, `stanford_cars`, `oxford_flowers`, `food101`, `fgvc_aircraft`, `sun397`, `dtd`, `eurosat`, `ucf101`.

### 2) Cross-Dataset Evaluation

Train on all 1000 ImageNet classes (16 shots), then test directly on the other 10 datasets with no fine-tuning.

```bash
# Train on ImageNet (source)
bash scripts/fuseprompt/xd_train.sh imagenet 1

# Evaluate on a target dataset using the ImageNet-trained model
bash scripts/fuseprompt/xd_test.sh caltech101 1
```

### 3) Evaluate a trained model (without retraining)

```bash
python train.py \
    --root /path/to/datasets \
    --seed 1 \
    --trainer FusePrompt \
    --dataset-config-file configs/datasets/eurosat.yaml \
    --config-file configs/trainers/FusePrompt/<config_name>.yaml \
    --output-dir output/eval/eurosat/seed1 \
    --model-dir output/base2new/train_base/eurosat/seed1 \
    --load-epoch 50 \
    --eval-only \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES new
```

### 4) Averaging results over seeds

```bash
# base-to-novel (example for the 'new' split of EuroSAT)
python parse_test_res.py output/base2new/test_new/eurosat/shots_16/FusePrompt/<config_name>

# add --multi-exp if the directory contains several experiments
```

---

## Reproducing the Paper Results

1. Install the environment and prepare all 11 datasets (above).
2. Use the **default configuration**, which matches the paper (see [Hyperparameters](#hyperparameters)).
3. Run **three seeds (1, 2, 3)** for every dataset. In the paper, tables report the best result across the three runs. The mean ± std over seeds is given in the supplementary material.
4. For base-to-novel: run `base2new_train.sh` then `base2new_test.sh` for each dataset and seed. 5. For cross-dataset: train once on ImageNet, then run `xd_test.sh` for the 10 target datasets.
6. Averages in the main table are taken over the 11 datasets.

A simple loop to run everything for base-to-novel:

```bash
for DATASET in imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101
do
  for SEED in 1 2 3
  do
    bash scripts/fuseprompt/base2new_train.sh ${DATASET} ${SEED}
    bash scripts/fuseprompt/base2new_test.sh  ${DATASET} ${SEED}
  done
done
```

**Hardware used in the paper:** few-shot experiments on a T4 GPU (batch size 4; ImageNet on an L4 with batch size 8), cross-dataset training on an A100 (batch size 16). Small differences from the reported numbers can occur with different GPUs, CUDA versions and seeds.

---

## Hyperparameters

| Setting | Value |
|---|---|
| Backbone | CLIP ViT-B/16 (frozen) |
| Prompt length (vision / text) | 4 / 4 |
| Epochs | 50 |
| Batch size | 4 (8 for ImageNet base-to-novel, 16 for cross-dataset) |
| Shots per class | 16 |
| Learning rate | 0.0025 |
| Prompt init | random, except first-layer text prompt = "a photo of a" |
| SCL weights | λ1 = 10 (image), λ2 = 25 (text) |
| Diversity weights | λ3 = 1, λ4 = 1 |
| Text templates | IPL: standard ImageNet templates; MMPL: 50 paraphrased templates |
| Inference | equal-weight average of IPL and MMPL logits, GPA on each branch |
| Seeds | 1, 2, 3 |

Everything above can be changed from the YAML config in `configs/trainers/FusePrompt/` or by overriding keys from the command line.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{bilal2026fuseprompt,
  title   = {FusePrompt: A Prompt Fusion Framework for Vision-Language Adaptation},
  author  = {Bilal, Muhammad and Hanif, Muhammad Shehzad and Bilal, Muhammad and Kang, Wenxiong and Mahmood, Arif},
  journal = {Image and Vision Computing},
  year    = {2026}
}
```

<!-- TODO: update volume, pages and DOI after publication -->

---

## Acknowledgements

This codebase builds on the excellent work of:

- [PromptSRC](https://github.com/muzairkhattak/PromptSRC) (IPL branch and training pipeline)
- [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) (MMPL branch)
- [CoOp / CoCoOp](https://github.com/KaiyangZhou/CoOp) (datasets and evaluation protocol)
- [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) (training framework)
- [CLIP](https://github.com/openai/CLIP)

We thank the authors for making their code publicly available.

---

## Contact

For questions or issues, please open a [GitHub issue](https://github.com/muhammad-bilal-01/FusePrompt/issues) or contact:

- Corresponding author: Muhammad Shehzad Hanif (mshanif@kau.edu.sa)
