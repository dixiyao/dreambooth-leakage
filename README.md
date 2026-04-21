# Risks When Sharing LoRA Fine-Tuned Diffusion Model Weights

[![arXiv](https://img.shields.io/badge/arXiv-2409.08482-b31b1b.svg)](https://arxiv.org/abs/2409.08482)

## Introduction

This repository contains the official code for the paper **"Risks When Sharing LoRA Fine-Tuned Diffusion Model Weights"** (arXiv 2409.08482). The work demonstrates a serious privacy vulnerability in the common practice of sharing DreamBooth LoRA fine-tuned diffusion model weights.

When a user fine-tunes a diffusion model (e.g., Stable Diffusion) on private images using DreamBooth and shares the resulting LoRA weights, an adversary who obtains only those weights — without any access to the original images or training prompts — can reconstruct images containing the same identities as the private training data. We propose a **variational network autoencoder** that exploits timestep embeddings to invert LoRA weight updates back into private images. We also show that existing defenses, including differential privacy, cannot adequately protect the training data without severely degrading model utility.

## How the Code Works

### Overview

The attack operates in two stages:

1. **DreamBooth LoRA Fine-tuning** — The victim fine-tunes a pre-trained diffusion model on private images with a personal trigger word, producing LoRA weight updates (∆W).
2. **Trigger Word & Image Inversion Attack** — The adversary, holding only the fine-tuned weights, runs an inversion pipeline to reconstruct the private images and infer the trigger word.

### Repository Structure

```
dreambooth-leakage/
├── train_dreambooth.py          # HuggingFace DreamBooth training script
├── train_dreambooth.sh          # Launch script for DreamBooth fine-tuning
├── requirements.txt
├── inversion_network/
│   ├── attack.py                # Main entry point for the inversion attack
│   ├── inversion_pipeline_diffuse.py   # Core attack pipeline (Algorithm 1)
│   ├── finetune_dreambooth.py   # LoRA fine-tuning utilities with DP support
│   ├── finetune_encoder.py      # Joint encoder training (Algorithm 2)
│   ├── nn_encoder.py            # HyperEncoder: maps LoRA ∆W → embeddings
│   ├── attack_matching_updates.py      # Gradient-matching attack variant
│   ├── matching_gradient_pipeline.py   # Pipeline for gradient-matching attack
│   ├── valid_attack.py          # Attack evaluation (CLIP, DINO, face similarity)
│   ├── metrics.py               # Similarity metrics
│   ├── config.py                # YAML config loader
│   ├── configs/                 # Per-experiment YAML configs
│   ├── dataset/                 # Dataset loaders (DreamBooth, CelebA, CelebA-HQ)
│   └── defense/
│       └── DPGM.py              # Differential Privacy with Gaussian Mixture defense
└── experiments/                 # Experiment configs and instance images
```

### Attack Methods

**Method 1 — Inversion Pipeline** (`inversion_pipeline_diffuse.py` + `finetune_encoder.py`):

- Jointly trains a DreamBooth LoRA model on auxiliary public data and a `HyperEncoder` network.
- `HyperEncoder` (`nn_encoder.py`) learns to map LoRA weight update matrices (∆W) into CLIP's embedding space, effectively decoding what concept the fine-tuning encoded.
- At inference time, the encoder inverts a victim's LoRA weights to produce image embeddings, which are decoded into reconstructed images.

**Method 2 — Gradient Matching** (`matching_gradient_pipeline.py`):

- Simulates DreamBooth fine-tuning steps and searches for a prompt/latent that reproduces the observed weight updates.
- Related to the DLG (Deep Leakage from Gradients) family of attacks adapted to the diffusion model setting.

### Defenses Evaluated

- **Differential Privacy (Opacus)** — gradient clipping + Gaussian noise during fine-tuning.
- **DPGM** (`defense/DPGM.py`) — DP loss via an RBF kernel with Gaussian Mixture noise.

Both defenses are integrated into `finetune_dreambooth.py` and can be toggled via the YAML configs (e.g., `config_celeba_dpgm.yml`).

### Evaluation Metrics (`metrics.py`, `valid_attack.py`)

| Metric | Description |
| --- | --- |
| CLIP Image Similarity | Semantic similarity between reconstructed and original images |
| Face Similarity | Identity preservation (InceptionResNetV1) |
| DINO Similarity | Fine-grained visual similarity via DINO features |

---

## Setup

```bash
pip install -r requirements.txt
```

### Fine-tuning a DreamBooth LoRA Model

```bash
# Using HuggingFace PEFT
git clone https://github.com/huggingface/peft
cd peft/examples/lora_dreambooth
pip install -r requirements.txt
pip install git+https://github.com/huggingface/peft
bash fine-tune.sh
```

Or use the provided script directly:

```bash
bash train_dreambooth.sh
```

`train_dreambooth.py` is the original training code provided by HuggingFace. Training Stable Diffusion v2.1 requires ~16 GB GPU memory (recommended on CUDA).

### Running the Inversion Attack

Edit the appropriate config in `inversion_network/configs/` then:

```bash
cd inversion_network
python attack.py --config configs/config_dreambooth.yml
```

For the gradient-matching variant:

```bash
python attack_matching_updates.py --config configs/config_dreambooth.yml
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{yao2024risks,
  title     = {Risks When Sharing LoRA Fine-Tuned Diffusion Model Weights},
  author    = {Dixi Yao},
  journal   = {arXiv preprint arXiv:2409.08482},
  year      = {2024}
}
```
