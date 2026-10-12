# LoRA-Pre-AdamW: A Reproduction

This repository contains a from-scratch implementation and reproduction of the paper:
**"Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation" (ICLR 2026)**

## Project Structure
- `lora_pre_adamw_optimizer.py`: Custom LoRA-Pre-AdamW optimizer
- `pretrain_adamw.py`: Baseline AdamW training script
- `pretrain_lora_pre_adamw.py`: LoRA-Pre-AdamW training script
- `plot_results.py`: Script for generating comparison charts
- `test_inference.py`: Inference and evaluation script

## Environment
- WSL2 + CUDA 12.4 + PyTorch 2.6
- Model: SmolLM2-135M, pre-trained from scratch on C4 dataset.

## Reproduction Plan
- [x] Environment Setup
- [x] Custom Optimizer Implementation
- [x] Initial Feasibility Runs (2000 & 5000 steps)
- [ ] Comprehensive Benchmarks & Ablations (Targeting Final Presentation)

## Citation
Zhengbo Wang et al. "Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation." ICLR 2026.