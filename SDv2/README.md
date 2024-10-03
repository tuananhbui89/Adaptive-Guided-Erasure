# Adaptive Guided Erasure with Stable Diffusion Version 2

This repository contains the code for the project "Adaptive Guided Erasure" with the Stable Diffusion version 2 codebase.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers==4.30.2
pip install diffusers==0.21.4
pip install ftfy==6.1.1
```

Clone and copy the `ldm` repository to the root of this repository:

```bash
git clone https://github.com/Stability-AI/stablediffusion.git
cp -rf stablediffusion/ldm Adaptive-Guided-Erasure/SDv2/
cp -rf stablediffusion/configs Adaptive-Guided-Erasure/SDv2/
cp -rf stablediffusion/scripts Adaptive-Guided-Erasure/SDv2/
```

Download the model weights and put them in the `checkpoints` directory:

Note: The inference config for all model versions is designed to be used with EMA-only checkpoints. For this reason use_ema=False is set in the configuration, otherwise the code will try to switch from non-EMA to EMA weights.

```bash
mkdir checkpoints 
cd checkpoints
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt?download=true
```

## Change in the codebase

- File `ldm/modules/encoders/modules.py`, class `FrozenOpenCLIPEmbedder`, method `__init__`: get tokenizer
- File `ldm/models/diffusion/ddim.py`, class `DDIMSampler`, method `sample`: add `till_T` and `t_start` parameters as similar to version 1


