# Adaptive-Guided-Erasure

Code for the paper *"Optimal Targets for Concept Erasure in Diffusion Models and Where to Find Them"*.

## Installation Guide

```bash
cd Adaptive-Guided-Erasure
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt
mkdir models/erase
mv sd-v1-4-full-ema.ckpt models/erase/
wget https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/config.json
mv config.json models/erase/
```

Requirements:

```bash
pip install omegaconf
pip install pytorch-lightning==1.6.5
pip install taming-transformers-rom1504
pip install kornia==0.5.11
pip install git+https://github.com/openai/CLIP.git
pip install diffusers==0.21.4
pip install -U transformers
pip install --upgrade nudenet
pip install lpips
```

