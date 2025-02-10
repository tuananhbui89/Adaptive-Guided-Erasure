<div align="center">

# Adaptive Guided Erasure

**"Fantastic Targets for Concept Erasure in Diffusion Models and Where To Find Them"** (ICLR 2025)

[[📄 Paper]](https://arxiv.org/abs/2501.18950)

Contact: tuananh.bui@monash.edu

<div align="left">

**(Shameless plug :grin:) Our other papers on Concept Erasing/Unlearning:**

> [**Fantastic Targets for Concept Erasure in Diffusion Models and Where to Find Them**](https://www.dropbox.com/scl/fi/pf2190qpfpiuo05mhcqmi/Adaptive-Guide-Erasure.pdf?rlkey=63s7ruwqxhrdsc4i603gjmsri&st=y79mr0ej&dl=0),       
> Tuan-Anh Bui, Trang Vu, Long Vuong, Trung Le, Paul Montague, Tamas Abraham, Dinh Phung       
> *ICLR 2025 ([arXiv 2501.18950](https://arxiv.org/abs/2501.18950))*

> [**Erasing Undesirable Concepts in Diffusion Models with Adversarial Preservation**](https://arxiv.org/abs/2410.15618),       
> Tuan-Anh Bui, Long Vuong, Khanh Doan, Trung Le, Paul Montague, Tamas Abraham, Dinh Phung       
> *NeurIPS 2024 ([arXiv 2410.15618](https://arxiv.org/abs/2410.15618))*

> [**Removing Undesirable Concepts in Text-to-Image Generative Models with Learnable Prompts**](https://arxiv.org/abs/2403.12326),       
> Tuan-Anh Bui, Khanh Doan, Trung Le, Paul Montague, Tamas Abraham, Dinh Phung       
> *Preprint ([arXiv 2403.12326](https://arxiv.org/abs/2403.12326))*

---

## Abstract

Concept erasure has emerged as a promising technique for mitigating the risk of harmful content generation in diffusion models by selectively unlearning undesirable concepts. The common principle of previous works to remove a specific concept is to map it to a fixed generic concept, such as a neutral concept or just an empty text prompt. In this paper, we demonstrate that this fixed-target strategy is suboptimal, as it fails to account for the impact of erasing one concept on the others. To address this limitation, we model the concept space as a graph and empirically analyze the effects of erasing one concept on the remaining concepts. Our analysis uncovers intriguing geometric properties of the concept space, where the influence of erasing a concept is confined to a local region. Building on this insight, we propose the Adaptive Guided Erasure (AGE) method, which \emph{dynamically} selects optimal target concepts tailored to each undesirable concept, minimizing unintended side effects. Experimental results show that AGE significantly outperforms state-of-the-art erasure methods on preserving unrelated concepts while maintaining effective erasure performance.


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
****

## Usage

We provide training and evaluation scripts for the experiments in the paper in the `scripts` folder.

To produce the results in Section 3 of the paper (i.e., Concept Graph on NetFive dataset), run the bash files in the `scripts/netfive` folder.

To produce the results in Table 1 of the paper (i.e., Erasing object-related concepts), run the following command:

```bash
bash scripts/imagenette/erase_imagenette.sh
```

To produce the results in Table 2 of the paper (i.e., Erasing nudity concept), run the following command:

```bash
bash scripts/erasing_nudity.sh
```

To produce the results in Table 3 of the paper (i.e., Erasing Artistic Concepts), run the following command:

```bash
bash scripts/artist/erase_artist.sh
```

The list of prompts used in the paper can be found in the `data` folder, including:

- `english_3000.csv`: List of 3000 English words
- `imagenette.csv`: List of imagenette classes, 500 images per class
- `unsafe-prompts4703.csv`: List of unsafe prompts I2P, 4703 prompts
- `long_nich_art_prompts.csv`: List to generate artistic from five artists
- `netfive.csv`: List of objects from NetFive dataset

We provide implementation of our method and baselines:

- `train_age.py`: Implementation of our method
- `train_esd.py`: Implementation of ESD 
- `train_uce.py`: Implementation of UCE
- `train-esd-preserve.py`: Implementation of ESD with preservation to study the impact of erasing nudity and garbage truck concepts

To set concepts to erase, modify the `utils_exp.py` file and change the argument `--prompt` in the bash files.

## Stable Diffusion version 2

We provide the code for experiments in the paper with Stable Diffusion version 2 in the `SDv2` folder.

## MACE

We provide the code for MACE in the `MACE` folder. This code has been cloned from the original repository at [MACE](https://github.com/Shilin-LU/MACE/tree/main). Please see the README.md in the `MACE` folder to set up its required environment.

To run related experiments, please see the `run_mace_imagenette.sh` and `run_mace_art.sh` files.

## Forget-Me-Not

We have tried to replicate the results of Forget-Me-Not in the paper using the original code from the authors' repository [Forget-Me-Not](https://github.com/SHI-Labs/Forget-Me-Not). However, we have not been able to replicate the results.
We would like to provide the code here for readers' reference.

Specifically, we followed these steps:

1. Provided representative images of the target concepts (8 images per concept) to train the inversion models.
2. Executed the inversion process using the `ti_config.yaml` configuration file.
3. Conducted the erasure process with the `attn.yaml` configuration file.
4. Generated and stored images with the erased concepts in the `evaluation_folder/exps_attn` directory.
****

## Citation

If you find this work useful in your research, please consider citing our paper (or our other papers :grin:):

```bibtex
@article{bui2024erasing,
  title={Erasing Undesirable Concepts in Diffusion Models with Adversarial Preservation},
  author={Bui, Anh and Vuong, Long and Doan, Khanh and Le, Trung and Montague, Paul and Abraham, Tamas and Phung, Dinh},
  booktitle={NeurIPS},
  year={2024}
}

@article{bui2025fantastic,
  title={Fantastic Targets for Concept Erasure in Diffusion Models and Where to Find Them},
  author={Bui, Anh and Vu, Trang and Vuong, Long and Le, Trung and Montague, Paul and Abraham, Tamas and Phung, Dinh},
  journal={ICLR},
  year={2025}
}

@article{bui2024removing,
  title={Removing Undesirable Concepts in Text-to-Image Generative Models with Learnable Prompts},
  author={Bui, Anh and Doan, Khanh and Le, Trung and Montague, Paul and Abraham, Tamas and Phung, Dinh},
  journal={arXiv preprint arXiv:2403.12326},
  year={2024}
}
```

## References

This repository is based on the repository [Erasing Concepts from Diffusion Models](https://github.com/rohitgandikota/erasing)