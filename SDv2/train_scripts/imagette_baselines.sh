CUDA_VISIBLE_DEVICES=1 python3 train-esd.py \
    --seperator "," \
    --train_method "xattn" \
    --ckpt_path "checkpoints/v2-1_512-ema-pruned.ckpt" \
    --diffusers_config_path "../Better_Erasing/models/erase/config.json" \
    --config_path "configs/stable-diffusion/v2-inference-fp32.yaml" \
    --prompt "imagenette_v1_wo" \
    --info "SDv21" | tee "logs/train_compvis-word_imagenette_v1_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.log"

CUDA_VISIBLE_DEVICES=1 python3 generate_images_ldm_sdv2.py \
    --config="configs/stable-diffusion/v2-inference.yaml" \
    --ckpt "models/compvis-word_imagenette_v1_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/compvis-word_imagenette_v1_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.pt" \
    --prompts_path "../Adversarial-Adaptive/data/imagenette.csv" \
    --H 512 --W 512 --n_samples 1 \
    --outdir "evaluation_massive/compvis-word_imagenette_v1_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/imagenette" | tee "logs/generate_compvis-word_imagenette_v1_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.log"

CUDA_VISIBLE_DEVICES=1 python3 eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path="evaluation_massive/compvis-word_imagenette_v1_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/imagenette/samples/" \
    --prompts_path="../Adversarial-Adaptive/data/imagenette.csv" \
    --save_path="evaluation_folder/compvis-word_imagenette_v1_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21-imagenette.csv"

# prompt = imagenette_v2_wo

CUDA_VISIBLE_DEVICES=1 python3 train-esd.py \
    --seperator "," \
    --train_method "xattn" \
    --ckpt_path "checkpoints/v2-1_512-ema-pruned.ckpt" \
    --diffusers_config_path "../Better_Erasing/models/erase/config.json" \
    --config_path "configs/stable-diffusion/v2-inference-fp32.yaml" \
    --prompt "imagenette_v2_wo" \
    --info "SDv21" | tee "logs/train_compvis-word_imagenette_v2_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.log"

CUDA_VISIBLE_DEVICES=1 python3 generate_images_ldm_sdv2.py \
    --config="configs/stable-diffusion/v2-inference.yaml" \
    --ckpt "models/compvis-word_imagenette_v2_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/compvis-word_imagenette_v2_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.pt" \
    --prompts_path "../Adversarial-Adaptive/data/imagenette.csv" \
    --H 512 --W 512 --n_samples 1 \
    --outdir "evaluation_massive/compvis-word_imagenette_v2_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/imagenette" | tee "logs/generate_compvis-word_imagenette_v2_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.log"

CUDA_VISIBLE_DEVICES=1 python3 eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path="evaluation_massive/compvis-word_imagenette_v2_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/imagenette/samples/" \
    --prompts_path="../Adversarial-Adaptive/data/imagenette.csv" \
    --save_path="evaluation_folder/compvis-word_imagenette_v2_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21-imagenette.csv"


# prompt = imagenette_v3_wo

CUDA_VISIBLE_DEVICES=1 python3 train-esd.py \
    --seperator "," \
    --train_method "xattn" \
    --ckpt_path "checkpoints/v2-1_512-ema-pruned.ckpt" \
    --diffusers_config_path "../Better_Erasing/models/erase/config.json" \
    --config_path "configs/stable-diffusion/v2-inference-fp32.yaml" \
    --prompt "imagenette_v3_wo" \
    --info "SDv21" | tee "logs/train_compvis-word_imagenette_v3_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.log"

CUDA_VISIBLE_DEVICES=1 python3 generate_images_ldm_sdv2.py \
    --config="configs/stable-diffusion/v2-inference.yaml" \
    --ckpt "models/compvis-word_imagenette_v3_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/compvis-word_imagenette_v3_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.pt" \
    --prompts_path "../Adversarial-Adaptive/data/imagenette.csv" \
    --H 512 --W 512 --n_samples 1 \
    --outdir "evaluation_massive/compvis-word_imagenette_v3_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/imagenette" | tee "logs/generate_compvis-word_imagenette_v3_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.log"

CUDA_VISIBLE_DEVICES=1 python3 eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path="evaluation_massive/compvis-word_imagenette_v3_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/imagenette/samples/" \
    --prompts_path="../Adversarial-Adaptive/data/imagenette.csv" \
    --save_path="evaluation_folder/compvis-word_imagenette_v3_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21-imagenette.csv"

# prompt = imagenette_v4_wo

CUDA_VISIBLE_DEVICES=1 python3 train-esd.py \
    --seperator "," \
    --train_method "xattn" \
    --ckpt_path "checkpoints/v2-1_512-ema-pruned.ckpt" \
    --diffusers_config_path "../Better_Erasing/models/erase/config.json" \
    --config_path "configs/stable-diffusion/v2-inference-fp32.yaml" \
    --prompt "imagenette_v4_wo" \
    --info "SDv21" | tee "logs/train_compvis-word_imagenette_v4_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.log"

CUDA_VISIBLE_DEVICES=1 python3 generate_images_ldm_sdv2.py \
    --config="configs/stable-diffusion/v2-inference.yaml" \
    --ckpt "models/compvis-word_imagenette_v4_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/compvis-word_imagenette_v4_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.pt" \
    --prompts_path "../Adversarial-Adaptive/data/imagenette.csv" \
    --H 512 --W 512 --n_samples 1 \
    --outdir "evaluation_massive/compvis-word_imagenette_v4_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/imagenette" | tee "logs/generate_compvis-word_imagenette_v4_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.log"

CUDA_VISIBLE_DEVICES=1 python3 eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path="evaluation_massive/compvis-word_imagenette_v4_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/imagenette/samples/" \
    --prompts_path="../Adversarial-Adaptive/data/imagenette.csv" \
    --save_path="evaluation_folder/compvis-word_imagenette_v4_wo-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21-imagenette.csv"
