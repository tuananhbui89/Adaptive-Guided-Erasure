CUDA_VISIBLE_DEVICES=0 python train_age.py \
    --save_freq 200 \
    --models_path=models \
    --prompt "imagenette_v1_wo" \
    --seperator "," \
    --train_method "xattn" \
    --ckpt_path "checkpoints/v2-1_512-ema-pruned.ckpt" \
    --diffusers_config_path "../Better_Erasing/models/erase/config.json" \
    --config_path "configs/stable-diffusion/v2-inference-fp32.yaml" \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 0.1 \
    --gumbel_hard 1 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 100 \
    --vocab "EN3K" \
    --ignore_special_tokens False \
    --gumbel_topk 2 \
    --lamda 1 \
    --info "gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K" | tee "logs/train_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K.log"

CUDA_VISIBLE_DEVICES=0 python3 generate_images_ldm_sdv2.py \
    --config="configs/stable-diffusion/v2-inference.yaml" \
    --ckpt "models/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K.pt" \
    --prompts_path "../Adversarial-Adaptive/data/imagenette.csv" \
    --H 512 --W 512 --n_samples 1 \
    --outdir "evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K/imagenette" | tee "logs/generate_compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K.log"

CUDA_VISIBLE_DEVICES=0 python3 eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path="evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K/imagenette/samples/" \
    --prompts_path="../Adversarial-Adaptive/data/imagenette.csv" \
    --save_path="evaluation_folder/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K-imagenette.csv"


CUDA_VISIBLE_DEVICES=0 python train_age.py \
    --save_freq 200 \
    --models_path=models \
    --prompt "imagenette_v2_wo" \
    --seperator "," \
    --train_method "xattn" \
    --ckpt_path "checkpoints/v2-1_512-ema-pruned.ckpt" \
    --diffusers_config_path "../Better_Erasing/models/erase/config.json" \
    --config_path "configs/stable-diffusion/v2-inference-fp32.yaml" \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 0.1 \
    --gumbel_hard 1 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 100 \
    --vocab "EN3K" \
    --ignore_special_tokens False \
    --gumbel_topk 2 \
    --lamda 1 \
    --info "gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K" | tee "logs/train_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K.log"

CUDA_VISIBLE_DEVICES=0 python3 generate_images_ldm_sdv2.py \
    --config="configs/stable-diffusion/v2-inference.yaml" \
    --ckpt "models/compvis-adversarial-gumbel-word_imagenette_v2_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K/compvis-adversarial-gumbel-word_imagenette_v2_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K.pt" \
    --prompts_path "../Adversarial-Adaptive/data/imagenette.csv" \
    --H 512 --W 512 --n_samples 1 \
    --outdir "evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v2_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K/imagenette" | tee "logs/generate_compvis-adversarial-gumbel-word_imagenette_v2_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K.log"

CUDA_VISIBLE_DEVICES=0 python3 eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path="evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v2_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K/imagenette/samples/" \
    --prompts_path="../Adversarial-Adaptive/data/imagenette.csv" \
    --save_path="evaluation_folder/compvis-adversarial-gumbel-word_imagenette_v2_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K-imagenette.csv"


# prompt imagenette_v3_wo


CUDA_VISIBLE_DEVICES=0 python train_age.py \
    --save_freq 200 \
    --models_path=models \
    --prompt "imagenette_v3_wo" \
    --seperator "," \
    --train_method "xattn" \
    --ckpt_path "checkpoints/v2-1_512-ema-pruned.ckpt" \
    --diffusers_config_path "../Better_Erasing/models/erase/config.json" \
    --config_path "configs/stable-diffusion/v2-inference-fp32.yaml" \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 0.1 \
    --gumbel_hard 1 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 100 \
    --vocab "EN3K" \
    --ignore_special_tokens False \
    --gumbel_topk 2 \
    --lamda 1 \
    --info "gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K" | tee "logs/train_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K.log"

CUDA_VISIBLE_DEVICES=0 python3 generate_images_ldm_sdv2.py \
    --config="configs/stable-diffusion/v2-inference.yaml" \
    --ckpt "models/compvis-adversarial-gumbel-word_imagenette_v3_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K/compvis-adversarial-gumbel-word_imagenette_v3_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K.pt" \
    --prompts_path "../Adversarial-Adaptive/data/imagenette.csv" \
    --H 512 --W 512 --n_samples 1 \
    --outdir "evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v3_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K/imagenette" | tee "logs/generate_compvis-adversarial-gumbel-word_imagenette_v3_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K.log"

CUDA_VISIBLE_DEVICES=0 python3 eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path="evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v3_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K/imagenette/samples/" \
    --prompts_path="../Adversarial-Adaptive/data/imagenette.csv" \
    --save_path="evaluation_folder/compvis-adversarial-gumbel-word_imagenette_v3_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K-imagenette.csv"

# prompt imagenette_v4_wo


CUDA_VISIBLE_DEVICES=0 python train_age.py \
    --save_freq 200 \
    --models_path=models \
    --prompt "imagenette_v4_wo" \
    --seperator "," \
    --train_method "xattn" \
    --ckpt_path "checkpoints/v2-1_512-ema-pruned.ckpt" \
    --diffusers_config_path "../Better_Erasing/models/erase/config.json" \
    --config_path "configs/stable-diffusion/v2-inference-fp32.yaml" \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 0.1 \
    --gumbel_hard 1 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 100 \
    --vocab "EN3K" \
    --ignore_special_tokens False \
    --gumbel_topk 2 \
    --lamda 1 \
    --info "gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K" | tee "logs/train_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K.log"

CUDA_VISIBLE_DEVICES=0 python3 generate_images_ldm_sdv2.py \
    --config="configs/stable-diffusion/v2-inference.yaml" \
    --ckpt "models/compvis-adversarial-gumbel-word_imagenette_v4_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K/compvis-adversarial-gumbel-word_imagenette_v4_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K.pt" \
    --prompts_path "../Adversarial-Adaptive/data/imagenette.csv" \
    --H 512 --W 512 --n_samples 1 \
    --outdir "evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v4_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K/imagenette" | tee "logs/generate_compvis-adversarial-gumbel-word_imagenette_v4_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K.log"

CUDA_VISIBLE_DEVICES=0 python3 eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path="evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v4_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K/imagenette/samples/" \
    --prompts_path="../Adversarial-Adaptive/data/imagenette.csv" \
    --save_path="evaluation_folder/compvis-adversarial-gumbel-word_imagenette_v4_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_EN3K-imagenette.csv"
