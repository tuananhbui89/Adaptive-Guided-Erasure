
python train_age.py \
    --save_freq 200 \
    --models_path=models \
    --prompt 'nudity' \
    --seperator ',' \
    --train_method 'noxattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path 'models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path 'models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_topk 2 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 100 \
    --lamda 1 \
    --vocab 'EN3K' \
    --info 'gumbel_lr_1e-2_temp_2_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_EN3K' \

MODEL_NAME='compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_EN3K'
CUDA_VISIBLE_DEVICES=1 python eval-scripts/generate-images.py  \
    --models_path=models \
    --model_name=$MODEL_NAME \
    --prompts_path 'data/unsafe-prompts4703.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0 \
    --to_case -1 &> 'logs/generate_nudity_noxattn_EN3K.log' &


CUDA_VISIBLE_DEVICES=1 python eval-scripts/nudenet-classes.py \
    --threshold 0.0 \
    --folder='evaluation_massive/compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_EN3K/unsafe-prompts4703' \
    --prompts_path='data/unsafe-prompts4703.csv' \
    --save_path='evaluation_folder/unsafe/compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_EN3K-data-unsafe.csv' 

