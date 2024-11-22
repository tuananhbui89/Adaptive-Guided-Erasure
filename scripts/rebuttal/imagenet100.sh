# gumbel-v7
CUDA_VISIBLE_DEVICES=0 python train_age.py                     --save_freq 200                     --models_path=models                     --prompt "netfive"                     --seperator ","                     --train_method "xattn"                     --config_path "configs/stable-diffusion/v1-inference.yaml"                     --ckpt_path "models/erase/sd-v1-4-full-ema.ckpt"                     --diffusers_config_path "models/erase/config.json"                     --lr 1e-5                     --gumbel_lr 1e-2                     --gumbel_temp 0.1                     --gumbel_hard 1                     --gumbel_num_centers 100                     --gumbel_update -1                     --gumbel_time_step 0                     --gumbel_multi_steps 2                     --gumbel_k_closest 100                     --vocab "Imagenet"                     --ignore_special_tokens False                     --gumbel_topk 2                     --lamda 1                     --info "gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet"                     
CUDA_VISIBLE_DEVICES=0 python generate_images_ldm.py                     --models_path=models                     --model_name="compvis-age-word_netfive-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet"                     --prompts_path "data/imagenet100_100.csv"                     --save_path "evaluation_massive"                     --num_samples 1                     --from_case 7270                     --to_case -1 &> logs/imagenet100_100_gumbel_v7_from_7270_to_10000.log &              
CUDA_VISIBLE_DEVICES=0 python eval-scripts/imageclassify.py                     --topk=10                     --folder_path="evaluation_massive/compvis-age-word_netfive-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet/ldm-imagenet100_100/"                     --prompts_path="data/imagenet100_100.csv"                     --save_path="evaluation_folder/compvis-age-word_netfive-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v7_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet-ldm-imagenet100_100.csv"                     

# SD-v1-4

CUDA_VISIBLE_DEVICES=1 python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name 'SD-v1-4' \
    --prompts_path 'data/imagenet100_100.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0 \
    --to_case -1 &> logs/imagenet100_100_sd_v1_4_from_0_to_10000.log &

CUDA_VISIBLE_DEVICES=1 python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/SD-v1-4/imagenet100_100/' \
    --prompts_path='data/imagenet100_100.csv' \
    --save_path='evaluation_folder/SD-v1-4-imagenet100_100.csv'

# ESD 

CUDA_VISIBLE_DEVICES=0 python train-esd.py --seperator "," --train_method "xattn" --ckpt_path "models/erase/sd-v1-4-full-ema.ckpt" --diffusers_config_path "models/erase/config.json" --prompt "netfive" --config_path "configs/stable-diffusion/v1-inference.yaml" --info "none" | tee "logs/target-empty_0_train_netfive.log"
CUDA_VISIBLE_DEVICES=0 python eval-scripts/generate-images.py --models_path=models --model_name="compvis-word_netfive-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none" --prompts_path "data/imagenet100_100.csv" --save_path "evaluation_massive" --num_samples 1 --from_case 0 --to_case -1 &> logs/imagenet100_100_esd_from_0_to_10000.log &
CUDA_VISIBLE_DEVICES=0 python eval-scripts/imageclassify.py --topk=10 --folder_path="evaluation_massive/compvis-word_netfive-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none/imagenet100_100/" --prompts_path="data/imagenet100_100.csv" --save_path="evaluation_folder/compvis-word_netfive-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none-imagenet100_100.csv"

CUDA_VISIBLE_DEVICES=0 python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name="compvis-word_netfive-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none" \
    --prompts_path "data/CliPNet75_100.csv" \
    --save_path "evaluation_massive" \
    --num_samples 1 \
    --from_case 0 \
    --to_case -1 &> logs/CliPNet75_100_esd_from_0_to_7500.log &

CUDA_VISIBLE_DEVICES=0 python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path="evaluation_massive/compvis-word_netfive-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none/CliPNet75_100/" \
    --prompts_path="data/CliPNet75_100.csv" \
    --save_path="evaluation_folder/compvis-word_netfive-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none-CliPNet75_100.csv"

# UCE 

CUDA_VISIBLE_DEVICES=0 python train-uce.py --prompt 'netfive' --technique 'tensor' --concept_type 'object' --base '1.4' --add_prompts False --info 'none' # DONE

CUDA_VISIBLE_DEVICES=0 python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='uce-erased-netfive-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none' \
    --prompts_path 'data/imagenet100_100.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0 \
    --to_case -1 &> logs/imagenet100_100_uce_from_0_to_10000.log &

CUDA_VISIBLE_DEVICES=0 python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/uce-erased-netfive-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none/imagenet100_100/' \
    --prompts_path='data/imagenet100_100.csv' \
    --save_path='evaluation_folder/uce-erased-netfive-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none-imagenet100_100.csv'
