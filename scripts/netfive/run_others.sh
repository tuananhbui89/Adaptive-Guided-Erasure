# erasing nudity

# CUDA_VISIBLE_DEVICES=2 python eval-scripts/imageclassify.py --topk=10 --folder_path="evaluation_netfive_500/compvis-word_nudity-method_noxattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_separated/netfive_500/" --prompts_path="data/netfive_500.csv" --save_path="evaluation_folder/compvis-word_nudity-method_noxattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_separated-netfive_500.csv"

# erasing Taylor Swift
# CUDA_VISIBLE_DEVICES=2 python train-esd.py --seperator "," --train_method "xattn" --ckpt_path "../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt" --diffusers_config_path "../Better_Erasing/models/erase/config.json" --prompt "Taylor Swift"  --config_path "configs/stable-diffusion/v1-inference.yaml" --info "none" | tee "logs/erase_TaylorSwift_target_train.log"
CUDA_VISIBLE_DEVICES=2 python eval-scripts/generate-images.py --models_path=models --model_name="compvis-word_taylorswift-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none" --prompts_path "data/netfive_500.csv" --save_path "evaluation_netfive_500" --num_samples 1 --from_case 0 --to_case -1 | tee -a "logs/erase_TaylorSwift_target_generate.log"
CUDA_VISIBLE_DEVICES=2 python eval-scripts/imageclassify.py --topk=10 --folder_path="evaluation_netfive_500/compvis-word_taylorswift-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none/netfive_500/" --prompts_path="data/netfive_500.csv" --save_path="evaluation_folder/compvis-word_taylorswift-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none-netfive_500.csv"

# erasing Van Gogh 

CUDA_VISIBLE_DEVICES=2 python train-esd.py --seperator "," --train_method "xattn" --ckpt_path "../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt" --diffusers_config_path "../Better_Erasing/models/erase/config.json" --prompt "Van Gogh"  --config_path "configs/stable-diffusion/v1-inference.yaml" --info "none" | tee "logs/erase_VanGogh_target_train.log"
CUDA_VISIBLE_DEVICES=2 python eval-scripts/generate-images.py --models_path=models --model_name="compvis-word_VanGogh-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none" --prompts_path "data/netfive_500.csv" --save_path "evaluation_netfive_500" --num_samples 1 --from_case 0 --to_case -1 | tee -a "logs/erase_VanGogh_target_generate.log"
CUDA_VISIBLE_DEVICES=2 python eval-scripts/imageclassify.py --topk=10 --folder_path="evaluation_netfive_500/compvis-word_VanGogh-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none/netfive_500/" --prompts_path="data/netfive_500.csv" --save_path="evaluation_folder/compvis-word_VanGogh-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none-netfive_500.csv"

# erasing gun 

CUDA_VISIBLE_DEVICES=2 python train-esd.py --seperator "," --train_method "xattn" --ckpt_path "../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt" --diffusers_config_path "../Better_Erasing/models/erase/config.json" --prompt "gun"  --config_path "configs/stable-diffusion/v1-inference.yaml" --info "none" | tee "logs/erase_gun_target_train.log"
CUDA_VISIBLE_DEVICES=2 python eval-scripts/generate-images.py --models_path=models --model_name="compvis-word_gun-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none" --prompts_path "data/netfive_500.csv" --save_path "evaluation_netfive_500" --num_samples 1 --from_case 0 --to_case -1 | tee -a "logs/erase_gun_target_generate.log"
CUDA_VISIBLE_DEVICES=2 python eval-scripts/imageclassify.py --topk=10 --folder_path="evaluation_netfive_500/compvis-word_gun-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none/netfive_500/" --prompts_path="data/netfive_500.csv" --save_path="evaluation_folder/compvis-word_gun-target_-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none-netfive_500.csv"