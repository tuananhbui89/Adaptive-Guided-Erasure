# KellyMckernan
CUDA_VISIBLE_DEVICES=0 python data_preparation.py configs/art/erase_KellyMckernan.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python training.py configs/art/erase_KellyMckernan.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
          --prompts_path ./data/short_niche_art_prompts.csv \
          --save_path evaluation_folder/art/KellyMckernan/small_imagenet_prompts \
          --model_name saved_model/LoRA_fusion_model_KellyMckernan \
          --step 1

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
          --prompts_path ./data/long_niche_art_prompts.csv \
          --save_path evaluation_massive/KellyMckernan/long_niche_art_prompts \
          --model_name saved_model/LoRA_fusion_model_KellyMckernan \
          --step 1 --num_samples 5 &> log_KellyMckernan_long.txt &

# KilianEng
CUDA_VISIBLE_DEVICES=0 python data_preparation.py configs/art/erase_KilianEng.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python training.py configs/art/erase_KilianEng.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
          --prompts_path ./data/short_niche_art_prompts.csv \
          --save_path evaluation_folder/art/KilianEng/small_imagenet_prompts \
          --model_name saved_model/LoRA_fusion_model_KilianEng \
          --step 1

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python src/sample_images_from_csv.py \
          --prompts_path ./data/long_niche_art_prompts.csv \
          --save_path evaluation_massive/KilianEng/long_niche_art_prompts \
          --model_name saved_model/LoRA_fusion_model_KilianEng \
          --step 1 --num_samples 5 &> log_KilianEng_long.txt &

# AjinDemiHuman
CUDA_VISIBLE_DEVICES=0 python data_preparation.py configs/art/erase_AjinDemiHuman.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python training.py configs/art/erase_AjinDemiHuman.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
          --prompts_path ./data/short_niche_art_prompts.csv \
          --save_path evaluation_folder/art/AjinDemiHuman/small_imagenet_prompts \
          --model_name saved_model/LoRA_fusion_model_AjinDemiHuman \
          --step 1

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=3 python src/sample_images_from_csv.py \
          --prompts_path ./data/long_niche_art_prompts.csv \
          --save_path evaluation_massive/AjinDemiHuman/long_niche_art_prompts \
          --model_name saved_model/LoRA_fusion_model_AjinDemiHuman \
          --step 1 --num_samples 5 &> log_AjinDemiHuman_long.txt &

# ThomasKinkade
CUDA_VISIBLE_DEVICES=0 python data_preparation.py configs/art/erase_ThomasKinkade.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python training.py configs/art/erase_ThomasKinkade.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
          --prompts_path ./data/short_niche_art_prompts.csv \
          --save_path evaluation_folder/art/ThomasKinkade/small_imagenet_prompts \
          --model_name saved_model/LoRA_fusion_model_ThomasKinkade \
          --step 1

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
          --prompts_path ./data/long_niche_art_prompts.csv \
          --save_path evaluation_massive/ThomasKinkade/long_niche_art_prompts \
          --model_name saved_model/LoRA_fusion_model_ThomasKinkade \
          --step 1 --num_samples 5 &> log_ThomasKinkade_long.txt &

# TylerEdlin
CUDA_VISIBLE_DEVICES=0 python data_preparation.py configs/art/erase_TylerEdlin.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python training.py configs/art/erase_TylerEdlin.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
          --prompts_path ./data/short_niche_art_prompts.csv \
          --save_path evaluation_folder/art/TylerEdlin/small_imagenet_prompts \
          --model_name saved_model/LoRA_fusion_model_TylerEdlin \
          --step 1

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python src/sample_images_from_csv.py \
          --prompts_path ./data/long_niche_art_prompts.csv \
          --save_path evaluation_massive/TylerEdlin/long_niche_art_prompts \
          --model_name saved_model/LoRA_fusion_model_TylerEdlin \
          --step 1 --num_samples 5 &> log_TylerEdlin_long.txt &


