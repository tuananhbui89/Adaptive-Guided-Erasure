# imagenette_v1
CUDA_VISIBLE_DEVICES=0 python data_preparation.py configs/object/imagenette_v1.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python training.py configs/object/imagenette_v1.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
          --prompts_path ./data/small_imagenet_prompts.csv \
          --save_path evaluation_folder/imagenette_v1/small_imagenet_prompts \
          --model_name saved_model/LoRA_fusion_model_imagenette_v1 \
          --step 1

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=3 python src/sample_images_from_csv.py \
          --prompts_path ./data/imagenette.csv \
          --save_path evaluation_massive/imagenette_v1/imagenette \
          --model_name saved_model/LoRA_fusion_model_imagenette_v1 \
          --step 1

python ../eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/imagenette_v1/imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/mace_imagenette_v1.csv' \


# imagenette_v2
CUDA_VISIBLE_DEVICES=0 python data_preparation.py configs/object/imagenette_v2.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python training.py configs/object/imagenette_v2.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
          --prompts_path ./data/small_imagenet_prompts.csv \
          --save_path evaluation_folder/imagenette_v2/small_imagenet_prompts \
          --model_name saved_model/LoRA_fusion_model_imagenette_v2 \
          --step 1

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
            --prompts_path ./data/imagenette.csv \
            --save_path evaluation_massive/imagenette_v2/imagenette \
            --model_name saved_model/LoRA_fusion_model_imagenette_v2 \
            --step 1

python ../eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/imagenette_v2/imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/mace_imagenette_v2.csv' \

# imagenette_v3
CUDA_VISIBLE_DEVICES=0 python data_preparation.py configs/object/imagenette_v3.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python training.py configs/object/imagenette_v3.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
          --prompts_path ./data/small_imagenet_prompts.csv \
          --save_path evaluation_folder/imagenette_v3/small_imagenet_prompts \
          --model_name saved_model/LoRA_fusion_model_imagenette_v3 \
          --step 1

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
            --prompts_path ./data/imagenette.csv \
            --save_path evaluation_massive/imagenette_v3/imagenette \
            --model_name saved_model/LoRA_fusion_model_imagenette_v3 \
            --step 1

python ../eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/imagenette_v3/imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/mace_imagenette_v3.csv' \

# imagenette_v4
CUDA_VISIBLE_DEVICES=0 python data_preparation.py configs/object/imagenette_v4.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python training.py configs/object/imagenette_v4.yaml

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
          --prompts_path ./data/small_imagenet_prompts.csv \
          --save_path evaluation_folder/imagenette_v4/small_imagenet_prompts \
          --model_name saved_model/LoRA_fusion_model_imagenette_v4 \
          --step 1

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python src/sample_images_from_csv.py \
            --prompts_path ./data/imagenette.csv \
            --save_path evaluation_massive/imagenette_v4/imagenette \
            --model_name saved_model/LoRA_fusion_model_imagenette_v4 \
            --step 1

python ../eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/imagenette_v4/imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/mace_imagenette_v4.csv' \

CUDA_VISIBLE_DEVICES=3 python src/sample_images_from_csv.py \
          --prompts_path ./data/coco_30k.csv \
          --save_path evaluation_massive/imagenette_v1/coco_30k \
          --model_name saved_model/LoRA_fusion_model_imagenette_v1 \
          --from_case 0 \
          --to_case -1 \
          --step 1 &> imagenette_v1_coco_30k_0.log &

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=2 python src/sample_images_from_csv.py \
          --prompts_path ./data/imagenette_synonyms_200.csv \
          --save_path evaluation_massive/imagenette_v1/imagenette_synonyms_200 \
          --model_name saved_model/LoRA_fusion_model_imagenette_v1 \
          --step 1 &> imagenette_v1_synonyms_200.log &

python ../eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/imagenette_v1/imagenette_synonyms_200/' \
    --prompts_path='data/imagenette_synonyms_200.csv' \
    --save_path='evaluation_folder/mace_imagenette_v1_imagenette_synonyms_200.csv'

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python src/sample_images_from_csv.py \
          --prompts_path ./data/imagenette_synonyms_200.csv \
          --save_path evaluation_massive/imagenette_v2/imagenette_synonyms_200 \
          --model_name saved_model/LoRA_fusion_model_imagenette_v2 \
          --step 1 &> imagenette_v2_synonyms_200.log &

python ../eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/imagenette_v2/imagenette_synonyms_200/' \
    --prompts_path='data/imagenette_synonyms_200.csv' \
    --save_path='evaluation_folder/mace_imagenette_v2_imagenette_synonyms_200.csv'

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python src/sample_images_from_csv.py \
          --prompts_path ./data/imagenette_synonyms_200.csv \
          --save_path evaluation_massive/imagenette_v3/imagenette_synonyms_200 \
          --model_name saved_model/LoRA_fusion_model_imagenette_v3 \
          --step 1 &> imagenette_v3_synonyms_200.log &

python ../eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/imagenette_v3/imagenette_synonyms_200/' \
    --prompts_path='data/imagenette_synonyms_200.csv' \
    --save_path='evaluation_folder/mace_imagenette_v3_imagenette_synonyms_200.csv'

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=3 python src/sample_images_from_csv.py \
          --prompts_path ./data/imagenette_synonyms_200.csv \
          --save_path evaluation_massive/imagenette_v4/imagenette_synonyms_200 \
          --model_name saved_model/LoRA_fusion_model_imagenette_v4 \
          --step 1 &> imagenette_v4_synonyms_200.log &

python ../eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/imagenette_v4/imagenette_synonyms_200/' \
    --prompts_path='data/imagenette_synonyms_200.csv' \
    --save_path='evaluation_folder/mace_imagenette_v4_imagenette_synonyms_200.csv'