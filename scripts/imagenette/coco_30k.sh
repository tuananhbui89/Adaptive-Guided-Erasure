MODEL_NAME='compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet'
MODEL_NAME='compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_EN3K' # DONE
CUDA_VISIBLE_DEVICES=1 python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name=$MODEL_NAME \
    --prompts_path 'data/coco_30k.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0 \
    --to_case 5000 &> logs/coco_30k_gumbel_from_0_to_5000.log &

CUDA_VISIBLE_DEVICES=1 python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name=$MODEL_NAME \
    --prompts_path 'data/coco_30k.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 5000 \
    --to_case 10000 &> logs/coco_30k_gumbel_from_5000_to_10000.log &

CUDA_VISIBLE_DEVICES=2 python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name=$MODEL_NAME \
    --prompts_path 'data/coco_30k.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 10000 \
    --to_case 15000 &> logs/coco_30k_gumbel_from_10000_to_15000.log &

CUDA_VISIBLE_DEVICES=2 python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name=$MODEL_NAME \
    --prompts_path 'data/coco_30k.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 15000 \
    --to_case 20000 &> logs/coco_30k_gumbel_from_15000_to_20000.log &

CUDA_VISIBLE_DEVICES=1 python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name=$MODEL_NAME \
    --prompts_path 'data/coco_30k.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 20000 \
    --to_case 25000 &> logs/coco_30k_gumbel_from_20000_to_25000.log &

CUDA_VISIBLE_DEVICES=1 python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name=$MODEL_NAME \
    --prompts_path 'data/coco_30k.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 25000 \
    --to_case 30000 &> logs/coco_30k_gumbel_from_25000_to_30000.log &
