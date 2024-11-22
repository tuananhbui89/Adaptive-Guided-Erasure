
CUDA_VISIBLE_DEVICES=2 python generate_images_ldm.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet' \
    --prompts_path 'data/imagenette_synonyms_200.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 &> logs/gumbel_imagenette_v1_wo_synonym.log &

CUDA_VISIBLE_DEVICES=2 python generate_images_ldm.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_imagenette_v2_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet' \
    --prompts_path 'data/imagenette_synonyms_200.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 &> logs/gumbel_imagenette_v2_wo_synonym.log &

CUDA_VISIBLE_DEVICES=0 python generate_images_ldm.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_imagenette_v3_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet' \
    --prompts_path 'data/imagenette_synonyms_200.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 &> logs/gumbel_imagenette_v3_wo_synonym.log &

CUDA_VISIBLE_DEVICES=0 python generate_images_ldm.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_imagenette_v4_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet' \
    --prompts_path 'data/imagenette_synonyms_200.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 &> logs/gumbel_imagenette_v4_wo_synonym.log &

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet/ldm-imagenette_synonyms_200/' \
    --prompts_path='data/imagenette_synonyms_200.csv' \
    --save_path='evaluation_folder/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet-ldm-imagenette_synonyms_200.csv'

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v2_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet/ldm-imagenette_synonyms_200/' \
    --prompts_path='data/imagenette_synonyms_200.csv' \
    --save_path='evaluation_folder/compvis-adversarial-gumbel-word_imagenette_v2_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet-ldm-imagenette_synonyms_200.csv'

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v3_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet/ldm-imagenette_synonyms_200/' \
    --prompts_path='data/imagenette_synonyms_200.csv' \
    --save_path='evaluation_folder/compvis-adversarial-gumbel-word_imagenette_v3_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet-ldm-imagenette_synonyms_200.csv'

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v4_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet/ldm-imagenette_synonyms_200/' \
    --prompts_path='data/imagenette_synonyms_200.csv' \
    --save_path='evaluation_folder/compvis-adversarial-gumbel-word_imagenette_v4_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_0.1_hard_1_topk_2_num_100_update_-1_timestep_0_multi_2_kclosest_100_vocab_Imagenet-ldm-imagenette_synonyms_200.csv'd
