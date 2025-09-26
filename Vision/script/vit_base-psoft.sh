#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export PYTHONHASHSEED=0
datasets=(
  'cifar'
  'caltech101'
  'dtd'
  'flowers102'
  'pets'
  'svhn'
  'sun397'
  'patch_camelyon'
  'eurosat'
  'resisc45'
  'retinopathy'
  'clevr_count'
  'clevr_dist'
  'dmlab'
  'kitti_dist'
  'dsprites_loc'
  'dsprites_ori'
  'smallnorb_azi'
  'smallnorb_ele'
)
export peft_name="psoft"
export output_dir="../results/$peft_name"
export rank=46
export orth="True"
export mag_b="True"
export mag_a="True"
export use_neumann="True"
export neumann_n=5

for dataset_name in "${datasets[@]}"
do
  current_output_dir="${output_dir}/RTX4090_${dataset_name}_PSOFT_r${rank}_neumann_n${neumann_n}"

  python ../fine-tuning_vision.py \
  --dataset_name $dataset_name \
  --model_name vit-base \
  --output_dir $current_output_dir \
  --num_train_epochs 50 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 256 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --gradient_accumulation_steps 1 \
  --logging_steps 10 \
  --load_best_model_at_end True \
  --save_total_limit 1 \
  --metric_for_best_model eval_accuracy \
  --label_names labels \
  --remove_unused_columns False \
  --seed 42 \
  --learning_rate 5e-4 \
  --cls_learning_rate 5e-3 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --weight_decay 1e-3 \
  --peft_name $peft_name \
  --peft_dropout 1e-1 \
  --peft_rank $rank \
  --psoft_orth $orth \
  --psoft_mag_b $mag_b \
  --psoft_mag_a $mag_a \
  --psoft_use_cayley_neumann $use_neumann \
  --psoft_num_cayley_neumann_terms $neumann_n
done
