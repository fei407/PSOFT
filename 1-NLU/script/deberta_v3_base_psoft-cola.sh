#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0

export task_name="cola"
export output_dir="../results/$task_name"
export max_seq_length=64
export model_name_or_path="microsoft/deberta-v3-base"
export per_device_train_batch_size=32
export per_device_eval_batch_size=32
export warmup_ratio=0.1
export peft_name="psoft"
export rank=46
export orth="True"
export mag_b="True"
export mag_a="True"
export use_neumann="False"
export neumann_n=5

current_output_dir="${output_dir}/RTX5090_PSOFT_r${rank}"

python ../fine-tuning_glue.py \
  --model_name_or_path $model_name_or_path \
  --task_name $task_name \
  --max_seq_length $max_seq_length \
  --do_train \
  --do_eval \
  --seed 42 \
  --output_dir $current_output_dir \
  --overwrite_output_dir \
  --logging_dir $current_output_dir/log \
  --num_train_epochs 20 \
  --per_device_train_batch_size $per_device_train_batch_size \
  --per_device_eval_batch_size $per_device_eval_batch_size \
  --warmup_ratio $warmup_ratio \
  --learning_rate 6e-4 \
  --logging_strategy epoch \
  --eval_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end True \
  --metric_for_best_model eval_matthews_correlation \
  --greater_is_better True \
  --save_total_limit 1 \
  --peft_name $peft_name \
  --peft_rank $rank \
  --psoft_orth $orth \
  --psoft_mag_b $mag_b \
  --psoft_mag_a $mag_a \
  --psoft_use_cayley_neumann $use_neumann \
  --psoft_num_cayley_neumann_terms $neumann_n
