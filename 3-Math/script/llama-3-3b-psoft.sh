#!/bin/bash
export task_name="metamath40k"
export model_name_or_path="meta-llama/Llama-3.2-3B"
export output_dir="../results/$task_name/${model_name_or_path##*/}"
export peft_name="psoft"
export rank=352
export orth="True"
export mag_b="True"
export mag_a="True"
export use_neumann="True"
export neumann_n=5

current_output_dir="${output_dir}/H100_PSOFT_r${rank}_neumann_n${neumann_n}"

python ../fine-tuning_math.py \
  --model_name_or_path $model_name_or_path \
  --data_path "meta-math/MetaMathQA-40K" \
  --data_length 10000000 \
  --bf16 True \
  --output_dir $current_output_dir \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 0 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --num_train_epochs 2 \
  --peft_name $peft_name \
  --peft_inserted_modules q_proj k_proj v_proj up_proj down_proj o_proj gate_proj \
  --peft_dropout 0.0 \
  --peft_rank $rank \
  --psoft_orth $orth \
  --psoft_mag_b $mag_b \
  --psoft_mag_a $mag_a \
  --psoft_use_cayley_neumann $use_neumann \
  --psoft_num_cayley_neumann_terms $neumann_n