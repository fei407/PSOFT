#!/bin/bash
export task_name="commonsense15k"
export model_name_or_path="meta-llama/Llama-3.1-8B"
export output_dir="../results/$task_name/${model_name_or_path##*/}"
export peft_name="psoft"
export rank=424
export orth="True"
export mag_b="True"
export mag_a="True"
export use_neumann="True"
export neumann_n=5

current_output_dir="${output_dir}/H100_PSOFT_r${rank}_neumann_n${neumann_n}"

python ../fine-tuning_comm.py \
  --base_model $model_name_or_path \
  --data_path '../ft-training_set/commonsense_15k.json' \
  --output_dir $current_output_dir \
  --batch_size 64 \
  --micro_batch_size 2 \
  --num_epochs 3 \
  --learning_rate 1e-4 \
  --eval_step 80 \
  --save_step 80 \
  --cutoff_len 512 \
  --val_set_size 120 \
  --peft_name $peft_name \
  --peft_inserted_modules "q_proj","k_proj","v_proj","up_proj","down_proj" \
  --peft_dropout 0.0 \
  --peft_rank $rank \
  --psoft_orth $orth \
  --psoft_mag_b $mag_b \
  --psoft_mag_a $mag_a \
  --psoft_use_cayley_neumann $use_neumann \
  --psoft_num_cayley_neumann_terms $neumann_n
