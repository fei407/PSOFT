#!/bin/bash
python eval_gsm8k.py --model './results/metamath40k/Llama-3.2-3B/H100_PSOFT_r168_peft_lr-4e-4/' --data_file ./data/test/GSM8K_test.jsonl
python eval_math.py --model './results/metamath40k/Llama-3.2-3B/H100_PSFOT_r168_peft_lr-4e-4/' --data_file ./data/test/MATH_test.jsonl

