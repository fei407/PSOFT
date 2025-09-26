import os
import sys
from typing import List

import fire
import torch
import argparse
import transformers
import yaml
from datasets import load_dataset
from typing import List, Optional, Union
import wandb

from tqdm import tqdm
import sys
from functools import partial, reduce

sys.path.append("../")
from baselines.lora_xs.initialization_utils import find_and_initialize
from baselines.svft.svft_layers import LinearWithSVFT, create_and_replace_modules, get_target_modules_list, replace_svft_with_fused_linear
from baselines.psoft.psoft_layers import create_and_insert

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))

from peft import (  # noqa: E402
    LoraConfig, OFTConfig, BOFTConfig, VeraConfig,
    PromptLearningConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, LlamaTokenizer

def check_lora_A_row_orthogonality(model, tol=1e-1):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            for adapter_name, A_layer in module.lora_A.items():
                A = A_layer.weight.data
                AA_t = A @ A.T
                identity = torch.eye(A.shape[0], device=A.device)
                deviation = torch.norm(AA_t - identity)

                print(f"[{name}] Adapter: {adapter_name} | ‖A·Aᵀ - I‖ = {deviation:.4e}")
                if deviation < tol:
                    print(" --> A is approximately row-orthogonal")
                else:
                    print(" --> A is NOT row-orthogonal")

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        load_8bit: bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # peft hyperparams
        peft_name: str = "lora",
        peft_rank: int = None,
        lora_alpha: int = 8,
        peft_dropout: float = 0.0,
        peft_inserted_modules: List[str] = None,
        boft_b: int = 2,
        boft_m: int = 2,
        svft_off_diag: int = 0,
        svft_pattern: str = "banded",
        svft_fill_orthonormal: bool = False,
        psoft_orth: bool = True,
        psoft_mag_out: bool = False,
        psoft_mag_b: bool = True,
        psoft_mag_a: bool = True,
        psoft_use_cayley_neumann: bool = True,
        psoft_num_cayley_neumann_terms: int = 5,
        goft_strict_oft: bool = True,
        goft_no_scaling: bool = True,
        oft_block_size: int = 32,
        oft_use_cayley_neumann: bool = True,
        oft_num_cayley_neumann_terms: int = 5,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"peft_name: {peft_name}\n"
        f"peft_rank: {peft_rank}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"peft_dropout: {peft_dropout}\n"
        f"peft_inserted_modules: {peft_inserted_modules}\n"
        f"boft_b: {boft_b}\n"
        f"boft_m: {boft_m}\n"
        f"svft_off_diag: {svft_off_diag}\n"
        f"svft_pattern: {svft_pattern}\n"
        f"svft_fill_orthonormal: {svft_fill_orthonormal}\n"
        f"psoft_orth: {psoft_orth}\n"
        f"psoft_mag_out: {psoft_mag_out}\n"
        f"psoft_mag_b: {psoft_mag_b}\n"
        f"psoft_mag_a: {psoft_mag_a}\n"
        f"psoft_use_cayley_neumann: {psoft_use_cayley_neumann}\n"
        f"psoft_num_cayley_neumann_terms: {psoft_num_cayley_neumann_terms}\n"
        f"goft_strict_oft: {goft_strict_oft}\n"
        f"goft_no_scaling: {goft_no_scaling}\n"
        f"oft_block_size: {oft_block_size}\n"
        f"oft_use_cayley_neumann: {oft_use_cayley_neumann}\n"
        f"oft_num_cayley_neumann_terms: {oft_num_cayley_neumann_terms}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )

    print(base_model)

    # assert (
    #     base_model
    # ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = "CommonsenseReasoning"
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = "all"
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = False

    load_8bit = bool(load_8bit) if isinstance(load_8bit, (bool, int)) else str(load_8bit).lower() == 'true'
    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

    print(f"model.config.model_type: {model.config.model_type}")
    if model.config.model_type == "llama":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        # need to handle llama 3 separately
        if "Llama-3" in base_model:
            print("load llama-3 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    if peft_name == "lora":
        peft_config = LoraConfig(
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type="CAUSAL_LM",
        )
    elif peft_name == "pissa":
        peft_config = LoraConfig(
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type="CAUSAL_LM",
            init_lora_weights = 'pissa',
            # init_lora_weights = 'pissa_niter_20',  # Using Fast-SVD，'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA initialization
        )
        print("PiSSA is Baking... (PiSSA initializing will take a while.)")
    elif peft_name == "dora":
        peft_config = LoraConfig(
            use_dora=True,
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type="CAUSAL_LM",
        )
    elif peft_name == "vera":
        peft_config = VeraConfig(
            r=peft_rank,
            vera_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type="CAUSAL_LM",
        )
    elif peft_name == "oft":
        peft_config = OFTConfig(
            oft_block_size=oft_block_size,
            use_cayley_neumann=oft_use_cayley_neumann,
            num_cayley_neumann_terms=oft_num_cayley_neumann_terms,
            module_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type="CAUSAL_LM",
        )
    elif peft_name == "boft":
        peft_config = BOFTConfig(
            boft_block_size=boft_b,
            boft_n_butterfly_factor=boft_m,
            boft_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type="CAUSAL_LM",
        )
    elif peft_name == 'svft':
        # for SVFT turn off gradient requirement for all layers
        # PEFT library handles this internally
        for param in model.parameters():
            param.requires_grad = False

        print(f"Target Modules: {peft_inserted_modules}")
        assign_svft_layer = partial(LinearWithSVFT,
                                    off_diag=svft_off_diag,
                                    pattern=svft_pattern,
                                    fill_orthonormal=svft_fill_orthonormal)

        create_and_replace_modules(model, get_target_modules_list(model, peft_inserted_modules), assign_svft_layer)
    elif peft_name == 'lora_xs':
        config = LoraConfig(
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type="CAUSAL_LM",
        )
        adapter_name = "default"
        peft_config_dict = {}
        if not isinstance(config, PromptLearningConfig):
            peft_config_dict[adapter_name] = config

        with open("../../baselines/lora_xs/config/reconstruct_config.yaml", "r") as stream:
            reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
        reconstr_type = reconstr_config["reconstruction_type"]
        reconstr_config[reconstr_type]["rank"] = peft_config_dict[adapter_name].r

        model = get_peft_model(model, config)

        find_and_initialize(
            model,
            peft_config_dict,
            adapter_name=adapter_name,
            reconstr_type=reconstr_type,
            reconstruct_config=reconstr_config,
        )
    elif peft_name == 'psoft':
        config = LoraConfig(
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type="CAUSAL_LM",
            init_lora_weights='pissa_orth',
            # init_lora_weights = 'pissa_niter_20',  # Using Fast-SVD，'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA initialization
        )
        print("PiSSA is Baking... (PiSSA initializing will take a while.)")
        model = get_peft_model(model, config)
        create_and_insert(model, config, psoft_orth, psoft_mag_out, psoft_mag_b, psoft_mag_a, psoft_use_cayley_neumann, psoft_num_cayley_neumann_terms)

        check_lora_A_row_orthogonality(model)
    elif peft_name == "full":
        pass
    else:
        raise ValueError("Unknown peft method")

    if peft_name not in ['full', 'svft', 'lora_xs', 'psoft', 'psoft-um']:
        model = get_peft_model(model, peft_config)

    # To make tensors contiguous for lora_xs and psoft.
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    if peft_name == "prefix-tuning":
        model.to('cuda')

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Output Dir: {output_dir}")

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 and peft_name not in ["boft", "vera"] else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    # if peft_name in ['full']:
    #     model = model.bfloat16()

    def log_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        non_classifier_trainable_params = 0
        all_param = 0

        trainable_param_details = []

        for name, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            count_params = True

            if count_params:
                all_param += num_params
                if param.requires_grad:
                    print(f"Parameter: {name}, Shape: {param.shape}")
                    trainable_params += num_params
                    trainable_param_details.append((name, num_params))
                    if "classifier" not in name and "pooler" not in name:
                        non_classifier_trainable_params += num_params
        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || "
            f"trainable%: {100 * trainable_params / all_param:.4f} \n"
        )

    log_trainable_parameters(model)

    # print("Exiting the program after logging trainable parameters.")
    # sys.exit("Program terminated intentionally after logging trainable parameters.")

    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    peak_memory_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
    print(f"Peak GPU memory allocated during training: {peak_memory_allocated:.2f} GB")
    wandb.log({'peak_gpu_memory_GB': peak_memory_allocated})

    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    print("model merging...")
    if peft_name == 'svft':
        replace_svft_with_fused_linear(model, get_target_modules_list(model, peft_inserted_modules))
    elif peft_name == "full":
        pass
    else:
        model = model.merge_and_unload()

    for param in model.parameters():
        param.data = param.data.contiguous()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')

    # model/data params
    parser.add_argument('--base_model', type=str, required=True, help='Base model')
    parser.add_argument('--data_path', type=str, default='yahma/alpaca-cleaned', help='Data path')
    parser.add_argument('--output_dir', type=str, default='./lora-alpaca', help='Output directory')
    parser.add_argument('--load_8bit', action='store_true', help='Load 8-bit')

    # training hyperparams
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='Micro batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='Cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='Validation set size')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('--eval_step', type=int, default=200, help='Evaluation step')
    parser.add_argument('--save_step', type=int, default=200, help='Save step')

    # peft hyperparams
    parser.add_argument('--peft_name', type=str, default='lora', help='PEFT name')
    parser.add_argument('--peft_rank', type=int, default=16, help='PEFT rank')
    parser.add_argument('--lora_alpha', type=int, default=8, help='Lora alpha')
    parser.add_argument('--peft_dropout', type=float, default=0.0, help='PEFT dropout')
    parser.add_argument('--peft_inserted_modules', nargs='+', help='PEFT inserted modules')
    parser.add_argument('--boft_b', type=int, default=2, help='BOFT block')
    parser.add_argument('--boft_m', type=int, default=2, help='BOFT factor')
    parser.add_argument('--svft_off_diag', type=int, default=0, help='SVFT off_diag')
    parser.add_argument('--svft_pattern', type=str, default='banded', help='SVFT banded')
    parser.add_argument('--svft_fill_orthonormal', type=bool, default=False, help='SVFT fill_orthonormal')
    parser.add_argument('--psoft_orth', type=bool, default=True, help='PSOFT orth')
    parser.add_argument('--psoft_mag_out', type=bool, default=False, help='PSOFT mag_out')
    parser.add_argument('--psoft_mag_b', type=bool, default=True, help='PSOFT mag_b')
    parser.add_argument('--psoft_mag_a', type=bool, default=True, help='PSOFT mag_a')
    parser.add_argument('--psoft_use_cayley_neumann', type=bool, default=True, help='PSOFT psoft_use_cayley_neumann')
    parser.add_argument('--psoft_num_cayley_neumann_terms', type=int, default=5, help='PSOFT psoft_num_cayley_neumann_terms')
    parser.add_argument('--goft_strict_oft', type=bool, default=True, help='GOFT goft_strict_oft')
    parser.add_argument('--goft_no_scaling', type=bool, default=True, help='GOFT goft_no_scaling')
    parser.add_argument('--oft_block_size', type=int, default=32, help='OFT oft_block_size')
    parser.add_argument('--oft_use_cayley_neumann', type=bool, default=True, help='OFT use_cayley_neumann')
    parser.add_argument('--oft_num_cayley_neumann_terms', type=int, default=5, help='OFT num_cayley_neumann_terms')

    # bottleneck adapter hyperparams
    parser.add_argument('--bottleneck_size', type=int, default=256, help='Bottleneck size')
    parser.add_argument('--non_linearity', type=str, default='tanh', help='Non-linearity')
    parser.add_argument('--adapter_dropout', type=float, default=0.0, help='Adapter dropout')
    parser.add_argument('--use_parallel_adapter', action='store_true', help='Use parallel adapter')
    parser.add_argument('--use_adapterp', action='store_true', help='Use adapterp')
    parser.add_argument('--target_modules', nargs='+', help='Target modules')
    parser.add_argument('--scaling', type=Union[float, str], default=1.0, help='Scaling')

    # prefix tuning hyperparams
    parser.add_argument('--num_virtual_tokens', type=int, default=30, help='Number of virtual tokens')

    # llm hyperparams
    parser.add_argument('--train_on_inputs', action='store_true', help='Train on inputs')
    parser.add_argument('--group_by_length', action='store_true', help='Group by length')

    # wandb params
    parser.add_argument('--wandb_project', type=str, default='', help='Wandb project')
    parser.add_argument('--wandb_run_name', type=str, default='', help='Wandb run name')
    parser.add_argument('--wandb_watch', type=str, default='', help='Wandb watch')
    parser.add_argument('--wandb_log_model', type=str, default='', help='Wandb log model')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Resume from checkpoint')

    return parser.parse_args()


if __name__ == "__main__":
    fire.Fire(train)

    # args = parse_args()
    # train(**vars(args))