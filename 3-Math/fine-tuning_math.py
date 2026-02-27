#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#    Modified by Zheng Yuan and Hongyi Yuan

#    Adapting the above code-base.

import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import io
import torch
import transformers
import yaml
from torch.utils.data import Dataset
from transformers import Trainer
import argparse
import json
import random

from peft import get_peft_model, LoraConfig, VeraConfig, OFTConfig, BOFTConfig, PromptLearningConfig, PsoftConfig
from tqdm import tqdm
from functools import partial, reduce

import sys
sys.path.append("../")

from datasets import load_dataset

import wandb

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
#### 28
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class PEFTArguments:
    """
    Arguments about PEFT configurations including:LoRA/DoRA/VeRA/OFT/BOFT/LoRA-XS/SVFT/PSOFT
    """
    peft_name: str = field(
        metadata={
            "help": "Specific PEFT methods including LoRA/DoRA/VeRA/OFT/BOFT/LoRA-XS/SVFT/PSOFT:"
        }
    )
    peft_rank: int = field(
        default=16,
        metadata={
            "help": "The rank (r) to be used for PEFT."
        }
    )
    lora_alpha: Optional[float] = field(
        default=8,
        metadata={"help": "multiplier (alpha) used for LoRA."}
    )
    peft_inserted_modules: Optional[List[str]] = field(
        # default_factory=lambda: ["query_proj", "key_proj", "value_proj", "attention.output.dense", "intermediate.dense", "output.dense"],
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # default_factory=lambda: ["query_proj"],
        metadata={"help": "The modules applying LoRA: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"}
    )
    peft_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "PEFT modules' dropout"}
    )
    boft_b: Optional[int] = field(
        default=2,
        metadata={"help": "Block size of the BOFT method"}
    )

    boft_m: Optional[int] = field(
        default=2,
        metadata={"help": "Number of sparse matrix multiplications"}
    )
    svft_off_diag: Optional[int] = field(
        default=0,
        metadata={"help": "Total off-diagonals to be used to populate matrix M (as referred in main paper)"}
    )

    svft_pattern: Optional[str] = field(
        default="banded",
        metadata={"help": "Choices: 'banded', 'random', 'top_k'. Using 'banded' with off_diag=1 simulates SVFT-plain"}
    )

    svft_fill_orthonormal: Optional[bool] = field(
        default=False,
        metadata={"help": "To determine if random orthonormal basis should be used"}
    )
    psoft_orth: Optional[bool] = field(
        default=True,
        metadata={"help": "Set this to use Cayley Parameterization on R"}
    )
    psoft_mag_b: Optional[bool] = field(
        default=True,
        metadata={"help": "Set this to tune scaling vector Beta for output of R"}
    )
    psoft_mag_a: Optional[bool] = field(
        default=True,
        metadata={"help": "Set this to tune scaling vector alpha for input of R"}
    )
    goft_strict_oft: Optional[bool] = field(
        default=True,
        metadata={"help": "Set this to True if the layer is strict orthogonal"}
    )
    goft_no_scaling: Optional[bool] = field(
        default=True,
        metadata={"help": "Set this to True if you don't want to fine tune the length"}
    )
    oft_block_size: Optional[int] = field(
        default=32,
        metadata={"help": "OFT block size across different layers"}
    )
    oft_use_cayley_neumann: Optional[bool] = field(
        default=True,
        metadata= {"help": "Whether to use the Cayley-Neumann Formulation of OFT or not. Set to True to improve computational efficiency but comes at costs of bigger approximation error for orthogonality."}
    )
    oft_num_cayley_neumann_terms: Optional[int] = field(
        default=5,
        metadata={"help": "Number of Cayley-Neumann terms to use. Higher number results in less approximation error for orthogonality."}
    )
    psoft_use_cayley_neumann: Optional[bool] = field(
        default=True,
        metadata= {"help": "Whether to use the Cayley-Neumann Formulation of OFT or not. Set to True to improve computational efficiency but comes at costs of bigger approximation error for orthogonality."}
    )
    psoft_num_cayley_neumann_terms: Optional[int] = field(
        default=5,
        metadata={"help": "Number of Cayley-Neumann terms to use. Higher number results in less approximation error for orthogonality."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=True)


def check_lora_A_row_orthogonality(model, tol=1e-1):
    for name, module in model.named_modules():
        # Only check PSOFT layers that have the cache getter and psoft_R
        if not (hasattr(module, "_get_psoft_ab_cache_buffers") and hasattr(module, "psoft_R")):
            continue

        for adapter_name in module.psoft_R.keys():
            # Read cached A, B
            try:
                A, _ = module._get_psoft_ab_cache_buffers(adapter_name)
            except Exception as e:
                if verbose:
                    print(f"[{name}] Adapter: {adapter_name} | cannot get cache buffers: {e}")
                continue

            # Compute deviation || A A^T - I ||
            AA_t = A @ A.T
            identity = torch.eye(A.shape[0], device=A.device, dtype=AA_t.dtype)
            deviation = torch.linalg.norm(AA_t - identity)

            print(f"[{name}] Adapter: {adapter_name} | ‖A·Aᵀ - I‖ = {deviation:.4e}")
            if deviation < tol:
                print(" --> A is approximately row-orthonormal")
            else:
                print(" --> A is NOT row-orthonormal")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data from Hugging Face datasets...")

        dataset = load_dataset(data_args.data_path)

        queries = dataset['train']['query'][:data_args.data_length]
        responses = dataset['train']['response'][:data_args.data_length]

        def get_input(query):
            if '\n' not in query:
                return ''
            return '\n'.join(query.split('\n')[1:])

        list_data_dict = [{'instruction': query.split('\n')[0],
                           'input': get_input(query),
                           'output': response}
                          for query, response in zip(queries, responses)]

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, PEFTArguments))
    model_args, data_args, training_args, peft_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    data_args.data_length = int(remaining_args[1])

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    ).to("cuda")

    # if peft_args.peft_name in ['full']:
    #     model = model.to(dtype=torch.bfloat16)

    print(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    # inspect token in Llama-3
    print("BOS token:", tokenizer.bos_token)
    print("EOS token:", tokenizer.eos_token)
    print("UNK token:", tokenizer.unk_token)
    if "llama-2" in model_args.model_name_or_path.lower():
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # peft logic
    task_type = "CAUSAL_LM"
    peft_name = peft_args.peft_name
    peft_rank = peft_args.peft_rank
    peft_dropout = peft_args.peft_dropout
    peft_inserted_modules = peft_args.peft_inserted_modules
    boft_b = peft_args.boft_b
    boft_m = peft_args.boft_m
    svft_off_diag = peft_args.svft_off_diag
    svft_pattern = peft_args.svft_pattern
    svft_fill_orthonormal = peft_args.svft_fill_orthonormal
    psoft_orth = peft_args.psoft_orth
    psoft_mag_b = peft_args.psoft_mag_b
    psoft_mag_a = peft_args.psoft_mag_a
    goft_strict_oft = peft_args.goft_strict_oft
    goft_no_scaling = peft_args.goft_no_scaling
    oft_block_size = peft_args.oft_block_size
    oft_use_cayley_neumann = peft_args.oft_use_cayley_neumann
    oft_num_cayley_neumann_terms = peft_args.oft_num_cayley_neumann_terms
    psoft_use_cayley_neumann = peft_args.psoft_use_cayley_neumann
    psoft_num_cayley_neumann_terms = peft_args.psoft_num_cayley_neumann_terms

    if peft_name == 'lora':
        peft_config = LoraConfig(
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type=task_type,
        )
    elif peft_name == "pissa":
        peft_config = LoraConfig(
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type=task_type,
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
            task_type=task_type,
        )
    elif peft_name == 'vera':
        peft_config = VeraConfig(
            r=peft_rank,
            vera_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type=task_type,
        )
    elif peft_name == "oft":
        peft_config = OFTConfig(
            oft_block_size=oft_block_size,
            use_cayley_neumann=oft_use_cayley_neumann,
            num_cayley_neumann_terms=oft_num_cayley_neumann_terms,
            module_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type=task_type,
        )
    elif peft_name == 'boft':
        peft_config = BOFTConfig(
            boft_block_size=boft_b,
            boft_n_butterfly_factor=boft_m,
            boft_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type=task_type,
        )
    elif peft_name == 'psoft':
        peft_config = PsoftConfig(
            r=peft_rank,
            psoft_alpha=peft_rank,
            psoft_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type=task_type,
            ab_svd_init="psoft_init",
            psoft_svd="lowrank",
            psoft_orth=psoft_orth,
            psoft_mag_a=psoft_mag_a,
            psoft_mag_b=psoft_mag_b,
            use_cayley_neumann=psoft_use_cayley_neumann,
            num_cayley_neumann_terms=psoft_num_cayley_neumann_terms,
            cayley_neumann_eps=None,
        )
        model = get_peft_model(model, peft_config)

        # check_lora_A_row_orthogonality(model)
    elif peft_name == 'full':
        pass
    else:
        raise ValueError("Unknown peft method")

    if peft_name not in ['full', 'svft', 'lora_xs', 'psoft', 'psoft-um']:
        model = get_peft_model(model, peft_config)

    # To make tensors contiguous for lora_xs and psoft.
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

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
                    print(f"Parameter: {name}, Shape: {param.shape}, Dtype: {param.dtype}")
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

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # print("Exiting the program after logging trainable parameters.")
    # sys.exit("Program terminated intentionally after logging trainable parameters.")

    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak_memory_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
    print(f"Peak GPU memory allocated during training: {peak_memory_allocated:.2f} GB")
    wandb.log({'peak_gpu_memory_GB': peak_memory_allocated})

    trainer.save_state()
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
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
