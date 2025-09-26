##!/usr/bin/env python
# coding=utf-8
# Code based on the HuggingFace transformers repository.
""" Finetuning the library models for sequence classification on GLUE."""
import datetime
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional,List
import math
import wandb

import datasets
import evaluate
import numpy as np
import torch
import yaml
from datasets import load_dataset
from torch import optim
import torch.profiler

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from peft import (
    get_peft_model,
    LoraConfig,
    PromptLearningConfig, BOFTConfig, VeraConfig, OFTConfig
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import send_example_telemetry
from baselines.lora_xs.initialization_utils import find_and_initialize
from baselines.psoft.psoft_layers import create_and_insert

from functools import partial

import safetensors.torch

# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
from transformers.utils.versions import require_version

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    cls_learning_rate: Optional[float] = field(
        default=5e-4,
        metadata={"help": "LR for classifier."}
    )
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
        default_factory=lambda: ["query_proj", "key_proj", "value_proj", "attention.output.dense", "intermediate.dense", "output.dense"],
        # default_factory=lambda: ["query_proj"],
        metadata={"help": "The modules applying LoRA: query_proj, key_proj, value_proj, attention.output.dense, intermediate.dense, output.dense"}
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
    psoft_mag_out: Optional[bool] = field(
        default=False,
        metadata={"help": "Set this to tune magnitude vector for output of W"}
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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (DataTrainingArguments, ModelArguments, PEFTArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, model_args, peft_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        data_args, model_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    torch.use_deterministic_algorithms(True)
    logger.info("use_deterministic_algorithms: " + str(torch.are_deterministic_algorithms_enabled()))

    print(peft_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    ### added PEFT logic
    task_type = "SEQ_CLS"
    peft_name = peft_args.peft_name
    peft_rank = peft_args.peft_rank
    peft_dropout = peft_args.peft_dropout
    peft_inserted_modules = peft_args.peft_inserted_modules
    boft_b = peft_args.boft_b
    boft_m = peft_args.boft_m
    # svft_off_diag = peft_args.svft_off_diag
    # svft_pattern = peft_args.svft_pattern
    # svft_fill_orthonormal = peft_args.svft_fill_orthonormal
    psoft_orth = peft_args.psoft_orth
    psoft_mag_out = peft_args.psoft_mag_out
    psoft_mag_b = peft_args.psoft_mag_b
    psoft_mag_a = peft_args.psoft_mag_a
    goft_strict_oft = peft_args.goft_strict_oft
    goft_no_scaling = peft_args.goft_no_scaling
    oft_block_size = peft_args.oft_block_size
    oft_use_cayley_neumann = peft_args.oft_use_cayley_neumann
    oft_num_cayley_neumann_terms = peft_args.oft_num_cayley_neumann_terms
    psoft_use_cayley_neumann = peft_args.psoft_use_cayley_neumann
    psoft_num_cayley_neumann_terms = peft_args.psoft_num_cayley_neumann_terms

    if peft_name == "lora":
        peft_config = LoraConfig(
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type=task_type,
            modules_to_save=["classifier","pooler"],
        )
    elif peft_name == "pissa":
        peft_config = LoraConfig(
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type=task_type,
            modules_to_save=["classifier", "pooler"],
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
            modules_to_save=["classifier","pooler"],
        )
    elif peft_name == "vera":
        peft_config = VeraConfig(
            r=peft_rank,
            vera_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            modules_to_save=["classifier","pooler"],
        )
    elif peft_name == "oft":
        peft_config = OFTConfig(
            oft_block_size=oft_block_size,
            use_cayley_neumann=oft_use_cayley_neumann,
            num_cayley_neumann_terms=oft_num_cayley_neumann_terms,
            module_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            modules_to_save=["classifier","pooler"],
        )
    elif peft_name == "boft":
        peft_config = BOFTConfig(
            boft_block_size=boft_b,
            boft_n_butterfly_factor=boft_m,
            boft_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            modules_to_save=["classifier","pooler"],
        )
    elif peft_name == 'lora_xs':
        config = LoraConfig(
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            task_type=task_type,
            modules_to_save=["classifier","pooler"],
        )
        adapter_name = "default"
        peft_config_dict = {}
        if not isinstance(config, PromptLearningConfig):
            peft_config_dict[adapter_name] = config

        with open("../baselines/lora_xs/config/reconstruct_config.yaml", "r") as stream:
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
            task_type=task_type,
            modules_to_save=["classifier","pooler"],
            init_lora_weights='pissa_orth',
            # init_lora_weights = 'pissa_niter_20',  # Using Fast-SVD，'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA initialization
        )
        print("PiSSA is Baking... (PiSSA initializing will take a while.)")
        model = get_peft_model(model, config)
        create_and_insert(model, config, psoft_orth, psoft_mag_out, psoft_mag_b, psoft_mag_a, psoft_use_cayley_neumann, psoft_num_cayley_neumann_terms)

        check_lora_A_row_orthogonality(model)
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
            # print(f"Parameter: {name}, Shape: {param.shape}")
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
            f"non-cls trainable params: {non_classifier_trainable_params:,} || "
            f"all params: {all_param:,} || "
            f"non-cls trainable%: {100 * non_classifier_trainable_params / all_param:.4f}"
        )

    log_trainable_parameters(model)

    # print("Exiting the program after logging trainable parameters.")
    # sys.exit("Program terminated intentionally after logging trainable parameters.")

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

        # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")

        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]

        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

        indices = np.arange(len(eval_dataset))
        np.random.shuffle(indices)

        half = len(indices) // 2
        idx_1 = indices[:half]
        idx_2 = indices[half:]

        eval_dataset_1 = eval_dataset.select(idx_1.tolist())
        eval_dataset_2 = eval_dataset.select(idx_2.tolist())

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    elif is_regression:
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")


    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    def get_classifier_modules(model_name):
        if model_name in {"microsoft/deberta-v3-base", "microsoft/deberta-v3-large"}:
            return [
                "pooler",
                "classifier",
            ]
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    # Setup Trainer
    peft_group = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
        and all(cls_name not in n for cls_name in get_classifier_modules(model_args.model_name_or_path))
    ]
    classifier_group = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
        and any(cls_name in n for cls_name in get_classifier_modules(model_args.model_name_or_path))
    ]
    optimizer = optim.AdamW(
        [
            {
                "params": peft_group,
                "lr": training_args.learning_rate,
            },
            {
                "params": classifier_group,
                "lr": model_args.cls_learning_rate,
            },
        ],
        weight_decay=training_args.weight_decay,
    )

    num_train_steps = math.ceil(
        len(train_dataset) / training_args.per_device_train_batch_size
    ) * training_args.num_train_epochs

    if training_args.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * training_args.warmup_ratio),
            num_training_steps=num_train_steps,
        )
    elif training_args.lr_scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * training_args.warmup_ratio),
            num_training_steps=num_train_steps,
        )
    print(optimizer)

    print(training_args)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset_1 if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        torch.cuda.reset_peak_memory_stats()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        peak_memory_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
        print(f"Peak GPU memory allocated during training: {peak_memory_allocated:.2f} GB")
        wandb.log({'peak_gpu_memory_GB': peak_memory_allocated})

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        # trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        trainer.state.epoch = training_args.num_train_epochs + 1
        logger.info("*** Evaluate ***")
        logger.info(f"*** best model checkpoint: {trainer.state.best_model_checkpoint} ***")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset_2]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            # trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    # print("model merging...")
    # if peft_name == 'svft':
    #     replace_svft_with_fused_linear(model, get_target_modules_list(model, peft_inserted_modules))
    # elif peft_name == "full" or peft_name == "head":
    #     pass
    # else:
    #     model = model.merge_and_unload()
    # model.save_pretrained(training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()










