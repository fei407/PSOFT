import json
import math
import os
import os.path
import random
import sys
import yaml
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from pprint import pprint
from typing import List, Literal, Optional

import evaluate
import numpy as np
import torch
from torch import optim
from datasets import load_dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import transformers
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import BOFTConfig, LoraConfig, OFTConfig, PromptLearningConfig, VeraConfig, get_peft_model
from wandb import require
import wandb

from baselines.lora_xs.initialization_utils import find_and_initialize
from baselines.svft.svft_layers import (
    LinearWithSVFT,
    create_and_replace_modules,
    get_target_modules_list,
    replace_svft_with_fused_linear,
)
from baselines.psoft.psoft_layers import create_and_insert

@dataclass
class ScriptArguments:
    results_json: str = field(default="results.json", metadata={"help": "Results json file"})
    model_name: Literal["dino-v2-large", "vit-base", "vit-large"] = field(default="vit-base", metadata={"help": "Model name"})
    dataset_name: Literal[
        'cifar',
        'caltech101',
        'dtd',
        'flowers102',
        'pets',
        'svhn',
        'sun397',
        'patch_camelyon',
        'eurosat',
        'resisc45',
        'retinopathy',
        'clevr_count',
        'clevr_dist',
        'dmlab',
        'kitti_dist',
        'dsprites_loc',
        'dsprites_ori',
        'smallnorb_azi',
        'smallnorb_ele',
    ] = field(default="cifar", metadata={"help": "Dataset name"})

    cls_learning_rate: float = field(
        default=5e-3, metadata={"help": "Classifier learning rate"}
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
        default_factory=lambda: ["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"],
        # default_factory=lambda: ["query_proj"],
        metadata={
            "help": "The modules applying LoRA: query, key, value, attention.output.dense, intermediate.dense, output.dense"}
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
        metadata={
            "help": "Choices: 'banded', 'random', 'top_k'. Using 'banded' with off_diag=1 simulates SVFT-plain"}
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

def check_lora_A_row_orthogonality(model, tol=1e-3):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            for adapter_name, A_layer in module.lora_A.items():
                A = A_layer.weight.data
                AA_t = A @ A.T
                identity = torch.eye(A.shape[0], device=A.device)
                deviation = torch.norm(AA_t - identity)

                print(f"[{name}] Adapter: {adapter_name} | ‖A·Aᵀ - I‖ = {deviation:.4e}")
                if deviation < tol:
                    print(" --> A is approximately orthogonal")
                else:
                    print(" --> A is NOT orthogonal")

##########################
# Metrics
##########################
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

##########################
# Utils
##########################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_trainable_params_dict(model):
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cls_trainable_p = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and "classifier" in n
    )
    peft_trainable_p = trainable_p - cls_trainable_p
    return {
        "total_p": total_p,
        "trainable_p": trainable_p,
        "cls_trainable_p": cls_trainable_p,
        "peft_trainable_p": peft_trainable_p,
    }

def print_trainable_parameters(model):
    params_dict = get_trainable_params_dict(model)
    total_p = params_dict["total_p"]
    trainable_p = params_dict["trainable_p"]
    cls_trainable_p = params_dict["cls_trainable_p"]
    peft_trainable_p = params_dict["peft_trainable_p"]
    print(
        f"trainable params: {trainable_p:,} || "
        f"all params: {total_p:,} || "
        f"trainable%: {100 * trainable_p / total_p:.4f} \n"
        f"non-cls trainable params: {peft_trainable_p:,} || "
        f"all params: {total_p:,} || "
        f"non-cls trainable%: {100 * peft_trainable_p / total_p:.4f}"
    )

##########################
# Dataset Utilities
##########################
label_key = "label"
image_path_key = "img"

def collate_fn(examples):
    pixel_values = torch.stack(
        [torch.Tensor(example["pixel_values"]) for example in examples]
    )
    labels = torch.tensor([example[label_key] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def preprocess(example_batch, transform_fn):
    example_batch["pixel_values"] = [
        transform_fn(image.convert("RGB")) for image in example_batch[image_path_key]
    ]
    return example_batch

def get_transforms(image_processor):
    if "height" in image_processor.size:
        return Compose(
            [
                Resize((image_processor.size["height"], image_processor.size["width"])),
                ToTensor(),
                Normalize(
                    mean=image_processor.image_mean, std=image_processor.image_std
                ),
            ]
        )
    elif "height" in image_processor.crop_size:
        return Compose(
            [
                Resize(
                    (
                        image_processor.crop_size["height"],
                        image_processor.crop_size["width"],
                    )
                ),
                ToTensor(),
                Normalize(
                    mean=image_processor.image_mean, std=image_processor.image_std
                ),
            ]
        )
    else:
        raise ValueError("Unknown image processor")

DATASET_NAME = (
    'cifar',
    'caltech101',
    'dtd',
    'flowers102',
    'pets',
    'svhn',
    'sun397',
    'patch_camelyon',
    'eurosat',
    'resisc45',
    'retinopathy',
    'clevr_count',
    'clevr_dist',
    'dmlab',
    'kitti_dist',
    'dsprites_loc',
    'dsprites_ori',
    'smallnorb_azi',
    'smallnorb_ele',
)
CLASSES_NUM = (100, 102, 47, 102, 37, 10, 397, 2, 10, 45, 5, 8, 6, 6, 4, 16, 16, 18, 9)

def get_classes_num(dataset_name):
    dict_ = {name: num for name, num in zip(DATASET_NAME, CLASSES_NUM)}
    return dict_[dataset_name]

DATASET_NAME_TO_URL = {
    "cifar": "fw407/vtab-1k_cifar",
    "caltech101": "fw407/vtab-1k_caltech101",
    "dtd": "fw407/vtab-1k_dtd",
    "flowers102": "fw407/vtab-1k_flowers102",
    "pets": "fw407/vtab-1k_pets",
    "svhn": "fw407/vtab-1k_svhn",
    "sun397": "fw407/vtab-1k_sun397",
    "patch_camelyon": "fw407/vtab-1k_patch_camelyon",
    "eurosat": "fw407/vtab-1k_eurosat",
    "resisc45": "fw407/vtab-1k_resisc45",
    "retinopathy": "fw407/vtab-1k_retinopathy",
    "clevr_count": "fw407/vtab-1k_clevr_count",
    "clevr_dist": "fw407/vtab-1k_clevr_distance",
    "dmlab": "fw407/vtab-1k_dmlab",
    "kitti_dist": "fw407/vtab-1k_kitti_distance",
    "dsprites_loc": "fw407/vtab-1k_dsprites_location",
    "dsprites_ori": "fw407/vtab-1k_dsprites_orientation",
    "smallnorb_azi": "fw407/vtab-1k_smallnorb_azimuth",
    "smallnorb_ele": "fw407/vtab-1k_smallnorb_elevation",
}
def get_dataset(dataset_name):
    dataset_url = DATASET_NAME_TO_URL[dataset_name]

    train_dataset = load_dataset(dataset_url, split="train")
    eval_dataset = load_dataset(dataset_url, split="validation")
    test_dataset = load_dataset(dataset_url, split="test")

    return  train_dataset, eval_dataset, test_dataset

MODEL_NAME_TO_URL = {
    "dino-v2-large": "facebook/dinov2-large",
    "vit-base": "google/vit-base-patch16-224-in21k",
    "vit-large": "google/vit-large-patch16-224-in21k",
}

def main():
    parser = HfArgumentParser((ScriptArguments, PEFTArguments, TrainingArguments))
    script_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    torch.use_deterministic_algorithms(True)
    set_seed(training_args.seed)

    # Load dataset
    train_dataset, eval_dataset, test_dataset = get_dataset(script_args.dataset_name)

    num_labels = get_classes_num(script_args.dataset_name)
    print(f"num_labels: {num_labels}")

    # Set image transforms
    model_name = script_args.model_name
    model_url = MODEL_NAME_TO_URL[model_name]
    image_processor = AutoImageProcessor.from_pretrained(model_url)
    transform_fn = get_transforms(image_processor)

    train_dataset.set_transform(lambda x: preprocess(x, transform_fn))
    eval_dataset.set_transform(lambda x: preprocess(x, transform_fn))
    test_dataset.set_transform(lambda x: preprocess(x, transform_fn))

    # Load model
    model = AutoModelForImageClassification.from_pretrained(
        model_url,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    ).to(device)

    # print(model)
    # print_trainable_parameters(model)

    ### added PEFT logic
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
            modules_to_save=["classifier"],
        )
    elif peft_name == "pissa":
        peft_config = LoraConfig(
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            modules_to_save=["classifier"],
            init_lora_weights='pissa',
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
            modules_to_save=["classifier"],
        )
    elif peft_name == "vera":
        peft_config = VeraConfig(
            r=peft_rank,
            vera_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            modules_to_save=["classifier"],
        )
    elif peft_name == "oft":
        peft_config = OFTConfig(
            oft_block_size=oft_block_size,
            use_cayley_neumann=oft_use_cayley_neumann,
            num_cayley_neumann_terms=oft_num_cayley_neumann_terms,
            module_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            modules_to_save=["classifier"],
        )
    elif peft_name == "boft":
        peft_config = BOFTConfig(
            boft_block_size=boft_b,
            boft_n_butterfly_factor=boft_m,
            boft_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            modules_to_save=["classifier"],
        )
    elif peft_name == "svft":
        # for SVFT turn off gradient requirement for all layers
        # PEFT library handles this internally
        for name, param in model.named_parameters():
            param.requires_grad = False

        assign_svft_layer = partial(LinearWithSVFT,
                                    off_diag=svft_off_diag,
                                    pattern=svft_pattern,
                                    fill_orthonormal=svft_fill_orthonormal)
        create_and_replace_modules(model, get_target_modules_list(model, peft_inserted_modules), assign_svft_layer)

        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
    elif peft_name == 'lora_xs':
        config = LoraConfig(
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=peft_dropout,
            target_modules=peft_inserted_modules,
            modules_to_save=["classifier"],
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
            modules_to_save=["classifier", "pooler"],
            init_lora_weights='pissa_orth',
            # init_lora_weights = 'pissa_niter_20',  # Using Fast-SVD，'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA initialization
        )
        print("PiSSA is Baking... (PiSSA initializing will take a while.)")
        model = get_peft_model(model, config)
        create_and_insert(model, config, psoft_orth, psoft_mag_out, psoft_mag_b, psoft_mag_a, psoft_use_cayley_neumann, psoft_num_cayley_neumann_terms)

        check_lora_A_row_orthogonality(model)
    elif peft_name == "head":
        classifier_modules = "classifier"
        for n, p in model.named_parameters():
            if all(c not in n for c in classifier_modules):
                p.requires_grad = False
    elif peft_name == 'full':
        pass
    else:
        raise ValueError("Unknown peft method")

    if peft_name not in ['head', 'full', 'svft', 'lora_xs', 'psoft', 'psoft-um']:
        model = get_peft_model(model, peft_config)

    print(model)

    # To make tensors contiguous for lora_xs and psoft.
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, Shape: {param.shape}")

    print_trainable_parameters(model)
    params_dict = get_trainable_params_dict(model)

    def get_classifier_modules(model_name):
        if model_name in {"dino-v2-large", "vit-base", "vit-large"}:
            return [
                "classifier",
            ]
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    # Setup Trainer
    peft_group = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
           and all(cls_name not in n for cls_name in get_classifier_modules(script_args.model_name))
    ]
    classifier_group = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
           and any(cls_name in n for cls_name in get_classifier_modules(script_args.model_name))
    ]
    optimizer = optim.AdamW(
        [
            {
                "params": peft_group,
                "lr": training_args.learning_rate,
            },
            {
                "params": classifier_group,
                "lr": script_args.cls_learning_rate,
            },
        ],
        weight_decay=training_args.weight_decay,
    )

    num_train_steps = math.ceil(
        len(train_dataset) / training_args.per_device_train_batch_size
    ) * training_args.num_train_epochs

    print(f"training_args.lr_scheduler_type: {training_args.lr_scheduler_type}")
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

    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers=(optimizer, scheduler),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # print("Exiting the program after logging trainable parameters.")
    # sys.exit("Program terminated intentionally after logging trainable parameters.")

    torch.cuda.reset_peak_memory_stats()
    train_results = trainer.train()
    best_checkpoint = trainer.state.best_model_checkpoint
    peak_memory_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
    print(f"Peak GPU memory allocated during training: {peak_memory_allocated:.2f} GB")
    print(f"best_checkpoint: {best_checkpoint}")
    wandb.log({'peak_gpu_memory_GB': peak_memory_allocated})

    with open(
            training_args.output_dir + f"_final_train_results.json", "w"
    ) as f:
        json.dump(train_results, f, indent=4)

    # print("model merging...")
    # if peft_name == 'svft':
    #     replace_svft_with_fused_linear(model, get_target_modules_list(model, peft_inserted_modules))
    # elif peft_name == "full" or peft_name == "head":
    #     pass
    # else:
    #     model = model.merge_and_unload()

    trainer.state.epoch = training_args.num_train_epochs + 1

    print("evaluating test dataset...")
    test_results = trainer.evaluate(test_dataset)
    print(test_results)
    #
    # for key in script_args.__dataclass_fields__:
    #     value = getattr(script_args, key)
    #     test_results[key] = value
    #
    # for key in training_args.__dataclass_fields__:
    #     if "accelerator" in key:
    #         continue
    #     value = getattr(training_args, key)
    #     test_results[key] = value
    #
    # test_results.update(params_dict)
    # pprint(test_results, indent=4)
    #
    # with open(
    #         training_args.output_dir + f"_final_test_results.json", "w"
    # ) as f:
    #     json.dump(test_results, f, indent=4)
    #
    # # Save to results.json
    # results_json_path = os.path.join(training_args.output_dir, script_args.results_json)
    # try:
    #     with open(results_json_path, "r") as f:
    #         results = json.load(f)
    # except FileNotFoundError:
    #     results = []
    #
    # results.append(test_results)
    # with open(results_json_path, "w") as f:
    #     json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()