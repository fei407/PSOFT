# Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation (PSOFT)

This is the official repository for the paper **Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation**.

The code framework is adapted from [SVFT](https://github.com/VijayLingam95/SVFT) and incorporates implementations from [LoRA](https://github.com/microsoft/LoRA) and [LoRA-XS](https://github.com/MohammadrezaBanaei/LoRA-XS).

**Overview of PSOFT**
![Overview of PSOFT](0-Fig/psoft.svg "Overview of PSOFT")

## Repository Overview
The repository is organized as follows:

* [NLU](NLU/):  Source code for fine-tuning and evaluation on the _GLUE_ benchmarks.
* [Vision](Vision/): Source code for fine-tuning and evaluation on the _VTAB-1K_ benchmarks.
* [Math](Math/):  Source code for fine-tuning on _MetaMathQA-40K_ and evaluation on the _GSM-8K_ and _MATH_ datasets.
* [Commonsense](Commonsense/): Source code for fine-tuning on _Commonsense-15K_ and evaluation on the _Commonsense Reasoning_ benchmarks.

## QuickStart of PSOFT

### Step 0. Preparations
Replace [yourworkspace] with your workspace path.
```bash
conda create -n psoft python=3.10
conda activate psoft 

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

cd peft-v0.17.0/
pip install -e .

export PYTHONPATH="/[yourworkspace]/PSOFT:$PYTHONPATH"
find . -name "*.sh" -exec chmod +x {} \;
```
Some models or datasets require permission for usage. Please log in to your Hugging Face account using an access token.

Generate your Access Token from [settings/tokens](https://huggingface.co/settings/tokens) and log in. 
```bash
huggingface-cli login
Access Tokens:[Copy and paste your Access Tokens]
```

### Step 1. Natural Language Understanding 

**Fine-tune** and **evaluate** using the [DeBERTaV3-base](https://huggingface.co/microsoft/deberta-v3-base) model:

```bash
cd NLU/script/
./deberta_v3_base_psoft-cola.sh
```

### Step 2. Visual Classification

**Fine-tune** and **evaluate** using the [ViT-base/16](https://huggingface.co/google/vit-base-patch16-224-in21k) model:

```bash
cd script/
./vit_base-psoft.sh
```

### Step 3. Mathematical Question Answering

**Fine-tune** using the [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) model:
```bash
cd Math/script/
./llama-3-3b-psoft.sh
```

To **evaluate** the fine-tuned model in a new environment:
```bash
conda create -n vllm python==3.10
conda activate vllm

pip install vllm==0.10.0
pip install fraction==2.2.0
pip install jsonlines==4.0.0
```
Before running the script please change the path name in [eval_all.sh](Math/) to match the path of results:
```bash
cd ../
./eval_all.sh
```

### Step 4. Commonsense Reasoning

**Fine-tune** using the [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) model:

```bash
cd Commonsense/script/
./llama-3-8b-psoft.sh
```

Prepare datasets before **evaluation**:

```bash
cd /PSOFT/..
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git
cd LLM-Adapters
mkdir -p ../PSOFT/Commonsense/dataset
cp -r dataset/* ../PSOFT/Commonsense/dataset
```

Edit the path in [eval_all.sh](Commonsense/) to match the results directory:

```bash
cd /PSOFT/Commonsense/
./eval_all.sh
```

### Key Parameter Descriptions
+ `psoft_orth`: Enables Cayley parameterization for the orthogonal matrix R.
+ `psoft_mag_b`: Enables tuning of the scaling vector Beta applied after R.
+ `psoft_mag_a`: Enables tuning of the scaling vector Alpha applied before R.
+ `psoft_use_cayley_neumann`: Enables Cayley Neumann.
+ `psoft_num_cayley_neumann_terms`: Enables the terms of Cayley Neumann.

## Citation
Please cite our paper if PSOFT provides insights or inspiration for your work:
```
@article{wu2025psoft,
  title={Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation},
  author={Wu, Fei and Hu, Jia and Min, Geyong and Wang, Shiqiang},
  journal={arXiv preprint arXiv:2505.11235},
  year={2025}
}
```



