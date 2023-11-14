# ZAIC-2023-Elementary-Math-Solving

# Cấu hình

Pytorch: 2.1.0
CUDA: 12.1

# Cài đặt

```bash
pip install -r requirements.txt
```

```bash
huggingface-cli login
```

# Huấn luyện

## Baseline Llama-2-7b LoRA 8bit

```bash
python llama_recipes/finetuning.py --use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b-hf --output_dir outputs
```

## Baseline zephyr-7b-alpha

```bash
python llama_recipes/finetuning.py --use_peft --peft_method lora --quantization --model_name HuggingFaceH4/zephyr-7b-alpha --dataset zalo_math_fill_missing_explain_35 --output_dir outputs --use_wandb --wandb_entity baolocpham --wandb_key key
```

# Inference

```bash
python llama_recipes/inference.py --quantization --model_name <model_name> --peft_model <output_dir> --max_new_tokens <max new tokens> --prompt_file test_prompt.txt
```

Template for my personal Experiment Tracking hyperparameters

This project base on this NeptuneAI's blog about [the Experiment Tracking](https://neptune.ai/blog/experiment-management)

Using [Hydra](https://hydra.cc) for reading config files and [WANDB](https://wandb.ai) for tracking experiments

This will help reduce repeated taskes and tracking your experiment more effective.

## Demo

```
python demo_hydra.py
```

Output:

```
project: ORGANIZATION/experiment-tracking
name: experiment-tracking-default-risk
wandb:
  WANDB_API_KEY: YOUR_WANDB_API
  entity: YOUR_WANDB_API
  project: experiment-tracking
  name: TEST
parameters:
  n_cv_splits: 5
  validation_size: 0.2
  stratified_cv: true
  shuffle: 1
  rf__n_estimators: 2000
  rf__criterion: gini
  rf__max_depth: 40
  rf__class_weight: balanced
  rf__max_features: 0.3
```

You can change your hyperparams in `cli` like this

```
python demo_hydra.py parameters.n_cv_splits=10 parameters.validation_size=0.9
```
