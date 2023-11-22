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
wandb login
```

# Huấn luyện

## Baseline Llama-2-7b LoRA 8bit

```bash
python llama_recipes/finetuning.py --use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b-hf --output_dir outputs
```

## Baseline zephyr-7b-alpha
with zalo_math_fill_missing_explain_4 (using GPT4)

now with `load_in` options `['4bit', '8bit']`

```bash
python llama_recipes/finetuning.py --use_peft --peft_method lora --quantization --model_name HuggingFaceH4/zephyr-7b-alpha --dataset zalo_math_fill_missing_explain_35 --output_dir outputs --use_wandb --wandb_entity baolocpham --wandb_key KEY --num_epochs 2
```

```bash
python llama_recipes/finetuning.py --use_peft --peft_method lora --quantization --model_name HuggingFaceH4/zephyr-7b-alpha --dataset zalo_math_fill_missing_explain_4 --output_dir outputs --batching_strategy packing --num_epochs 6 --load_in 4bit --use_wandb --wandb_entity baolocpham --wandb_key KEY
```

## Pretraining

```bash
bash run_pt.sh
```

## Finetune - SFTTrainer

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file <multi_gpu.yaml / deepspeed_zero3.yaml> --num_processes=1 sft.py config_lora.yaml
```

# Inference

```bash
python inference.py --model_name hllj/zephyr-7b-beta-vi-math --peft_model outputs-sft-zephyr-beta-v1/checkpoint-1500/ --load_in 4bit --max_new_tokens 512 --temperature 0.1
```

- model_name: base model mình sử dụng để finetune
- peft_model: thư mục chứa file LoRA đã finetune
- load_in: 4bit / 8bit quantization
- max_new_tokens: số lượng token generate tối đa.
- temperature: temperature cho sampling, nên để 0.1 - 0.5
