# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    run_validation: bool = True
    batch_size_training: int = 4
    batching_strategy: str = "padding"  # alternative: padding
    context_length: int | None = 1024
    max_length: int = 2048
    gradient_accumulation_steps: int = 4
    num_epochs: int = 6
    num_workers_dataloader: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    gamma: float = 0.85
    seed: int = 42
    use_fp16: bool = False
    mixed_precision: bool = True
    val_batch_size: int = 1
    dataset = "zalo_math_dataset"
    peft_method: str = "lora"  # None , llama_adapter, prefix
    use_peft: bool = False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    load_in: str = "8bit"
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str = (
        "PATH/to/save/FSDP/model"  # will be used if using FSDP
    )
    dist_checkpoint_folder: str = "fine-tuned"  # will be used if using FSDP
    save_optimizer: bool = False  # will be used if using FSDP
    use_fast_kernels: bool = False  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = False
    wandb_project: str = "zaloai_2023_math_solving"
    wandb_group: str = "finetuning_sft"
    wandb_entity: str = "baolocpham"
    wandb_key: str = "KEY"
