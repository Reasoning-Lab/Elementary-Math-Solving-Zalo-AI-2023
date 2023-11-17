# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaConfig


# Function to load the main model for text generation
def load_model(model_name, quantization, load_in):
    print(f"LOAD_IN model: {load_in}")
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        # load_in_4bit=quantization,
        load_in_4bit=True if load_in == "4bit" else None,
        load_in_8bit=True if load_in == "8bit" else None,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model


# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    model_config = LlamaConfig.from_pretrained(config_path)
    model = LlamaForCausalLM(config=model_config)
    return model
