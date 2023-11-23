# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import json
import re
import random

import pandas as pd

import torch
from transformers import LlamaTokenizer, AutoTokenizer

from inference.safety_utils import get_safety_checker
from inference.model_utils import load_model, load_peft_model

import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and set the log file name
file_handler = logging.FileHandler("inference.log")

# Create a stream handler to display log messages on the console
stream_handler = logging.StreamHandler()

# Configure the log message format
formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def get_user_prompt(example, one_shot):
    question = example["question"]
    choices = example["choices"]

    text_choices = "\n".join(choices)

    text_choices = "\n".join(choices)
    # logger.info(f"one_shot: {one_shot}")
    if one_shot:
        user_prompt = (
            "<s>[INST] <<SYS>>\n"
            "{{ Trả lời câu hỏi sau bằng cách đưa ra đáp án chính xác nhất. Đáp án sẽ là một trong các lựa chọn A, B, C, D. Hãy suy nghĩ từng bước một. }}\n"
            "<</SYS>>\n"
            "{{ "
            f"### Câu hỏi: {question}\n"
            "### Các lựa chọn: \n"
            f"{text_choices}"
            " }}"
            " [/INST]"
            " {{ "
            "### Đáp án (không cần giải thích): "
        )
    else:
        user_prompt = (
            "<s>[INST] <<SYS>>\n"
            "{{ Trả lời câu hỏi sau bằng cách đưa ra đáp án chính xác nhất. Đáp án sẽ là một trong các lựa chọn A, B, C, D. Hãy suy nghĩ từng bước một. }}\n"
            "<</SYS>>\n"
            "{{ "
            f"### Câu hỏi: {question}\n"
            "### Các lựa chọn: \n"
            f"{text_choices}"
            " }}"
            " [/INST]"
            " {{ "
            "### Giải thích: "
        )
    return user_prompt


def main(
    model_name,
    peft_model: str = None,
    quantization: bool = False,
    load_in: str = "4bit",
    max_length: int | None = None,
    max_new_tokens=100,  # The maximum numbers of tokens to generate
    one_shot: bool = False,
    test_file: str = "datasets/math_test.json",
    seed: int = 42,  # seed value for reproducibility
    do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool = True,  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float = 1.0,  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
    top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int = 1,  # [optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool = False,  # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool = False,  # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool = True,  # Enable safety check with Salesforce safety flan t5
    max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False,  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs,
):
    with open(test_file) as f:
        data = json.load(f)["data"]

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_name, quantization, load_in)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer

            model = BetterTransformer.transform(model)
        except ImportError:
            logger.info(
                "Module 'optimum' not found. Please install 'optimum' it before proceeding."
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    logger.info(f"TOKENIZER max_length: {max_length}")
    for idx, example in enumerate(data):
        logger.info(f"Processing {idx}")
        user_prompt = get_user_prompt(example, one_shot)
        id = example["id"]
        choices = example["choices"]
        input = tokenizer(
            user_prompt,
            max_length=max_length,
            truncation=True if max_length != None else False,
            return_tensors="pt",
        )

        batch = {k: v.to("cuda") for k, v in input.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs,
            )
        e2e_inference_time = (time.perf_counter() - start) * 1000
        logger.info(f"the inference time is {e2e_inference_time} ms")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_text = tokenizer.decode(
            outputs[0][input["input_ids"].shape[1] :], skip_special_tokens=True
        )
        if not one_shot:
            if len(gen_text.split("###")) > 1:
                answer_text = gen_text.split("###")[1]
            else:
                answer_text = gen_text

            print(f"Output text: {output_text}")
            print(f"Gen text {gen_text}")

            answer = None
            for choice in choices:
                full_answer = choice
                value_only = re.sub("[ABCD]. ", "", full_answer)
                if full_answer in answer_text or value_only in answer_text:
                    answer = choice
                    break
            print(f"Answer {answer}")
            if answer is None:
                answer = random.choice(choices)
                print(f"Random Answer {answer}")
        elif one_shot:
            if len(gen_text.split("###")) > 1:
                answer_text = gen_text.split("###")[1]
            else:
                answer_text = gen_text
            answer_to_map = gen_text[:-3]

            logger.info(f"Output text: {output_text}")
            logger.info(f"Gen text {answer_to_map}")
            # Initialize a dictionary to map answers
            answer_mapping = {}

            # Iterate through choices and map the answer
            for choice in choices:
                # Split the choice into option (e.g., "A.") and text (e.g., "24 phút")
                option, text = choice.split(". ")

                # Store the mapping in the dictionary (with option as the key)
                answer_mapping[option] = text

            # Map the answer to the full choice
            answer = None
            for option, text in answer_mapping.items():
                if text in answer_to_map:
                    answer = option + " " + text
                    break
            logger.info(f"Answer {answer}")
            if answer is None:
                answer = random.choice(choices)
                logger.info(f"Random Answer {answer}")
        results.append({"id": id, "answer": answer})

    result_df = pd.DataFrame.from_dict(results)
    result_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
