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
from transformers import LlamaTokenizer

from inference.safety_utils import get_safety_checker
from inference.model_utils import load_model, load_peft_model


def get_user_prompt(example):
    question = example["question"]
    choices = example["choices"]

    text_choices = "\n".join(choices)

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
            print(
                "Module 'optimum' not found. Please install 'optimum' it before proceeding."
            )

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # safety_checker = get_safety_checker(enable_azure_content_safety,
    #                                     enable_sensitive_topics,
    #                                     enable_salesforce_content_safety,
    #                                     )

    # # Safety check of the user prompt
    # safety_results = [check(user_prompt) for check in safety_checker]
    # are_safe = all([r[1] for r in safety_results])
    # if are_safe:
    #     print("User prompt deemed safe.")
    #     print(f"User prompt:\n{user_prompt}")
    # else:
    #     print("User prompt deemed unsafe.")
    #     for method, is_safe, report in safety_results:
    #         if not is_safe:
    #             print(method)
    #             print(report)
    #     print("Skipping the inference as the prompt is not safe.")
    #     sys.exit(1)  # Exit the program with an error status

    results = []
    print(f"TOKENIZER max_length: {max_length}")
    for idx, example in enumerate(data):
        print(f"Processing {idx}")
        user_prompt = get_user_prompt(example)
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
        print(f"the inference time is {e2e_inference_time} ms")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_text = tokenizer.decode(
            outputs[0][input["input_ids"].shape[1] :], skip_special_tokens=True
        )

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
        results.append({"id": id, "answer": answer})

    result_df = pd.DataFrame.from_dict(results)
    result_df.to_csv("submission.csv", index=False)

    # # Safety check of the model output
    # safety_results = [check(output_text) for check in safety_checker]
    # are_safe = all([r[1] for r in safety_results])
    # if are_safe:
    #     print("User input and model output deemed safe.")
    #     print(f"Model output:\n{output_text}")
    # else:
    #     print("Model output deemed unsafe.")
    #     for method, is_safe, report in safety_results:
    #         if not is_safe:
    #             print(method)
    #             print(report)


if __name__ == "__main__":
    fire.Fire(main)
