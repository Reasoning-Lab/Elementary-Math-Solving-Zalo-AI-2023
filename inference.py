import fire
import os
import sys
import time
import json
import re
import random
import logging

import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Function to load the main model for text generation
def load_model(model_name, quantization, load_in):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
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

def get_user_prompt(example):
    question = example["question"]
    choices = example["choices"]

    text_choices = '['
    for idx, choice in enumerate(choices):
        text_choices += f"'{choice}'"
        if idx != len(choices) - 1:
            text_choices += ','
    text_choices += ']'

    user_prompt = (
        "<s>\n"
        "Below is a math exercise. Provide a solution to that problem, if given multiple choices to answer; please give a final choice for solving that problem.\n"
        f"### Question: {question}\n"
        "### Choices: "
        f"{text_choices}\n"
        "### Explanation: "
    )
    return user_prompt


def main(
    model_name,
    peft_model: str = None,
    quantization: bool = False,
    load_in: str = "4bit",
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
    log_filename: str = "log.txt",
    **kwargs,
):
    logging.basicConfig(
        filename=log_filename,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )

    log = logging.getLogger(__name__)
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
            log.error(
                "Module 'optimum' not found. Please install 'optimum' it before proceeding."
            )

    tokenizer = AutoTokenizer.from_pretrained(peft_model)

    results = []

    for idx, example in enumerate(data):
        log.info(f"Processing {idx}")
        user_prompt = get_user_prompt(example)
        id = example["id"]
        choices = example["choices"]
        input = tokenizer(user_prompt, return_tensors="pt")

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
        log.info(f"the inference time is {e2e_inference_time} ms")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_text = tokenizer.decode(
            outputs[0][input["input_ids"].shape[1] :], skip_special_tokens=True
        )

        answer_text = None

        for text in gen_text.split("###"):
            if 'Final choice' in text:
                answer_text = text
                break

        if answer_text is None:
            answer_text = gen_text

        log.info(f"Output text: {output_text}")
        log.info(f"Gen text {gen_text}")
        log.info(f"Answer text {answer_text}")

        answer = None
        for choice in choices:
            full_answer = choice
            value_only = re.sub("[ABCD]. ", "", full_answer)
            if full_answer in answer_text:
                answer = choice
                break
        if answer is None:
            for choice in choices:
                value_only = re.sub("[ABCD]. ", "", full_answer)
                if value_only in answer_text:
                    answer = choice
                    break
        log.info(f"Answer {answer}")
        if answer is None:
            answer = random.choice(choices)
            log.info(f"Random Answer {answer}")
        results.append({"id": id, "answer": answer})

    result_df = pd.DataFrame.from_dict(results)
    result_df.to_csv(os.path.join(peft_model, "submission.csv"), index=False)


if __name__ == "__main__":
    fire.Fire(main)
