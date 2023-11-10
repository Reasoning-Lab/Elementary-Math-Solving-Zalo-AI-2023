# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

class ZaloMathDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=224):
        self.ann = json.load(open(dataset_config.data_path))['data']
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]

        id = ann['id']
        question = ann['question']
        choices = ann['choices']
        explanation = ann['explanation'] if 'explanation' in ann else ""
        answer = ann['answer']

        # print(id, choices)
        text_choices = "\n".join(choices)

        prompt = (
            "<s>[INST] <<SYS>>\n"
            "{{ Trả lời câu hỏi sau bằng cách đưa ra đáp án chính xác nhất. Đáp án sẽ là một trong các lựa chọn A, B, C, D. Hãy suy nghĩ từng bước một. }}\n"
            "<</SYS>>\n"
            "{{ "
            f"### Câu hỏi: {question}\n"
            "### Các lựa chọn: \n"
            f"{text_choices}"
            " }}"
            " [/INST]"
        )

        output = (
            " {{ "
            f"### Giải thích: {explanation}\n"
            f"### Đáp án: {answer}"
            " }} </s>"
        )

        example = prompt + output
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }