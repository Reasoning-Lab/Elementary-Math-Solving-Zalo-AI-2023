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

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        id = ann['id']
        question = ann['question']
        choices = ann['choices']
        explanation = ann['explanation'] if 'explanation' in ann else ""
        answer = ann['answer']

        # print(id, choices)
        text_choices = "\n".join(choices)

        prompt = (
             "Trả lời câu hỏi sau bằng cách đưa ra đáp án chính xác nhất. Đáp án sẽ là một trong các lựa chọn A, B, C, D. Hãy suy nghĩ từng bước một.\n"
            f"### Câu hỏi: {question}\n"
            "### Các lựa chọn: \n"
            f"{text_choices}\n"
        )

        output = (
            f"### Giải thích: {explanation}\n"
            f"### Đáp án: {answer}"
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
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }