import json
import logging
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence

import torch
from torch.utils.data import Dataset
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)

import random

from trl import SFTTrainer, is_xpu_available


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    seed: Optional[int] = field(
        default=42, metadata={"help": "Seed for reproducibility"}
    )

    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"}
    )
    dataset_path: Optional[str] = field(
        default="datasets/math_train.json", metadata={"help": "the dataset path"}
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )
    log_with: Optional[str] = field(
        default="none", metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    model_max_length: Optional[int] = field(
        default=1024, metadata={"help": "Input sequence length"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    use_peft: Optional[bool] = field(
        default=False, metadata={"help": "Wether to use PEFT or not to train adapters"}
    )
    trust_remote_code: Optional[bool] = field(
        default=False, metadata={"help": "Enable `trust_remote_code`"}
    )
    output_dir: Optional[str] = field(
        default="outputs", metadata={"help": "the output directory"}
    )
    peft_lora_r: Optional[int] = field(
        default=64, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    logging_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of logging steps"}
    )
    token: Optional[bool] = field(
        default=True, metadata={"help": "Use HF auth token to access the model"}
    )
    num_train_epochs: Optional[int] = field(
        default=3, metadata={"help": "the number of training epochs"}
    )
    max_steps: Optional[int] = field(
        default=-1, metadata={"help": "the number of training steps"}
    )
    save_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Number of updates steps before two checkpoint saves"},
    )
    save_total_limit: Optional[int] = field(
        default=10, metadata={"help": "Limits total number of checkpoints."}
    )
    push_to_hub: Optional[bool] = field(
        default=False, metadata={"help": "Push the model to HF Hub"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the model on HF Hub"}
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset_path, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(dataset_path))["data"]

        self.tokenizer = tokenizer

        self.prompt_template = (
            "<s>[INST] <<SYS>>\n"
            "{{ Trả lời câu hỏi sau bằng cách đưa ra đáp án chính xác nhất. Đáp án sẽ là một trong các lựa chọn A, B, C, D. Hãy suy nghĩ từng bước một. }}\n"
            "<</SYS>>\n"
            "{{ "
            "### Câu hỏi: {question}\n"
            "### Các lựa chọn: \n"
            "{text_choices}"
            " }}"
            " [/INST]"
        )

        self.output_template = (
            " {{ "
            "### Giải thích: {explanation}\n"
            "### Đáp án: {answer}" " }} </s>"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        question = self.data[i]['question']
        choices = self.data[i]['choices']
        explanation = self.data[i]["explanation"] if "explanation" in self.data[i] else ""
        answer = self.data[i]["answer"]
        text_choices = "\n".join(choices)

        data_map = {
            'question': question,
            'text_choices': text_choices,
            'explanation': explanation,
            'answer': answer
        }

        return dict(
            input_ids=self.prompt_template.format(question, text_choices), 
            labels=self.output_template.format(explanation, answer)
        )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Set seed for reproducibility
    set_seed(script_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = 'INFO'
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #     + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    # )
    logger.info(f"Script arguments {script_args}")

    ###############
    # Load datasets
    ###############
    raw_dataset = load_dataset("json", data_files=script_args.dataset_path, field="data")
    logger.info(
        f"Raw dataset {raw_dataset}"
    )

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name, trust_remote_code=script_args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.truncation_side is not None:
        tokenizer.truncation_side = script_args.truncation_side

    tokenizer.model_max_length = script_args.model_max_length

    #####################
    # Apply chat template
    #####################
    train_dataset = raw_dataset["train"]
    train_dataset = SupervisedDataset(tokenizer=tokenizer, dataset_path=script_args.dataset_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the processed training set:\n\n{train_dataset[index]}")

    # # Step 1: Load the model
    # if script_args.load_in_8bit and script_args.load_in_4bit:
    #     raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    # elif script_args.load_in_8bit or script_args.load_in_4bit:
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    #     )
    #     # Copy the model to each device
    #     device_map = (
    #         {"": f"xpu:{Accelerator().local_process_index}"}
    #         if is_xpu_available()
    #         else {"": Accelerator().local_process_index}
    #     )
    #     torch_dtype = torch.bfloat16
    # else:
    #     device_map = None
    #     quantization_config = None
    #     torch_dtype = None

    # model = AutoModelForCausalLM.from_pretrained(
    #     script_args.model_name,
    #     quantization_config=quantization_config,
    #     device_map=device_map,
    #     trust_remote_code=script_args.trust_remote_code,
    #     torch_dtype=torch_dtype,
    #     token=script_args.token,
    # )

    # print(script_args)

    # tokenizer = AutoTokenizer.from_pretrained(
    #     script_args.model_name, trust_remote_code=script_args.trust_remote_code
    # )
    # tokenizer.pad_token = tokenizer.eos_token

    # # Step 2: Load the dataset
    # dataset = load_dataset("json", data_files=script_args.dataset_path, field="data")

    # # Step 3: Define the training arguments
    # training_args = TrainingArguments(
    #     output_dir=script_args.output_dir,
    #     per_device_train_batch_size=script_args.batch_size,
    #     gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    #     learning_rate=script_args.learning_rate,
    #     logging_steps=script_args.logging_steps,
    #     num_train_epochs=script_args.num_train_epochs,
    #     max_steps=script_args.max_steps,
    #     report_to=script_args.log_with,
    #     save_steps=script_args.save_steps,
    #     save_total_limit=script_args.save_total_limit,
    #     push_to_hub=script_args.push_to_hub,
    #     hub_model_id=script_args.hub_model_id,
    #     gradient_checkpointing=script_args.gradient_checkpointing,
    #     # TODO: uncomment that on the next release
    #     # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
    # )

    # # Step 4: Define the LoraConfig
    # if script_args.use_peft:
    #     peft_config = LoraConfig(
    #         r=script_args.peft_lora_r,
    #         lora_alpha=script_args.peft_lora_alpha,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    # else:
    #     peft_config = None


    # def formatting_prompts_func(example):
    #     output_texts = []
    #     for i in range(len(example)):
    #         # id = example["id"][i]
    #         question = example["question"][i]
    #         choices = example["choices"][i]
    #         explanation = example["explanation"][i]
    #         answer = example["answer"][i]

    #         # print(id, choices)
    #         text_choices = "\n".join(choices)

    #         # Gen explanation
    #         text = (
    #             "Trả lời câu hỏi sau bằng cách đưa ra đáp án chính xác nhất. Đáp án sẽ là một trong các lựa chọn A, B, C, D. Hãy suy nghĩ từng bước một.\n"
    #             f"### Câu hỏi: {question}\n"
    #             "### Các lựa chọn: \n"
    #             f"{text_choices}\n"
    #             f"### Giải thích: {explanation}\n"
    #             f"### Đáp án: {answer}"
    #         )
    #         output_texts.append(text)
    #     return output_texts


    # # Step 5: Define the Trainer
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=dataset["train"],
    #     peft_config=peft_config,
    #     max_seq_length=script_args.seq_length,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     formatting_func=formatting_prompts_func,
    #     packing=False,
    # )

    # trainer.train()

    # # Step 6: Save the model
    # trainer.save_model(script_args.output_dir)

if __name__ == '__main__':
    main()