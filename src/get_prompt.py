import argparse
import json
import os
import openai
from tqdm import tqdm
from dotenv import load_dotenv
import time

load_dotenv()

openai.api_type = "azure"
openai.api_base = "https://openai-enterprise-dci-eastus2-001.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_arguments():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--data_example_path', type=str, default='datasets/sorted_with_missing_explain_4.json',
                        help='')
    parser.add_argument('--prompt_path', type=str, default='datasets/prompt/gen_explain.txt',
                        help='')
    parser.add_argument('--gpt_type', type=str, default="gpt-35-turbo",
                        help='')
    args = parser.parse_args()
    return args


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_text(path):
    with open(path, 'r') as file:
        return file.read()


def save_json(path, file_save):
    with open(path, 'w') as f:
        return json.dump(file_save, f)


def main():
    args = parse_arguments()
    data = load_json(args.data_example_path)['data'][423]
    prompt = load_text(args.prompt_path)
    prefixes = ['A.', 'B.', 'C.', 'D.']

    # for data in enumerate(raw_data):
    #     print(data)
    print(data['choices'])
    modified_choices = [choice if not any(choice.startswith(
        p) for p in prefixes) else '- ' + choice.split('. ', 1)[1] for choice in data['choices']]
    choices = '\n'.join(modified_choices)
    new_prompt = prompt.format(
        question=data['question'],
        choices=choices,
        answer=data['answer'][3:]  # Remove the A B C D.
    )
    print(new_prompt)


if __name__ == "__main__":
    main()
