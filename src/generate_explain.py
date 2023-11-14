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
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument(
        "--data_example_path", type=str, default="datasets/math_train.json", help=""
    )
    parser.add_argument(
        "--prompt_path", type=str, default="datasets/prompt/gen_explain.txt", help=""
    )
    parser.add_argument("--gpt_type", type=str, default="gpt-35-turbo", help="")
    args = parser.parse_args()
    return args


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_text(path):
    with open(path) as file:
        return file.read()


def save_json(path, file_save):
    with open(path, "w") as f:
        return json.dump(file_save, f)


def gen_text(message, args):
    response = openai.ChatCompletion.create(
        engine=args.gpt_type,
        messages=[{"role": "system", "content": message}],
        temperature=0,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    # max_tokens=350,
    return response["choices"][0]["message"]["content"]


def main():
    args = parse_arguments()
    raw_data = load_json(args.data_example_path)
    prompt = load_text(args.prompt_path)
    prefixes = ["A.", "B.", "C.", "D."]
    count = 0
    for _idx, data in enumerate(tqdm(raw_data["data"])):
        if "explanation" in set(data.keys()):
            continue
        # time.sleep(5)
        # print(data)
        modified_choices = [
            choice
            if not any(choice.startswith(p) for p in prefixes)
            else "- " + choice.split(". ", 1)[1]
            for choice in data["choices"]
        ]
        choices = "\n".join(modified_choices)
        new_prompt = prompt.format(
            question=data["question"],
            choices=choices,
            answer=data["answer"][3:],  # Remove the A B C D.
        )
        # print(new_prompt)
        respond = gen_text(new_prompt, args)
        data_format = {
            "id": data["id"],
            "question": data["question"],
            "explaination": respond,
        }
        save_json(
            path=f"datasets/raw/gpt-35-only-missing/file_{count}.json",
            file_save=data_format,
        )
        count += 1
        # print(data_format)
        # break


if __name__ == "__main__":
    main()
