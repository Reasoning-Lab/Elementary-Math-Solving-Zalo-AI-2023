import argparse
import json
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--data_example_path', type=str, default='datasets/math_train.json',
                        help='')
    parser.add_argument('--prompt_path', type=str, default='datasets/prompt/generate_multiple_questions.txt',
                        help='')
    parser.add_argument('--n_loop', type=int, default=10,
                        help='')
    args = parser.parse_args()
    return args


def load_json(path):
    with open(path, 'r') as f: return json.load(f)


def load_text(path):
    with open(path, 'r') as file: return file.read()


def number_choice_sample(number_sample=1200):
    number_choice1, number_choice2, number_choice3 = random.choices(range(number_sample), k=3)
    return number_choice1, number_choice2, number_choice3


def main():
    args = parse_arguments()
    raw_data = load_json(args.data_example_path)
    prompt = load_text(args.prompt_path)

    valid_number=False
    while not valid_number:
        number_choices = number_choice_sample()
        data_entries = [raw_data['data'][idx] for idx in number_choices]
        if all('explanation' in data for data in data_entries):
            data_list = [{k: v for k, v in data.items() if k not in ['id', 'choices']} for data in data_entries]
            valid_number = True
    
    new_prompt = prompt.format(
        multiple_choice1=data_list[0],
        multiple_choice2=data_list[1],
        multiple_choice3=data_list[2]
    )
    print(new_prompt)
    return new_prompt

if __name__ == "__main__":
    main()