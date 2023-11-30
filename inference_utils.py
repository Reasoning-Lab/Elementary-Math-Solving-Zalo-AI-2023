import re


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
        "Below is a math exercise. Provide a solution to that problem, if given multiple choices to answer; please give a final choice for solving that problem.\n"
        f"### Question: {question}\n"
        "### Choices: "
        f"{text_choices}\n"
        "### Explanation: "
    )
    return user_prompt

def post_processing_answer(answer_text, choices):
    answer = None
    for choice in choices:
        full_answer = choice
        if full_answer in answer_text:
            print('full answer process', full_answer)
            answer = choice
            break
    
    if answer is None:
        for choice in choices:
            value_only = re.sub("[ABCD]. ", "", choice)
            if value_only in answer_text:
                print('value only process', value_only)
                answer = choice
                break
    
    if answer is None:
        for choice in choices:
            tag_only = choice.split('.')[0].strip()
            print('*' * 10)
            print('check tag', tag_only)
            if tag_only in answer_text:
                print('tag only process', tag_only)
                answer = choice
                break

    return answer