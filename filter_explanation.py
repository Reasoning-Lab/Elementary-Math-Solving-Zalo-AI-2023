import json

train_path = 'datasets/math_train.json'

with open(train_path, 'r') as f:
    data = json.load(f)['data']

filter_explanation_data = []

for example in data:
    if 'explanation' in example:
        filter_explanation_data.append(example)

content = {
    "__count__": len(filter_explanation_data),
    "data": filter_explanation_data
}

with open('datasets/filter_explanation_math_train.json', 'w', encoding='utf-8') as f:
    json.dump(content, f, ensure_ascii=False, indent=4)