import argparse
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, required=True)
    parser.add_argument(
        "--label", type=str, default="datasets/math_test_with_hand_label.json"
    )

    args = parser.parse_args()
    print(args)

    label_dict = {}
    label_data = []
    with open(args.label) as f:
        label_data = json.load(f)["data"]
    for row in label_data:
        id = row["id"]
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]
        label_dict[id] = {"question": question, "choices": choices, "answer": answer}

    correct = 0
    n = len(label_data)

    submission_df = pd.read_csv(args.submission)
    for index, row in submission_df.iterrows():
        id = row["id"]
        answer = row["answer"]

        if id not in label_dict.keys():
            print("Error:", f"id {id} not in label")
            continue

        if answer not in label_dict[id]["choices"]:
            print("Error:", f"id {id} with answer {answer} not in choices")
            continue

        if label_dict[id]["answer"] not in label_dict[id]["choices"]:
            correct += 1
            print("Auto +1:", id, label_dict[id]["answer"])
            continue

        if answer == label_dict[id]["answer"]:
            correct += 1
        else:
            print(f"Wrong in id {id}:", answer, label_dict[id]["answer"])

    print("-" * 10)
    print("Accuracy:", correct / n * 100)


if __name__ == "__main__":
    main()
