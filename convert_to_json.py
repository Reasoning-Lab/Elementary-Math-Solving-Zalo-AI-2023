import pandas as pd
import json

df = pd.read_excel("/home/ZAIC-2023-Elementary-Math-Solving/collect_data.xlsx")

# Convert DataFrame to JSON format as specified
json_result = df.apply(
    lambda x: {
        "original id": x["original id"],
        "question": x["question"],
        "options": [x["A"], x["B"], x["C"], x["D"]],
        "explanation": x["explanation"],
        "answer": x["answer"],
    },
    axis=1,
).to_json(orient="records", force_ascii=False)

json_data = {}
json_data["data"] = json.loads(json_result)
# Write the dictionary to a JSON file with UTF-8 encoding
with open("collect_data.json", "w", encoding="utf8") as file:
    json.dump(json_data, file, ensure_ascii=False, indent=4)
