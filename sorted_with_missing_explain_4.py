import json

def sort_json_by_id(json_file):
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)

    def dict_to_int(item):
        string_value = item['id']
        int_value = int(string_value)
        return int_value

    data["data"].sort(key=dict_to_int)
    # Return the sorted data items.
    return data

if __name__ == "__main__":
     # Load the JSON file.
    json_file = "datasets/with-missing-explain-4.json"
    data = sort_json_by_id(json_file)

    with open("datasets/sorted_with_missing_explain_4.json", "w",encoding='utf8' ) as outfile:
        json.dump(data, outfile, ensure_ascii=False,indent=4),