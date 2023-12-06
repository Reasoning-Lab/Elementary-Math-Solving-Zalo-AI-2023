import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def read_data():
    import pandas as pd
    import json

    # Load JSON data from a file (replace 'your_file.json' with the actual file path)
    with open("./datasets/qualified_data.json", encoding="utf8") as json_file:
        data = json.load(json_file)["data"]
    return pd.DataFrame(data)


def get_model_and_tokenizer(model_path="intfloat/e5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return tokenizer, model


def process_data(df, need_header=False):
    if need_header:
        df["information"] = df[["question", "choices", "explanation"]].apply(
            lambda x: "\n".join(x.dropna().astype(str).values), axis=1
        )
    else:
        df["information"] = df[["question", "choices", "explanation"]].apply(
            lambda x: "\n".join(x.dropna().astype(str).values), axis=1
        )
        df["information"] = "passage: " + df["information"]
    return df


def process_inference_data(df):
    df["information"] = df[["question", "choices"]].apply(
        lambda x: "\n".join(x.dropna().astype(str).values), axis=1
    )


def embedding_text(tokenizer=None, model=None, input_texts=None):
    batch_dict = tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1).detach().numpy()
    return embeddings


def embedding_query_text(tokenizer=None, model=None, query_text=None):
    batch_dict = tokenizer(
        query_text, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    outputs = model(**batch_dict)
    embeddings = (
        average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        .detach()
        .numpy()
    )
    return embeddings


def get_relevance_embeddings(embeddings, query_embedding):
    scores = (query_embedding @ embeddings.T) * 100
    # import numpy as np

    # index_of_max_value = np.argmax(scores)
    # matched_text = input_texts[index_of_max_value]
    # matched_text
    return scores


def get_relevance_texts(input_texts=None, scores=None, top_k=3):
    scores = scores[0]
    top_k_indices = np.argsort(scores)[-top_k:]
    matched_text = [input_texts[index] for index in top_k_indices]
    matched_text
    return matched_text
