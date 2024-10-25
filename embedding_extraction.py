import os
import pickle
import time

import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import (AutoImageProcessor, AutoModel, AutoTokenizer, BlipForConditionalGeneration,
                          DetrForObjectDetection)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_OpenAI_embeddings(texts, model="text-embedding-3-small"):
    texts = [text.replace("\n", " ") for text in texts]  # Clean up text
    response = client.embeddings.create(input=texts, model=model)
    return [res_data.embedding for res_data in response.data]


def batched_embeddings(texts, batch_size=16, model="text-embedding-3-small"):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = get_OpenAI_embeddings(batch_texts, model=model)
        embeddings.extend(batch_embeddings)
        time.sleep(0.75)  # Prevent making too many requests too fast
    return torch.tensor(embeddings)


def extract_embeddings(text, foldername):
    filename = "embed_data"

    filename += "-gpt4"

    try:
        with open(f"{foldername}/{filename}.pickle", 'rb') as handle:
            obj = pickle.load(handle)
            embeddings = obj["embeddings"]
            del obj

            print(
                f"File '{foldername}/{filename}.pickle' loaded successfully.")
    except FileNotFoundError:
        print(
            f"Could not find file '{foldername}/{filename}.pickle'."
            "Regenerating the embeddings."
        )

        embeddings = batched_embeddings(text, batch_size=16)

        with open(f"{foldername}/{filename}.pickle", 'wb') as handle:
            pickle.dump(
                obj={
                    "embeddings": embeddings,
                },
                file=handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    return embeddings
