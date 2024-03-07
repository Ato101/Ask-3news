from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import requests
from schemes import Record
import pandas as pd
import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
import openai
from qdrant_client.models import VectorParams, Distance
from config import Setting

settings = Setting()
qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
app = FastAPI()
encoder = SentenceTransformer(
    "all-MiniLM-L6-v2",
)




def generate_embeddings():
    dataset = {'title': [], 'article': [], 'url': []}

    for i in range(1,201):
        url = f"https://3news.com/wp-json/wp/v2/posts?per_page=100&offset={i*100}"
        response = requests.get(url)

        if response.status_code == 200:
            news_data = response.json()
            print(len(news_data))

            articles = news_data

            for article in articles:
                dataset['title'].append(article['title']['rendered'])
                dataset['article'].append(article['content']['rendered'])
                dataset['url'].append(article['link'])

    df = pd.DataFrame(data=dataset)
    df = df[df['article'].str.split().apply(len) > 15]

    model = SentenceTransformer("paraphrase-mpnet-base-v2", device="cuda" if torch.cuda.is_available() else "cpu")

    vectors = []
    batch_size = 512
    batch = []

    for doc in df['article'].to_list():
        batch.append(doc)

        if len(batch) >= batch_size:
            vectors.append(model.encode(batch))
            batch = []

    if len(batch) > 0:
        vectors.append(model.encode(batch))
        batch = []

    vectors = np.concatenate(vectors)

    records = []

    for idx, (title, url, article) in enumerate(zip(df['title'], df['url'], df['article'])):
        try:
            article_embedding = model.encode(article).tolist()
            title_embedding = model.encode(title).tolist()
            record = Record(id=idx, vector=article_embedding, vector1=title_embedding, payload={'title': title, 'url': url})
            records.append(record)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")

    return records

def upload_embeddings(records):
    # Assuming QDRANT_URL, QDRANT_API_KEY, and models are defined elsewhere
    qdrant = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.openai_api_key,
    )
    encoder = SentenceTransformer("paraphrase-mpnet-base-v2")

    qdrant.recreate_collection(
        collection_name="news",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )

    qdrant.upload_records(
        collection_name="news",
        records=records  # corrected here
    )


# Example usage:
# records = generate_embeddings()
# upload_embeddings(records)

if __name__ == '__main__':
    records = generate_embeddings()
    upload_embeddings(records)

