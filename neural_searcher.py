from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import openai
from config import settings
openai.api_key = settings.openai_api_key

class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer("paraphrase-mpnet-base-v2", device="cpu")
        # initialize Qdrant client
        self.qdrant = QdrantClient(url=settings.qdrant_url,api_key= settings.qdrant_api_key)

    def build_prompt(self, question: str, references: list) -> tuple[str, str]:
        prompt = f"""
        You are a News bot who has been asked this : '{question}'

        You've selected the most relevant content from your post  to use as source for your answer. Cite them in your answer.

        References:
        """.strip()

        references_links = ""

        for i, reference in enumerate(references, start=1):
            links = reference.payload["url"].strip()
            references_links += f"\n[{i}]:{links}"

        prompt += (
                references_links
                + """How to cite a reference: This is a citation [1].
                 This one too [2].
                 And this is sentence with many citations [3].
                 Answer:
            """
        )
        return prompt, references_links

    def ask(self, question: str):
        retrieval_model = SentenceTransformer("paraphrase-mpnet-base-v2")
        similar_content = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=retrieval_model.encode(question),
            limit=7,
            append_payload=True,
        )

        prompt, references = self.build_prompt(question, similar_content)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
            temperature=0.6,
        )

        return {
            "response": response["choices"][0]["message"]["content"],
            "references":references,
        }


