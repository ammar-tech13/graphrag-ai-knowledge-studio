
import time
import random
from typing import List
from google import genai


def _retry_with_backoff(fn, max_retries: int):
    """Retry temporary rate-limit and server errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            error = str(e).lower()
            retryable = (
                "429" in error
                or "503" in error
                or "500" in error
                or "quota" in error
                or "unavailable" in error
            )

            if retryable and attempt < max_retries - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 1))
            else:
                raise


class EmbeddingGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        max_retries: int = 3,
    ):
        self.model = model
        self.max_retries = max_retries
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        text: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[float]:
        def request():
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config={"task_type": task_type},
            )
            return response.embeddings[0].values

        return _retry_with_backoff(request, self.max_retries)

    def generate_document_embedding(self, text: str) -> List[float]:
        return self.generate(text, task_type="RETRIEVAL_DOCUMENT")

    def generate_query_embedding(self, text: str) -> List[float]:
        return self.generate(text, task_type="RETRIEVAL_QUERY")
