from typing import List, Dict, Any
from google import genai
from embeddings import _retry_with_backoff


class ResponseGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash",
        max_retries: int = 3,
    ):
        self.max_retries = max_retries
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> str:
        prompt = self._build_prompt(query, context_chunks)

        def request():
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            return response.text

        return _retry_with_backoff(request, self.max_retries)

    def _build_prompt(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        context = "\n\n".join(
            f"[Source: {c['source']}, Chunk {c['chunk_index']}]\n{c['text']}"
            for c in chunks
        ) or "No relevant context found."

        return f"""You are a helpful AI assistant.
Answer using ONLY the context below.
Cite the source(s) you used.
If the context is insufficient, say so clearly.

Context:
{context}

Question: {query}

Answer:"""