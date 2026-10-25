
from pathlib import Path
from typing import List, Dict, Any

import pymupdf


class PDFProcessor:
    def __init__(self, min_chunk_length: int = 20):
        self.min_chunk_length = min_chunk_length

    def extract_chunks(self, pdf_path: str) -> List[Dict[str, Any]]:
        path = Path(pdf_path)
        source = path.name
        stem = path.stem
        chunks = []

        # Open the PDF and extract text from each page
        with pymupdf.open(str(path)) as document:
            chunk_index = 0

            for page_number, page in enumerate(document):
                page_text = page.get_text("text")

                # Split page text into paragraphs
                paragraphs = page_text.split("\n\n")

                for paragraph in paragraphs:
                    text = " ".join(paragraph.split())

                    if len(text) >= self.min_chunk_length:
                        chunks.append({
                            "id": f"{stem}_chunk_{chunk_index}",
                            "text": text,
                            "source": source,
                            "chunk_index": chunk_index,
                            "metadata": {
                                "length": len(text),
                                "method": "pymupdf",
                                "page_number": page_number + 1,
                            },
                        })

                        chunk_index += 1

        return chunks

    def get_chunk_statistics(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        if not chunks:
            return {
                "total_chunks": 0,
                "total_chars": 0,
                "avg_chunk_size": 0,
            }

        sizes = [chunk["metadata"]["length"] for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_chars": sum(sizes),
            "avg_chunk_size": round(sum(sizes) / len(sizes), 2),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
        }
