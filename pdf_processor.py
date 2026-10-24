from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


class PDFProcessor:
    def __init__(self, min_chunk_length: int = 20):
        self.min_chunk_length = min_chunk_length

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def extract_chunks(self, pdf_path: str) -> List[Dict[str, Any]]:
        result = self.converter.convert(pdf_path)
        stem = Path(pdf_path).stem
        source = Path(pdf_path).name
        chunks = []

        try:
            paragraphs = result.document.export_to_markdown().split("\n\n")
            method = "markdown"
        except Exception:
            paragraphs = [
                elem.text for elem in result.document.iterate_items()
                if hasattr(elem, "text")
            ]
            method = "iterate"

        for i, para in enumerate(paragraphs):
            text = para.strip()
            if len(text) >= self.min_chunk_length:
                chunks.append({
                    "id": f"{stem}_chunk_{i}",
                    "text": text,
                    "source": source,
                    "chunk_index": i,
                    "metadata": {"length": len(text), "method": method},
                })

        return chunks

    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {"total_chunks": 0, "total_chars": 0, "avg_chunk_size": 0}
        sizes = [c["metadata"]["length"] for c in chunks]
        return {
            "total_chunks": len(chunks),
            "total_chars": sum(sizes),
            "avg_chunk_size": round(sum(sizes) / len(sizes), 2),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
        }
