"""
RAGAS-based validation for the GraphRAG pipeline.

Metrics used:
  - Faithfulness      : Is the answer factually grounded in the retrieved context?
  - Answer Relevancy  : Does the answer actually address the question?
  - Context Precision : Are the top-ranked chunks the most relevant ones?  (requires reference)
  - Context Recall    : Does the retrieved context cover the ground-truth answer? (requires reference)

Usage
-----
  from validator import RAGValidator
  v = RAGValidator(api_key="...", llm_model="gemini-2.0-flash")

  # single evaluation (no ground truth)
  scores = v.validate(query="...", answer="...", contexts=["chunk1", "chunk2"])

  # with optional ground truth for precision/recall
  scores = v.validate(query="...", answer="...", contexts=[...], reference="expected answer")

  # batch evaluation
  results = v.validate_batch([
      {"query": "...", "answer": "...", "contexts": [...], "reference": "..."},
      ...
  ])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness


class RAGValidator:
    """Evaluates RAG outputs with RAGAS metrics backed by Gemini."""

    def __init__(
        self,
        api_key: str,
        llm_model: str = "gemini-2.0-flash",
        embedding_model: str = "models/text-embedding-004",
    ):
        llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key)
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model, google_api_key=api_key
        )
        wrapped_llm = LangchainLLMWrapper(llm)
        wrapped_emb = LangchainEmbeddingsWrapper(embeddings)

        # Metrics that work without a reference answer
        self._base_metrics = [
            Faithfulness(llm=wrapped_llm),
            AnswerRelevancy(llm=wrapped_llm, embeddings=wrapped_emb),
        ]
        # Metrics that require a reference answer
        self._reference_metrics = [
            ContextPrecision(llm=wrapped_llm),
            ContextRecall(llm=wrapped_llm),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single query/answer/context triple.

        Parameters
        ----------
        query     : The user question.
        answer    : The generated answer.
        contexts  : List of retrieved context strings.
        reference : Optional ground-truth answer (enables Precision & Recall).

        Returns
        -------
        Dict with metric names as keys and float scores (0-1) as values,
        plus an 'interpretation' sub-dict with plain-English labels.
        """
        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference,
        )
        metrics = self._select_metrics(reference)
        dataset = EvaluationDataset(samples=[sample])
        result_df = evaluate(dataset=dataset, metrics=metrics).to_pandas()

        scores = self._extract_scores(result_df.iloc[0].to_dict())
        return {**scores, "interpretation": self._interpret(scores)}

    def validate_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a list of samples.

        Each element must have keys: query, answer, contexts.
        Optionally: reference (str).

        Returns
        -------
        {
          "per_sample": [list of per-row score dicts],
          "averages":   {metric: mean_score},
        }
        """
        has_reference = any(s.get("reference") for s in samples)
        metrics = self._select_metrics("placeholder" if has_reference else None)

        ragas_samples = [
            SingleTurnSample(
                user_input=s["query"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
                reference=s.get("reference"),
            )
            for s in samples
        ]
        dataset = EvaluationDataset(samples=ragas_samples)
        result_df = evaluate(dataset=dataset, metrics=metrics).to_pandas()

        score_cols = [
            c for c in result_df.columns
            if c not in ("user_input", "response", "retrieved_contexts", "reference")
        ]
        per_sample = [
            {col: float(row[col]) for col in score_cols if col in row}
            for _, row in result_df.iterrows()
        ]
        averages = {col: float(result_df[col].mean()) for col in score_cols}

        return {"per_sample": per_sample, "averages": averages}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_metrics(self, reference: Optional[str]):
        if reference:
            return self._base_metrics + self._reference_metrics
        return self._base_metrics

    @staticmethod
    def _extract_scores(row: Dict[str, Any]) -> Dict[str, float]:
        skip = {"user_input", "response", "retrieved_contexts", "reference"}
        return {k: float(v) for k, v in row.items() if k not in skip}

    @staticmethod
    def _interpret(scores: Dict[str, float]) -> Dict[str, str]:
        """Map numeric scores to human-readable quality labels."""
        def label(v: float) -> str:
            if v >= 0.8:
                return "Excellent"
            if v >= 0.6:
                return "Good"
            if v >= 0.4:
                return "Fair"
            return "Poor"

        return {k: label(v) for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Convenience wrapper used by app.py
# ---------------------------------------------------------------------------

def format_validation_report(result: Dict[str, Any]) -> str:
    """Return a Markdown-formatted string of a validate() result."""
    lines = ["### RAGAS Validation Report", ""]
    interp = result.pop("interpretation", {})
    for metric, score in result.items():
        bar = _score_bar(score)
        tag = interp.get(metric, "")
        lines.append(f"**{metric.replace('_', ' ').title()}**: {score:.3f} {bar} — {tag}")
    result["interpretation"] = interp  # restore
    return "\n".join(lines)


def _score_bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"
