from __future__ import annotations

from sentence_transformers import CrossEncoder


class RecommendationRanker:
    """Cross-encoder reranker that boosts semantic precision for recommendations."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _model_instance(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, candidate_text: str) -> float:
        if not query or not candidate_text:
            return 0.0
        model = self._model_instance()
        prediction = model.predict([(query, candidate_text)], convert_to_numpy=True)[0]
        # Clamp score to [0, 1] for consistent weighting
        return max(0.0, min(1.0, float(prediction)))
