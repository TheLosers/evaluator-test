from typing import List, Dict
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deep_translator import GoogleTranslator

from .metrics import MetricRegistry

app = FastAPI()


_translator = GoogleTranslator(source="auto", target="ko")


def _maybe_translate(text: str) -> str:
    """Translate English text to Korean before evaluation."""
    if re.search(r"[A-Za-z]", text):
        try:
            return _translator.translate(text)
        except Exception:
            return text
    return text


class EvaluationRequest(BaseModel):
    candidate: str
    reference: str
    metrics: List[str] = ["bertscore", "summac"]


@app.post("/evaluate")
def evaluate(request: EvaluationRequest) -> Dict[str, float]:
    candidate = _maybe_translate(request.candidate)
    reference = _maybe_translate(request.reference)
    results: Dict[str, float] = {}
    for name in request.metrics:
        try:
            metric = MetricRegistry.get(name)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Metric '{name}' is not available")
        try:
            results[name] = metric.evaluate(candidate, reference)
        except ImportError as e:
            raise HTTPException(status_code=500, detail=str(e))
    return results
