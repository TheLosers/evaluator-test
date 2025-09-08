from typing import List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .metrics import MetricRegistry

app = FastAPI()


class EvaluationRequest(BaseModel):
    candidate: str
    reference: str
    metrics: List[str] = ["bertscore", "summac"]


@app.post("/evaluate")
def evaluate(request: EvaluationRequest) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for name in request.metrics:
        try:
            metric = MetricRegistry.get(name)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Metric '{name}' is not available")
        try:
            results[name] = metric.evaluate(request.candidate, request.reference)
        except ImportError as e:
            raise HTTPException(status_code=500, detail=str(e))
    return results
