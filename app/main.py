from typing import List, Dict
import asyncio
import logging
import os

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from .metrics import MetricRegistry

app = FastAPI()


METRIC_TIMEOUT = float(os.getenv("METRIC_TIMEOUT", "10"))


@app.middleware("http")
async def cancel_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except asyncio.CancelledError:
        logging.warning("Request cancelled by client")
        return Response(status_code=499)


class EvaluationRequest(BaseModel):
    candidate: str
    reference: str
    metrics: List[str] = ["bertscore", "summac"]


@app.post("/evaluate")
async def evaluate(request: Request, payload: EvaluationRequest) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for name in payload.metrics:
        try:
            metric = MetricRegistry.get(name)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Metric '{name}' is not available")
        try:
            results[name] = await asyncio.wait_for(
                asyncio.to_thread(metric.evaluate, payload.candidate, payload.reference),
                timeout=METRIC_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"Metric '{name}' evaluation timed out")
        except asyncio.CancelledError:
            logging.info("Metric evaluation cancelled")
            raise
        except ImportError as e:
            raise HTTPException(status_code=500, detail=str(e))
        if await request.is_disconnected():
            logging.info("Client disconnected during evaluation")
            raise asyncio.CancelledError()
    return results
