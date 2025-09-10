import os
import sys
from typing import List, Dict, Any

from .base import Metric, MetricRegistry
from .summac_xnli_ko import load_nli, summac_like_score

# CPU 강제 사용
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


class SummaCMetric(Metric):
    name = "summac"

    def __init__(self) -> None:
        self._pipe = None
        self._granularity = "sentence"
        self._alpha = 0.5  # contradiction 페널티 가중치

    def evaluate(self, prediction: str, reference: str) -> float:
        # NLI 모델 로드
        if self._pipe is None:
            try:
                print("Loading NLI model...")
                self._pipe = load_nli("joeddav/xlm-roberta-large-xnli")
                print("NLI model loaded successfully")
            except Exception as e:
                error_msg = str(e)
                print(f"Error loading NLI model: {error_msg}")
                import traceback
                traceback.print_exc()
                raise ImportError(f"transformers library issue: \n{error_msg}") from e
        
        # SummaCZS 평가 수행
        try:
            result = summac_like_score(
                reference, prediction, 
                self._pipe,
                granularity=self._granularity, 
                batch_size=16,
                alpha=self._alpha
            )
            return result["score"]
        except Exception as e:
            print(f"Error in SummaC evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0


MetricRegistry.register(SummaCMetric())
