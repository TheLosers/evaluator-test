from abc import ABC, abstractmethod
from typing import Dict


class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    name: str

    @abstractmethod
    def evaluate(self, prediction: str, reference: str) -> float:
        """Compute a score for a prediction/reference pair."""


class MetricRegistry:
    """Registry to keep track of available metrics."""

    _metrics: Dict[str, Metric] = {}

    @classmethod
    def register(cls, metric: Metric) -> None:
        cls._metrics[metric.name] = metric

    @classmethod
    def get(cls, name: str) -> Metric:
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' is not registered")
        return cls._metrics[name]

    @classmethod
    def available(cls) -> Dict[str, Metric]:
        return dict(cls._metrics)
