from .base import Metric, MetricRegistry


class SummaCMetric(Metric):
    name = "summac"

    def __init__(self) -> None:
        self._model = None

    def evaluate(self, prediction: str, reference: str) -> float:
        try:
            from summac.model_summac import SummaCZS
        except ImportError as e:
            raise ImportError("summac library is not installed") from e

        if self._model is None:
            self._model = SummaCZS(granularity="sentence", base_model="vitc")

        scores = self._model.score([reference], [prediction])
        return float(scores["scores"][0])


MetricRegistry.register(SummaCMetric())
