from .base import Metric, MetricRegistry


class BertScoreMetric(Metric):
    name = "bertscore"

    def evaluate(self, prediction: str, reference: str) -> float:
        try:
            from bert_score import score
        except ImportError as e:
            raise ImportError("bertscore library is not installed") from e

        P, R, F1 = score([prediction], [reference], lang="en", model_type="bert-base-uncased")
        return float(F1.mean())


MetricRegistry.register(BertScoreMetric())
