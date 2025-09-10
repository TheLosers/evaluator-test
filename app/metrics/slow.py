import time
from .base import Metric, MetricRegistry


class SlowMetric(Metric):
    """Metric that simulates a long-running computation."""

    name = "slow"

    def evaluate(self, prediction: str, reference: str) -> float:
        # Simulate a heavy computation that takes 5 seconds
        time.sleep(5)
        return 0.0


MetricRegistry.register(SlowMetric())
