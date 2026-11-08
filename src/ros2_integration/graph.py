"""
In-process publish/subscribe graph with ROS 2 QoS semantics we actually enforce.

``KEEP_LAST`` uses a bounded deque and counts dropped samples. Callbacks do not
run inference. A subscriber hands the sample to ``TimestampSynchronizer`` or a
bounded worker queue.
"""

from collections import deque
from typing import Deque, Dict

from .messages import TOPIC_QOS, SensorSample


class InProcessGraph:
    """Topic transport for the software-in-the-loop loop."""

    def __init__(self):
        self._queues: Dict[str, Deque[SensorSample]] = {}
        self.published: Dict[str, int] = {}
        self.dropped: Dict[str, int] = {}

    def publish(self, sample: SensorSample) -> None:
        qos = TOPIC_QOS[sample.topic]
        queue = self._queues.get(sample.topic)
        if queue is None:
            queue = deque(maxlen=qos.depth if qos.history == "KEEP_LAST" else None)
            self._queues[sample.topic] = queue
        if qos.history == "KEEP_LAST" and queue.maxlen is not None and len(queue) == queue.maxlen:
            self.dropped[sample.topic] = self.dropped.get(sample.topic, 0) + 1
        queue.append(sample)
        self.published[sample.topic] = self.published.get(sample.topic, 0) + 1

    def take(self, topic: str) -> list:
        queue = self._queues.get(topic)
        if not queue:
            return []
        items = list(queue)
        queue.clear()
        return items

    def depth(self, topic: str) -> int:
        queue = self._queues.get(topic)
        return 0 if queue is None else len(queue)
