"""
Align sensor samples by acquisition time.

Fusing whatever arrived most recently puts a moving object in two places.
This synchronizer picks, for one anchor stamp, the closest sample of each
other modality and rejects samples outside a bounded slop.
"""

from typing import Dict, List, Optional

from .messages import SensorSample


class TimestampSynchronizer:
    """Bounded, acquisition-time association across modalities."""

    def __init__(self, modalities: List[str], slop_ns: int, anchor: str = "lidar"):
        if anchor not in modalities:
            raise ValueError(f"anchor {anchor} is not in {modalities}")
        self.modalities = list(modalities)
        self.slop_ns = slop_ns
        self.anchor = anchor
        self._buffers: Dict[str, List[SensorSample]] = {name: [] for name in modalities}
        self.rejected = 0

    def push(self, sample: SensorSample) -> None:
        if sample.modality not in self._buffers:
            raise KeyError(sample.modality)
        bucket = self._buffers[sample.modality]
        bucket.append(sample)
        if len(bucket) > 64:
            bucket.pop(0)

    def pop_aligned(self) -> Optional[Dict[str, Optional[SensorSample]]]:
        """
        Return one group keyed by modality.

        A modality outside the slop is ``None`` and its sample stays buffered.
        Returns ``None`` when the anchor modality has no sample yet.
        """
        anchor_buf = self._buffers[self.anchor]
        if not anchor_buf:
            return None
        anchor = anchor_buf.pop(0)
        group: Dict[str, Optional[SensorSample]] = {self.anchor: anchor}
        for name in self.modalities:
            if name == self.anchor:
                continue
            match = self._take_closest(name, anchor.stamp_ns)
            if match is None:
                self.rejected += 1
            group[name] = match
        return group

    def _take_closest(self, modality: str, stamp_ns: int) -> Optional[SensorSample]:
        bucket = self._buffers[modality]
        if not bucket:
            return None
        best_i = min(range(len(bucket)), key=lambda i: abs(bucket[i].stamp_ns - stamp_ns))
        best = bucket[best_i]
        if abs(best.stamp_ns - stamp_ns) > self.slop_ns:
            return None
        return bucket.pop(best_i)
