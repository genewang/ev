"""
Constant-velocity tracker for associated detections.

Frame-level boxes are not tracks. This module predicts forward with ego-motion
removed by the caller, associates by a distance gate, and counts births,
confirmations, and losses.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Detection:
    stamp_ns: int
    position: tuple
    source_id: Optional[str] = None


@dataclass
class Track:
    track_id: int
    position: tuple
    velocity: tuple
    stamp_ns: int
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    lost: bool = False


@dataclass
class TrackingStep:
    tracks: List[Track] = field(default_factory=list)
    births: int = 0
    matches: int = 0
    losses: int = 0


class ConstantVelocityTracker:
    """Nearest-neighbor association inside a meter gate."""

    def __init__(self, gate_m: float = 2.0, confirm_hits: int = 2, max_misses: int = 3):
        self.gate_m = gate_m
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self._tracks: List[Track] = []
        self._next_id = 1

    def step(self, detections: List[Detection]) -> TrackingStep:
        births = matches = losses = 0
        unmatched = set(range(len(detections)))
        for track in self._tracks:
            if track.lost:
                continue
            if not unmatched:
                track.misses += 1
                if track.misses > self.max_misses:
                    track.lost = True
                    losses += 1
                continue
            dt_s = 0.0
            if detections:
                dt_s = max(0.0, (detections[0].stamp_ns - track.stamp_ns) / 1e9)
            predicted = tuple(track.position[i] + track.velocity[i] * dt_s for i in range(3))
            best_i = None
            best_d = self.gate_m
            for i in unmatched:
                dist = _distance(predicted, detections[i].position)
                if dist < best_d:
                    best_d = dist
                    best_i = i
            if best_i is None:
                track.misses += 1
                track.position = predicted
                if track.misses > self.max_misses:
                    track.lost = True
                    losses += 1
                continue
            det = detections[best_i]
            unmatched.remove(best_i)
            dt_s = max((det.stamp_ns - track.stamp_ns) / 1e9, 1e-3)
            track.velocity = tuple((det.position[i] - track.position[i]) / dt_s for i in range(3))
            track.position = det.position
            track.stamp_ns = det.stamp_ns
            track.hits += 1
            track.misses = 0
            track.confirmed = track.hits >= self.confirm_hits
            matches += 1
        for i in sorted(unmatched):
            det = detections[i]
            self._tracks.append(
                Track(self._next_id, det.position, (0.0, 0.0, 0.0), det.stamp_ns)
            )
            self._next_id += 1
            births += 1
        live = [track for track in self._tracks if not track.lost]
        return TrackingStep(tracks=live, births=births, matches=matches, losses=losses)


def _distance(a: tuple, b: tuple) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5
