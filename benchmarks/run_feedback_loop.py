#!/usr/bin/env python3
"""Run the in-process sensor, fusion, and timing loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.feedback_loop import main


if __name__ == "__main__":
    main()
