#!/usr/bin/env python3
"""Smoke test for src/core/multimodal.py

Instantiates MultimodalDangerousEventRecognizer and runs process_frame on a
blank image to verify imports and basic processing pipeline don't crash.
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
try:
    from src.core.multimodal import MultimodalDangerousEventRecognizer
except Exception:
    from core.multimodal import MultimodalDangerousEventRecognizer

def main():
    print('Running multimodal smoke test')
    det = MultimodalDangerousEventRecognizer(experiment_id='E1_SMOKE')
    # create a blank RGB image 720x1280
    img = (np.zeros((720, 1280, 3), dtype=np.uint8) + 127)
    out = det.process_frame(img)
    print('process_frame output shapes:', [len(x) if hasattr(x, '__len__') else type(x) for x in out])
    print('Smoke test completed successfully')

if __name__ == '__main__':
    main()
