from __future__ import annotations

from typing import Dict, List

import numpy as np


def build_calibration_table(raw_scores: Dict[str, List[float]]) -> Dict[str, Dict]:
    calibration = {}

    for method, scores in raw_scores.items():
        scores = np.array(scores)

        calibration[method] = {
            "p10": np.percentile(scores, 10),
            "p25": np.percentile(scores, 25),
            "p50": np.percentile(scores, 50),
            "p75": np.percentile(scores, 75),
            "p90": np.percentile(scores, 90),
        }

    return calibration


def calibrate_score(method: str, raw_score: float, calibration: Dict) -> float:
    c = calibration[method]

    if raw_score <= c["p10"]:
        return 10
    elif raw_score <= c["p25"]:
        return 30
    elif raw_score <= c["p50"]:
        return 50
    elif raw_score <= c["p75"]:
        return 70
    elif raw_score <= c["p90"]:
        return 85
    else:
        return 95