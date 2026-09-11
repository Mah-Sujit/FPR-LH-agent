"""Small statistical helpers with no third-party dependency."""

from __future__ import annotations

import math


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def calibration_rows(pairs: list[tuple[float, bool]], bins: int = 5) -> list[dict[str, float | int]]:
    rows = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [(risk, actual) for risk, actual in pairs
                    if lower <= risk < upper or (index == bins - 1 and risk == upper)]
        if selected:
            rows.append({"bin_lower": lower, "bin_upper": upper, "count": len(selected),
                         "mean_risk": sum(item[0] for item in selected) / len(selected),
                         "observed_failure_rate": sum(item[1] for item in selected) / len(selected)})
    return rows

