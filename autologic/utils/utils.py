import statistics
import math

def format_mean_std(values: list[float]) -> str:
    if not values:
        return "N/A"
    cleaned = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            cleaned.append(fv)
    if not cleaned:
        return "N/A"
    mean = statistics.mean(cleaned)
    if len(cleaned) < 2:
        return f"{mean:.2f}±0.00"
    std_dev = statistics.stdev(cleaned)
    return f"{mean:.2f}±{std_dev:.2f}"



def format_mean_std_four(values: list[float]) -> str:
    if not values:
        return "N/A"
    cleaned = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            cleaned.append(fv)
    if not cleaned:
        return "N/A"
    mean = statistics.mean(cleaned)
    if len(cleaned) < 2:
        return f"{mean:.4f}±0.0000"
    std_dev = statistics.stdev(cleaned)
    return f"{mean:.4f}±{std_dev:.4f}"
