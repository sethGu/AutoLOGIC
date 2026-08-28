import statistics


def format_mean_std(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values) if len(values) >= 2 else 0.0
    return f"{mean:.2f}±{std_dev:.2f}"


def format_mean_std_four(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values) if len(values) >= 2 else 0.0
    return f"{mean:.4f}±{std_dev:.4f}"
