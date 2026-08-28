import numpy as np


def ece_binary(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    y_prob = np.clip(y_prob, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    ece = 0.0
    n = y_true.size
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        w = float(np.sum(mask)) / float(n)
        ece += w * abs(acc - conf)
    return float(ece)

