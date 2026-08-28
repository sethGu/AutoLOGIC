import statistics as _statistics


_orig_stdev = _statistics.stdev


def _safe_stdev(data, *args, **kwargs):
    try:
        if len(data) < 2:
            return 0.0
    except Exception:
        pass
    return _orig_stdev(data, *args, **kwargs)


_statistics.stdev = _safe_stdev
