import contextlib
import signal
from typing import Iterator, Optional


class TimeoutExceeded(TimeoutError):
    pass


@contextlib.contextmanager
def time_limit(seconds: int, *, enabled: bool = True) -> Iterator[None]:
    if not enabled or seconds is None:
        yield
        return

    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    try:
        old_handler = signal.getsignal(signal.SIGALRM)
    except Exception:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutExceeded(f"Time limit exceeded: {seconds}s")

    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass

