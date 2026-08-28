import contextlib


class TimeoutExceeded(TimeoutError):
    pass


@contextlib.contextmanager
def time_limit(seconds, *, enabled=True):
    yield
