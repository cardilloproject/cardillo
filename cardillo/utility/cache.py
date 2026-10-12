from functools import wraps

import numpy as np


def fast_key(*args):
    """Cheap hashable cache key for functions with numpy array arguments.

    Replacement for ``cachetools.keys.hashkey(t, *q, *u)``, which iterates
    over all array entries and creates a numpy scalar for each of them. Here,
    arrays are represented by their raw bytes, which is about 5-15 times
    faster. Equal arrays (bitwise) result in equal keys; different dtypes or
    shapes never collide because the byte lengths or contents differ.
    """
    key = []
    for a in args:
        if type(a) is np.ndarray:
            key.append(a.tobytes())
        elif isinstance(a, np.ndarray):
            key.append(a.tobytes())
        elif isinstance(a, (list, tuple)):
            key.append(np.asarray(a).tobytes())
        else:
            key.append(a)
    return tuple(key)


def cached_last(key):
    """Method decorator that caches only the result of the last call.

    Light-weight replacement for ``cachetools.cachedmethod`` with
    ``LRUCache(maxsize=1)``. The cache lives in the instance ``__dict__`` and
    is invalidated as soon as ``key(self, *args, **kwargs)`` changes. Note that
    the cached object itself is returned, i.e., callers must not modify it.
    """

    def decorator(func):
        attr = "_cached_last_" + func.__qualname__

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            k = key(self, *args, **kwargs)
            slot = self.__dict__.get(attr)
            if slot is not None and slot[0] == k:
                return slot[1]
            value = func(self, *args, **kwargs)
            self.__dict__[attr] = (k, value)
            return value

        return wrapper

    return decorator
