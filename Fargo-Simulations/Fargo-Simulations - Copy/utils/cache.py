"""
Content-addressed memoisation for the analysis pipeline.

Keys are hashes of the *actual input arrays*, so a cache entry can never be stale:
if a snapshot file is overwritten (re-running FARGO3D into the same folder), the
arrays change, the hash changes, and the result is recomputed.  Because the store is
module-level, it is shared by every loader / calculator instance in the session, so
re-creating SimulationDataLoader(path) inside a notebook function no longer throws
the work away.

Two byte-budgeted LRU stores:
  BULK     large intermediates (streamline arrays)          default 512 MiB
  RESULTS  small/medium products (fits, a-meshes, Statler)   default 1 GiB

Set FARGO_NO_CACHE=1 in the environment (or cache.set_enabled(False)) to bypass
caching, e.g. to check that cached and uncached results agree.
"""
import hashlib
import os
from collections import OrderedDict

import numpy as np


def content_key(*parts):
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if isinstance(p, np.ndarray):
            a = np.ascontiguousarray(p)
            h.update(f"{a.dtype}{a.shape}".encode())
            h.update(a.tobytes())
        else:
            h.update(repr(p).encode())
        h.update(b"|")
    return h.hexdigest()


def _nbytes(o):
    if isinstance(o, np.ndarray):
        return o.nbytes
    if isinstance(o, (tuple, list)):
        return sum(_nbytes(x) for x in o)
    if isinstance(o, dict):
        return sum(_nbytes(x) for x in o.values())
    return 64


def _freeze(o):
    """Cached arrays are shared, so make them read-only (an accidental in-place edit
    raises instead of silently corrupting later results)."""
    if isinstance(o, np.ndarray):
        o.flags.writeable = False
    elif isinstance(o, (tuple, list)):
        for x in o:
            _freeze(x)
    return o


class ByteLRU:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self._d = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key, fn):
        if not _ENABLED:
            return fn()
        if key in self._d:
            self._d.move_to_end(key)
            self.hits += 1
            return self._d[key][0]
        self.misses += 1
        val = _freeze(fn())
        nb = _nbytes(val)
        if nb <= self.max_bytes:
            self._d[key] = (val, nb)
            self._bytes += nb
            while self._bytes > self.max_bytes and len(self._d) > 1:
                _, (_, b) = self._d.popitem(last=False)
                self._bytes -= b
        return val

    def clear(self):
        self._d.clear()
        self._bytes = 0

    def info(self):
        return dict(entries=len(self._d), MiB=round(self._bytes / 2**20, 1),
                    hits=self.hits, misses=self.misses)


_ENABLED = os.environ.get("FARGO_NO_CACHE", "0") != "1"
BULK = ByteLRU(512 * 2**20)
RESULTS = ByteLRU(1024 * 2**20)


def set_enabled(flag):
    global _ENABLED
    _ENABLED = bool(flag)


def clear_all():
    BULK.clear()
    RESULTS.clear()


def info():
    return {"BULK": BULK.info(), "RESULTS": RESULTS.info()}
