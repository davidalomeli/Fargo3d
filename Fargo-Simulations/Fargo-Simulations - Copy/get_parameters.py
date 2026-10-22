import os
from functools import lru_cache


@lru_cache(maxsize=None)
def _parse(folder, mtime_ns):
    params = {}
    with open(f'{folder}variables.par', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                params[key.strip()] = value.strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    params[parts[0].strip()] = parts[1].strip()
    return params


def read_par(folder):
    """Returns .par file info. Parsed once per (folder, file modification time)."""
    return dict(_parse(folder, os.stat(f'{folder}variables.par').st_mtime_ns))
