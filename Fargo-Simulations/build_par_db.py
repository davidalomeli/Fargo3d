"""
Index every variables.par under a root directory into a SQLite database.

Usage:
    python build_par_db.py /path/to/fargo  sims.db

Then query it (Python, DB Browser for SQLite, or the sqlite3 CLI):
    SELECT path FROM sims WHERE ASPECTRATIO = 0.05 AND EPSILON > 0.01;

Re-run the script any time to rebuild the database from scratch.
"""
import os
import sqlite3
import sys


def parse_par(filepath):
    """Parse a FARGO3D .par file into {KEY: value}. Numbers become floats, the rest stay strings."""
    params = {}
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()      # drop comments
            if not line:
                continue
            parts = line.split(None, 1)
            key = parts[0].upper()
            value = parts[1].strip() if len(parts) > 1 else ""
            params[key] = to_value(value)
    return params


def to_value(s):
    try:
        return float(s)
    except ValueError:
        return s


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def build_db(root, db_path, filename="variables.par"):
    found = []
    for dirpath, _, files in os.walk(root):
        if filename in files:
            folder = os.path.abspath(dirpath)
            found.append((folder, parse_par(os.path.join(folder, filename))))

    if not found:
        print(f"No {filename} found under {root}")
        return

    # one column per parameter name seen in any simulation (missing -> NULL)
    keys = sorted({k for _, p in found for k in p})
    keys = [k for k in keys if k not in ("PATH", "NAME")]

    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS sims")
    # no declared type on the parameter columns, so numbers and text both store naturally
    cols = ", ".join(quote(k) for k in keys)
    con.execute(f"CREATE TABLE sims (path TEXT PRIMARY KEY, name TEXT, {cols})")

    placeholders = ", ".join("?" * (2 + len(keys)))
    for folder, p in found:
        row = [folder, os.path.basename(folder)] + [p.get(k) for k in keys]
        con.execute(f"INSERT INTO sims VALUES ({placeholders})", row)

    con.commit()
    con.close()
    print(f"Indexed {len(found)} simulations, {len(keys)} parameters -> {db_path}")


def find_sims(db_path, where):
    """Return the paths of simulations matching a SQL WHERE clause, e.g. 'ASPECTRATIO = 0.05'."""
    con = sqlite3.connect(db_path)
    rows = con.execute(f"SELECT path FROM sims WHERE {where} ORDER BY path").fetchall()
    con.close()
    return [r[0] for r in rows]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    build_db(sys.argv[1], sys.argv[2])
