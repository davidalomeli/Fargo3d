import os
import numpy as np


class SimulationDataLoader:
    """Loads raw binary output files produced by FARGO3D."""

    def __init__(self, path):
        self.path = path

        self.phi_grid = np.loadtxt(os.path.join(path, "domain_x.dat"))[:-1]
        r_grid = np.loadtxt(os.path.join(path, "domain_y.dat"))
        r_grid = (r_grid[1:] + r_grid[:-1]) / 2
        self.r_grid   = r_grid[3:-3]

        self.nr   = len(self.r_grid)
        self.nphi = len(self.phi_grid)
        self._raw = {}   # filename -> (mtime_ns, size, array); validated on every access

        PHI, R   = np.meshgrid(self.phi_grid, self.r_grid, indexing="xy")
        self.X   = np.cos(PHI) * R
        self.Y   = np.sin(PHI) * R
        self.R   = R
        self.PHI = PHI

    def _read(self, fname):
        """np.fromfile + reshape, cached per file and re-read if the file changes on disk.
        Returned arrays are read-only (shared)."""
        full = os.path.join(self.path, fname)
        st = os.stat(full)                       # raises FileNotFoundError like np.fromfile
        hit = self._raw.get(fname)
        if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
            return hit[2]
        arr = np.fromfile(full).reshape(self.nr, self.nphi)
        arr.flags.writeable = False
        if len(self._raw) >= 32:                 # small bound: ~32 snapshots-fields in RAM
            self._raw.pop(next(iter(self._raw)))
        self._raw[fname] = (st.st_mtime_ns, st.st_size, arr)
        return arr

    def clear_cache(self):
        self._raw.clear()

    # ── gas ──────────────────────────────────────────────────────────────
    def load_gasdens(self, orbit):
        return self._read(f"gasdens{orbit}.dat")

    def load_gasvel(self, orbit):
        vphi_faces = self._read(f"gasvx{orbit}.dat")
        vr_faces   = self._read(f"gasvy{orbit}.dat")

        # Azimuthal velocity: staggered in phi. Wrap column 0 as the upper face of the last cell (periodic).
        vphi_ext = np.concatenate((vphi_faces, vphi_faces[:, :1]), axis=1)   # (nr, nphi+1)
        vphi     = 0.5 * (vphi_ext[:, :-1] + vphi_ext[:, 1:])                # (nr, nphi)

        # Radial velocity: staggered in r. Duplicate the last row as the missing outer face.
        vr_ext = np.concatenate((vr_faces, vr_faces[-1:, :]), axis=0)        # (nr+1, nphi)
        vr     = 0.5 * (vr_ext[:-1, :] + vr_ext[1:, :])                      # (nr, nphi)

        return vr, vphi
    # ── dust ─────────────────────────────────────────────────────────────
    def load_dustdens(self, orbit, dust_num):
        try:
            return self._read(f"dustdens{orbit}.dat")
        except FileNotFoundError:
            return self._read(f"dust{dust_num}dens{orbit}.dat")

    def load_dustvel(self, orbit, dust_num):
        try:
            vr_faces   = self._read(f"dustvy{orbit}.dat")
            vphi_faces = self._read(f"dustvx{orbit}.dat")
        except FileNotFoundError:
            vr_faces   = self._read(f"dust{dust_num}vy{orbit}.dat")
            vphi_faces = self._read(f"dust{dust_num}vx{orbit}.dat")

        vphi_ext = np.concatenate((vphi_faces, vphi_faces[:, :1]), axis=1)
        vphi     = 0.5 * (vphi_ext[:, :-1] + vphi_ext[:, 1:])

        vr_ext = np.concatenate((vr_faces, vr_faces[-1:, :]), axis=0)
        vr     = 0.5 * (vr_ext[:-1, :] + vr_ext[1:, :])

        return vr, vphi