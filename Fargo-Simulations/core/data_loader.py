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

        PHI, R   = np.meshgrid(self.phi_grid, self.r_grid, indexing="xy")
        self.X   = np.cos(PHI) * R
        self.Y   = np.sin(PHI) * R
        self.R   = R
        self.PHI = PHI

    # ── gas ──────────────────────────────────────────────────────────────
    def load_gasdens(self, orbit):
        Sigma = np.fromfile(os.path.join(self.path, f"gasdens{orbit}.dat")).reshape(self.nr, self.nphi)

        return Sigma

    def load_gasvel(self, orbit):
        vphi_faces = np.fromfile(os.path.join(self.path, f"gasvx{orbit}.dat")).reshape(self.nr, self.nphi)
        vr_faces   = np.fromfile(os.path.join(self.path, f"gasvy{orbit}.dat")).reshape(self.nr, self.nphi)

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
            return np.fromfile(os.path.join(self.path, f"dustdens{orbit}.dat")).reshape(self.nr, self.nphi)
            
        except:
            return np.fromfile(os.path.join(self.path, f"dust{dust_num}dens{orbit}.dat")).reshape(self.nr, self.nphi)

    def load_dustvel(self, orbit, dust_num):
        try:
            vr_faces   = np.fromfile(os.path.join(self.path, f"dustvy{orbit}.dat")).reshape(self.nr, self.nphi)
            vphi_faces = np.fromfile(os.path.join(self.path, f"dustvx{orbit}.dat")).reshape(self.nr, self.nphi)
        except:
            vr_faces   = np.fromfile(os.path.join(self.path, f"dust{dust_num}vy{orbit}.dat")).reshape(self.nr, self.nphi)
            vphi_faces = np.fromfile(os.path.join(self.path, f"dust{dust_num}vx{orbit}.dat")).reshape(self.nr, self.nphi)

        vphi_ext = np.concatenate((vphi_faces, vphi_faces[:, :1]), axis=1)
        vphi     = 0.5 * (vphi_ext[:, :-1] + vphi_ext[:, 1:])

        vr_ext = np.concatenate((vr_faces, vr_faces[-1:, :]), axis=0)
        vr     = 0.5 * (vr_ext[:-1, :] + vr_ext[1:, :])

        return vr, vphi