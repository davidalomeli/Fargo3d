"""Synthetic FARGO3D-style output: log-spaced radial grid with 3 ghost cells each side,
mildly eccentric Keplerian streamlines, spiral density perturbation."""
import os, numpy as np, sys
out = sys.argv[1]; nr, nphi = int(sys.argv[2]), int(sys.argv[3])
os.makedirs(out, exist_ok=True)
redges = np.logspace(np.log10(0.25), np.log10(30), nr + 7)
pedges = np.linspace(-np.pi, np.pi, nphi + 1)
np.savetxt(f"{out}/domain_y.dat", redges)
np.savetxt(f"{out}/domain_x.dat", pedges)
rc = 0.5 * (redges[1:] + redges[:-1])[3:-3]
phi = pedges[:-1]
R, P = np.meshgrid(rc, phi, indexing="ij")
rng = np.random.default_rng(0)
for orbit in (19, 20):
    t = 0.05 * orbit
    Sigma = 1e-4 / R * (1 + 0.15 * np.cos(P - 0.5 - 0.3 * R + t)) * np.exp(-(R / 25) ** 2)
    vphi = R ** -0.5 * (1 + 0.02 * np.sin(P + t))
    vr = 0.05 * R ** -0.5 * np.sin(P - 0.3 + t) * np.exp(-((R - 4) / 8) ** 2) + 1e-4 * rng.standard_normal(R.shape)
    dvr = vr - 2e-3 * R ** 0.5
    dvphi = vphi * (1 - 1e-3)
    for name, arr in [(f"gasdens{orbit}", Sigma), (f"gasvx{orbit}", vphi), (f"gasvy{orbit}", vr),
                      (f"dust1dens{orbit}", 0.01 * Sigma), (f"dust1vx{orbit}", dvphi), (f"dust1vy{orbit}", dvr)]:
        arr.astype(np.float64).tofile(f"{out}/{name}.dat")
open(f"{out}/variables.par", "w").write(
    "ASPECTRATIO 0.05\nPARTICLESIZE1 1.0\nRHOSOLID 2.0\nINVSTOKES1 10.0\nTS 0.1\nEPSILON 0.01\n")
print("wrote", out, "nr", nr, "nphi", nphi)
