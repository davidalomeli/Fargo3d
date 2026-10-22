import numpy as np
from scipy.integrate import solve_ivp
from utils.interpolation import interpolate, interpolate_vectorized


class SimulationDataStreamlines:
    """Integrates fluid streamlines using RK4/RK5 in polar coordinates."""

    def __init__(self, data_loader, variables_calculator):
        self.data_loader          = data_loader
        self.variables_calculator = variables_calculator

    # ── single streamline (RK5 Cash-Karp) ───────────────────────────────
    def get_streamline(self, r0, phi_index, orbit, material, dust_num):
        r_grid, phi_grid = self.data_loader.r_grid, self.data_loader.phi_grid

        if material == "gas":
            vy, vx = self.data_loader.load_gasvel(orbit)
        else:
            vy, vx = self.data_loader.load_dustvel(orbit, dust_num)

        phi0 = phi_grid[phi_index]
        step = np.pi / 100

        rvec   = np.zeros(len(phi_grid))
        phivec = np.zeros(len(phi_grid))
        rvec[0]   = r0
        phivec[0] = phi0

        def f(phip, rp):
            return (
                interpolate(rp, phip, vy, r_grid, phi_grid)
                * rp
                / interpolate(rp, phip, vx, r_grid, phi_grid)
            )

        for i in range(len(phi_grid) - 1):
            phi_i, r_i = phivec[i], rvec[i]

            k1 = step * f(phi_i,              r_i)
            k2 = step * f(phi_i + step / 5,   r_i + k1 / 5)
            k3 = step * f(phi_i + 3*step/10,  r_i + 3*k1/40 + 9*k2/40)
            k4 = step * f(phi_i + 3*step / 5, r_i + 3*k1/10 - 9*k2/10 + 6*k3/5)
            k5 = step * f(phi_i + step,        r_i - 11*k1/54 + 5*k2/2 - 70*k3/27 + 35*k4/27)
            k6 = step * f(phi_i + 7*step/8,
                          r_i + 1631*k1/55296 + 175*k2/512
                          + 575*k3/13824 + 44275*k4/110592 + 253*k5/4096)

            rvec[i + 1]   = r_i + 37*k1/378 + 250*k3/621 + 125*k4/594 + 512*k6/1771
            phivec[i + 1] = phi_i + step

        rvec_oriented = np.zeros(len(phi_grid))
        for i in range(len(phi_grid)):
            rvec_oriented[i] = rvec[i - phi_index]
        return rvec_oriented

    # ── vectorised streamlines (RK4) ─────────────────────────────────────
    def get_streamline_vectorized(self, r0_array, phi_index, orbit, material, dust_num, direction=1):

        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid

        if material == "gas":
            vy, vx = self.data_loader.load_gasvel(orbit)
        else:
            vy, vx = self.data_loader.load_dustvel(orbit, dust_num)

        phi0 = phi_grid[phi_index]
        base_step = phi_grid[1] - phi_grid[0]
        step = direction * base_step

        n_r0 = len(r0_array)
        n_phi = len(phi_grid)

        rvec = np.zeros((n_r0, n_phi))
        phivec = np.zeros((n_r0, n_phi))
        tvec = np.zeros((n_r0, n_phi))

        rvec[:, 0] = r0_array
        phivec[:, 0] = phi0
        tvec[:, 0] = 0.0

        def f(phi, r):
            phi = phi % (2 * np.pi)
            vy_i = interpolate_vectorized(r, phi, vy, r_grid, phi_grid)
            vx_i = interpolate_vectorized(r, phi, vx, r_grid, phi_grid)
            return vy_i * r / vx_i

        def g(phi, r):
            phi = phi % (2 * np.pi)
            vx_i = interpolate_vectorized(r, phi, vx, r_grid, phi_grid)
            return r / vx_i

        for i in range(n_phi - 1):

            phi_c = phi0 + i * step
            r_i = rvec[:, i]

            k1 = f(phi_c, r_i)
            k1t = g(phi_c, r_i)

            r_stage2 = r_i + step * k1 / 2
            k2 = f(phi_c + step / 2, r_stage2)
            k2t = g(phi_c + step / 2, r_stage2)

            r_stage3 = r_i + step * k2 / 2
            k3 = f(phi_c + step / 2, r_stage3)
            k3t = g(phi_c + step / 2, r_stage3)

            r_stage4 = r_i + step * k3
            k4 = f(phi_c + step, r_stage4)
            k4t = g(phi_c + step, r_stage4)

            rvec[:, i + 1] = r_i + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            tvec[:, i + 1] = tvec[:, i] + base_step / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
            phivec[:, i + 1] = phi_c + step

        orbit_time = tvec[:, -1]

        return rvec, phivec, tvec, orbit_time