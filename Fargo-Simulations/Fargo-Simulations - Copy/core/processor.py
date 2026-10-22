import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.signal import savgol_filter
from utils.interpolation import interpolate
from analysis.orbital import fit_ellipse_polar, getChiSquare, fit_ellipses, ellipse_radii
from utils.interpolation import interpolate_points
from utils import cache
from scipy.interpolate import griddata

def build_a_mesh(r_grid, phi_grid, a_list, r_vec_list):
    key = cache.content_key("a_mesh", r_grid, phi_grid, np.asarray(a_list, dtype=float),
                            np.asarray(r_vec_list, dtype=float))
    return cache.RESULTS.get_or_compute(
        key, lambda: _build_a_mesh_uncached(r_grid, phi_grid, a_list, r_vec_list))


def _build_a_mesh_uncached(r_grid, phi_grid, a_list, r_vec_list):

    r_arr = np.asarray(r_vec_list)
    a_arr = np.asarray(a_list)

    # Make sure a has one value per streamline
    if a_arr.ndim != 1:
        raise ValueError("a_list should contain one a value per streamline.")

    if r_arr.shape[0] != len(a_arr):
        raise ValueError("Number of streamlines in r_vec_list and a_list must match.")

    # Grid coordinates
    R, PHI = np.meshgrid(r_grid, phi_grid, indexing='ij')

    # Every streamline point becomes a sample
    R_samples = r_arr.ravel()

    # phi is the same for every streamline
    PHI_samples = np.tile(phi_grid, r_arr.shape[0])

    # Same a value along an entire streamline
    A_samples = np.repeat(a_arr, r_arr.shape[1])

    # Remove invalid points
    valid = (
        np.isfinite(R_samples) &
        np.isfinite(PHI_samples) &
        np.isfinite(A_samples)
    )

    points = np.column_stack((
        R_samples[valid],
        PHI_samples[valid]
    ))

    values = A_samples[valid]

    # Interpolate all streamline points onto (r, phi) grid
    a_mesh = griddata(
        points,
        values,
        (R, PHI),
        method='linear'
    )

    return a_mesh


class SimulationDataProcessor:
    """High-level analysis: Statler reconstruction, flux, FFT decomposition."""

    def __init__(self, data_loader, variables_calculator, streamline_calculator):
        self.data_loader           = data_loader
        self.variables_calculator  = variables_calculator
        self.streamline_calculator = streamline_calculator

    # ── mu(a) via flux ────────────────────────────────────────────────────
    def get_mu_a_flux(self, orbit, phi_index):
        phi_grid = self.data_loader.phi_grid
        r_grid   = self.data_loader.r_grid
        vr, vphi = self.data_loader.load_gasvel(orbit)
        Sigma    = self.data_loader.load_gasdens(orbit)

        key = cache.content_key("mu_a_flux", Sigma, vr, vphi, r_grid, phi_grid, int(phi_index))

        def compute():
            r_array = np.logspace(np.log10(0.3), np.log10(22), 1_000)
            rvecs, _, _, _ = self.streamline_calculator.get_streamline_vectorized(
                r_array, phi_index=phi_index, orbit=orbit, material="gas", dust_num=None
            )

            # one batched (and cached) ellipse fit for all streamlines
            a_list, e_list, varpi_raw = fit_ellipses(rvecs, phi_grid)
            varpi_list = (varpi_raw + np.pi) % (2 * np.pi) - np.pi
            r_vec_list = ellipse_radii(a_list, e_list, varpi_list, phi_grid)

            # Sigma along every streamline in one vectorised call
            # (was 3 x 1000 x nphi scalar interpolate() calls; the vr/vphi copies were never used)
            Sigma_list = interpolate_points(rvecs, phi_grid[None, :], Sigma, r_grid, phi_grid)

            dphi    = phi_grid[1] - phi_grid[0]
            mu_list = np.sum(
                Sigma_list[:-1] * rvecs[:-1] * (rvecs[1:] - rvecs[:-1]) * dphi, axis=1
            ) / (a_list[1:] - a_list[:-1])

            return a_list, mu_list, r_vec_list, e_list, varpi_list

        return cache.RESULTS.get_or_compute(key, compute)

    def _statler_compute(self, orbit, phi_index):
        """Statler reconstruction (no plotting). Cached by content of the gas fields."""
        phi_grid = self.data_loader.phi_grid
        r_grid   = self.data_loader.r_grid
        nr, nphi = len(r_grid), len(phi_grid)
        vr, vphi = self.data_loader.load_gasvel(orbit)
        key = cache.content_key("statler", self.data_loader.load_gasdens(orbit), vr, vphi,
                                r_grid, phi_grid, int(phi_index))

        def compute():
            a_list, mu_a, r_vec_list, e_list, varpi_list = self.get_mu_a_flux(orbit, phi_index)

            de_list     = [(e_list[i + 1]     - e_list[i])     / (a_list[i + 1] - a_list[i]) for i in range(len(e_list)     - 1)]
            dvarpi_list = [(varpi_list[i + 1] - varpi_list[i]) / (a_list[i + 1] - a_list[i]) for i in range(len(varpi_list) - 1)]

            n_shells = len(a_list) - 1

            # Collect each shell's physical radius (r_vec, varies with phi since eccentric)
            # and its reconstructed surface-density value, instead of scattering into r_grid bins.
            equation_all = np.full((n_shells, nphi), np.nan)
            r_vec_all = np.full((n_shells, nphi), np.nan)

            for i in range(n_shells):
                r_vec = r_vec_list[i]

                mu_term = mu_a[i] / (2 * np.pi * a_list[i])
                numerator = np.sqrt(1 - e_list[i] ** 2)

                denom1 = numerator**2 - de_list[i] * (2 * a_list[i] * e_list[i] + r_vec * np.cos(phi_grid - varpi_list[i]))
                denom2 = r_vec * e_list[i] * dvarpi_list[i] * np.sin(phi_grid - varpi_list[i])

                equation = mu_term * numerator / (denom1 - denom2)

                valid = np.isfinite(r_vec) & np.isfinite(equation)

                r_vec_all[i, valid] = r_vec[valid]
                equation_all[i, valid] = equation[valid]

            Sigma = np.full((nr, nphi), np.nan)

            for j in range(nphi):
                r_col = r_vec_all[:, j]
                sigma_col = equation_all[:, j]

                valid = np.isfinite(r_col) & np.isfinite(sigma_col)

                if np.sum(valid) < 2:
                    continue

                r_col = r_col[valid]
                sigma_col = sigma_col[valid]

                order = np.argsort(r_col)

                r_col = r_col[order]
                sigma_col = sigma_col[order]

                r_col, unique_idx = np.unique(r_col, return_index=True)
                sigma_col = sigma_col[unique_idx]

                if len(r_col) < 2:
                    continue

                mask = (r_grid >= r_col[0]) & (r_grid <= r_col[-1])

                Sigma[mask, j] = np.interp(r_grid[mask], r_col, sigma_col)

            return Sigma, a_list, e_list, r_vec_list, mu_a, varpi_list

        return cache.RESULTS.get_or_compute(key, compute)

    # ── Statler surface density reconstruction ────────────────────────────
    def statler_model(self, orbit, phi_index, plot):
        phi_grid = self.data_loader.phi_grid
        r_grid   = self.data_loader.r_grid
        X, Y     = self.data_loader.X, self.data_loader.Y
        nr, nphi = len(r_grid), len(phi_grid)

        gas_dens = self.data_loader.load_gasdens(orbit)

        if plot == "yes":
            fig, ax = plt.subplots(2, 2, figsize=(8, 8))
            pcm = ax[0, 0].pcolormesh(X, Y, np.log10(gas_dens), cmap="viridis", vmin=-7, vmax=-3)
            fig.colorbar(pcm, ax=ax[0, 0], label=r'$\log_{10}(\Sigma_{\rm gas})$')
            ax[0, 0].set_title("Gas Density"); ax[0, 0].set_xlabel("X"); ax[0, 0].set_ylabel("Y")
            ax[0, 0].set_aspect("equal"); ax[0, 0].set_ylim(-20, 20); ax[0, 0].set_xlim(-20, 20)

        Sigma, a_list, e_list, r_vec_list, mu_a, varpi_list = self._statler_compute(orbit, phi_index)
        Sigma_ratio = Sigma / gas_dens

        if plot == "yes":
            pcm = ax[0, 1].pcolormesh(X, Y, np.log10(Sigma), cmap="viridis", vmin=-7, vmax=-3)
            fig.colorbar(pcm, ax=ax[0, 1], label=r'$\log_{10}(\Sigma_{\rm gas})$')
            ax[0, 1].set_title("Gas Density (Statler)"); ax[0, 1].set_xlabel("X"); ax[0, 1].set_ylabel("Y")
            ax[0, 1].set_aspect("equal"); ax[0, 1].set_ylim(-20, 20); ax[0, 1].set_xlim(-20, 20)

            pcm = ax[1, 0].pcolormesh(X, Y, 1 - Sigma_ratio, cmap="viridis", vmin=-0.2, vmax=0.2)
            fig.colorbar(pcm, ax=ax[1, 0], label=r'$1-(\Sigma_{\rm Statler}/\Sigma_{\rm gas})$')
            ax[1, 0].set_title("Gas Density Ratio"); ax[1, 0].set_xlabel("X"); ax[1, 0].set_ylabel("Y")
            ax[1, 0].set_aspect("equal"); ax[1, 0].set_ylim(-20, 20); ax[1, 0].set_xlim(-20, 20)

            ax[1, 1].plot(a_list[1:], mu_a / (2 * np.pi * a_list[1:]))
            ax[1, 1].set_xlabel("a"); ax[1, 1].set_ylabel(r'$\mu(a) / 2 \pi a$')
            plt.tight_layout(); plt.show()

            plt.plot(a_list, varpi_list); plt.ylabel(r'$\varpi$'); plt.xlabel("a"); plt.show()
            plt.hist(Sigma[~np.isnan(Sigma)], bins=100); plt.yscale("log"); plt.show()
            plt.plot(a_list, e_list, ".-"); plt.ylabel("e"); plt.xlabel("a"); plt.show()

        return Sigma, a_list, e_list, r_vec_list

    def get_a_from_ew(self, r, phi, e_of_a, varpi_of_a, a_min, a_max):
        def residual(a):
            e     = e_of_a(a)
            varpi = varpi_of_a(a)
            r_ellipse = a * (1 - e**2) / (1 + e * np.cos(phi - varpi))
            return r_ellipse - r
        try:
            return brentq(residual, a_min, a_max)
        except ValueError:
            return np.nan

    from scipy.signal import savgol_filter

    def statler_model_test(self, orbit, phi_index, plot):
        phi_grid = self.data_loader.phi_grid
        r_grid   = self.data_loader.r_grid
        X, Y     = self.data_loader.X, self.data_loader.Y
        nr, nphi = len(r_grid), len(phi_grid)

        gas_dens = self.data_loader.load_gasdens(orbit)

        if plot == "yes":
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            pcm = ax[0, 0].pcolormesh(X, Y, np.log10(gas_dens), cmap="viridis", vmin=-7, vmax=-3)
            fig.colorbar(pcm, ax=ax[0, 0], label=r'$\log_{10}(\Sigma_{\rm gas})$')
            ax[0, 0].set_title("Gas Density"); ax[0, 0].set_xlabel("x"); ax[0, 0].set_ylabel("y")
            ax[0, 0].set_aspect("equal"); ax[0, 0].set_ylim(-20, 20); ax[0, 0].set_xlim(-20, 20)

        a_list, mu_a, r_vec_list, e_list, varpi_list = self.get_mu_a_flux(orbit, phi_index)

        # --- Smooth e_list and varpi_list before differentiating ---
        valid_e = np.isfinite(e_list)
        e_filled = np.interp(a_list, a_list[valid_e], e_list[valid_e])

        valid_v = np.isfinite(varpi_list)
        varpi_unwrapped = np.unwrap(varpi_list)
        varpi_filled = np.interp(a_list, a_list[valid_v], varpi_unwrapped[valid_v])

        window_length = 51  # must be odd and < len(a_list)
        polyorder = 3
        e_smooth     = savgol_filter(e_filled, window_length, polyorder)
        varpi_smooth = savgol_filter(varpi_filled, window_length, polyorder)

        if plot == "yes":
            plt.figure()
            plt.plot(a_list, e_list, '.', alpha=0.3, label='raw')
            plt.plot(a_list, e_smooth, '-', label='smoothed')
            plt.xlabel('a'); plt.ylabel('e'); plt.legend()
            plt.show()

        de_list     = [(e_smooth[i + 1]     - e_smooth[i])     / (a_list[i + 1] - a_list[i]) for i in range(len(e_smooth)     - 1)]
        dvarpi_list = [(varpi_smooth[i + 1] - varpi_smooth[i]) / (a_list[i + 1] - a_list[i]) for i in range(len(varpi_smooth) - 1)]

        n_shells = len(a_list) - 1
        equation_all = np.full((n_shells, nphi), np.nan)
        r_vec_all    = np.full((n_shells, nphi), np.nan)

        for r in range(n_shells):
            r_vec       = r_vec_list[r]
            mu_term     = mu_a[r] / (2 * np.pi * a_list[r])
            numerator   = np.sqrt(1 - e_smooth[r] ** 2)
            denom1      = numerator**2 - de_list[r] * (2 * a_list[r] * e_smooth[r] + r_vec * np.cos(phi_grid - varpi_smooth[r]))
            denom2      = r_vec * e_smooth[r] * dvarpi_list[r] * np.sin(phi_grid - varpi_smooth[r])
            equation    = mu_term * numerator / (denom1 - denom2)

            if np.all(np.isfinite(equation)):
                equation_all[r, :] = equation
                r_vec_all[r, :]    = r_vec

        Sigma = np.zeros((nr, nphi))
        for j in range(nphi):
            r_col = r_vec_all[:, j]
            v_col = equation_all[:, j]

            valid = np.isfinite(r_col) & np.isfinite(v_col)
            if valid.sum() < 2:
                continue

            r_valid = r_col[valid]
            v_valid = v_col[valid]
            order   = np.argsort(r_valid)
            r_sorted = r_valid[order]
            v_sorted = v_valid[order]

            r_unique, unique_idx = np.unique(r_sorted, return_index=True)
            v_unique = v_sorted[unique_idx]

            if len(r_unique) < 2:
                continue

            col = np.interp(r_grid, r_unique, v_unique, left=np.nan, right=np.nan)
            Sigma[:, j] = col

        Sigma_ratio = Sigma / gas_dens

        if plot == "yes":
            pcm = ax[0, 1].pcolormesh(X, Y, np.log10(Sigma), cmap="viridis", vmin=-7, vmax=-3)
            fig.colorbar(pcm, ax=ax[0, 1], label=r'$\log_{10}(\Sigma_{\rm gas})$')
            ax[0, 1].set_title("Gas Density"); ax[0, 1].set_xlabel("x"); ax[0, 1].set_ylabel("y")
            ax[0, 1].set_aspect("equal"); ax[0, 1].set_ylim(-20, 20); ax[0, 1].set_xlim(-20, 20)

            pcm = ax[1, 0].pcolormesh(X, Y, 1 - Sigma_ratio, cmap="viridis", vmin=-0.2, vmax=0.2)
            fig.colorbar(pcm, ax=ax[1, 0], label=r'$1-(\Sigma_{\rm Statler}/\Sigma_{\rm gas})$')
            ax[1, 0].set_title("Gas Density Ratio"); ax[1, 0].set_xlabel("x"); ax[1, 0].set_ylabel("y")
            ax[1, 0].set_aspect("equal"); ax[1, 0].set_ylim(-20, 20); ax[1, 0].set_xlim(-20, 20)

            ax[1, 1].plot(a_list[1:], mu_a / (2 * np.pi * a_list[1:]))
            ax[1, 1].set_xlabel("a"); ax[1, 1].set_ylabel(r'$\mu(a) / 2 \pi a$')
            plt.tight_layout(); plt.show()

            plt.plot(a_list, varpi_smooth); plt.ylabel(r'$\varpi$ (smoothed)'); plt.xlabel("a"); plt.show()
            plt.hist(Sigma[~np.isnan(Sigma)], bins=100); plt.yscale("log"); plt.show()
            plt.plot(a_list, e_smooth, ".-"); plt.ylabel("e (smoothed)"); plt.xlabel("a"); plt.show()

        return Sigma, a_list, r_vec_list