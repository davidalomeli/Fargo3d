import numpy as np
from scipy.optimize import curve_fit
from utils import cache


def getChiSquare(a, e, varpi, radii, phi_grid):
    """
    Penalty function: sum of squared residuals between ellipse r(phi)
    and the observed streamline radii.
    """
    r  = a * (1 - e**2) / (1 + e * np.cos(phi_grid - varpi))
    return np.sum((r - radii) ** 2)


def polar_ellipse(phi, a, e, varpi):
    """Keplerian ellipse in polar form r(phi)."""
    if e < 0 or e >= 1:
        return np.full_like(phi, 1e10)
    return a * (1 - e**2) / (1 + e * np.cos(phi - varpi))


def fit_ellipse_polar(r_data, phi_data):

    r_data = np.asarray(r_data).ravel()
    phi_data = np.asarray(phi_data).ravel()

    p0 = [r_data[0], 0.1, 0.0]
    bounds = ([0, 0.0, -2 * np.pi], [np.inf, 1.0, 2 * np.pi])

    try:
        popt, _ = curve_fit(
            polar_ellipse,
            phi_data,
            r_data,
            p0=p0,
            bounds=bounds,
            sigma=np.ones_like(r_data) * 10,
        )

        a_fit, e_fit, varpi_fit = popt
        varpi_fit = varpi_fit % (2 * np.pi)

        return a_fit, e_fit, varpi_fit

    except RuntimeError as err:
        print(f"Ellipse fit failed: {err}")
        return None, None, None


def fit_ellipses(r_vecs, phi_data):
    """
    Fit a Keplerian ellipse to every streamline (row) of r_vecs, sharing one phi array.
    Returns (a, e, varpi) arrays; a failed fit gives NaN (fit_ellipse_polar returns None,
    which numpy already turned into NaN in the old assignment loops).
    Cached by content: the same streamlines are never fitted twice.
    """
    r_vecs   = np.ascontiguousarray(r_vecs, dtype=float)
    phi_data = np.ascontiguousarray(phi_data, dtype=float).ravel()

    def compute():
        n = len(r_vecs)
        a, e, v = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        for i in range(n):
            fit = fit_ellipse_polar(r_vecs[i], phi_data)
            if fit[0] is not None:
                a[i], e[i], v[i] = fit
        return a, e, v

    return cache.RESULTS.get_or_compute(cache.content_key("fit_ellipses", r_vecs, phi_data), compute)


def ellipse_radii(a, e, varpi, phi_grid):
    """r(phi) for many ellipses at once -> shape (n_ellipses, len(phi_grid))."""
    a, e, varpi = (np.asarray(x)[:, None] for x in (a, e, varpi))
    return a * (1 - e**2) / (1 + e * np.cos(np.asarray(phi_grid)[None, :] - varpi))


def get_eccentricity_varpi_a(X, Y, PHI, vx, vy):
    """
    Compute per-cell eccentricity vector, argument of pericenter, and
    semi-major axis from position and velocity fields.
    """
    R_vec = np.linalg.norm(np.stack([X, Y]), axis=0)

    e_x = R_vec * vx * vy * np.sin(PHI) + (R_vec * vx**2 - 1) * np.cos(PHI)
    e_y = -R_vec * vx * vy * np.cos(PHI) + (R_vec * vx**2 - 1) * np.sin(PHI)
    es  = np.sqrt(e_x**2 + e_y**2)

    varpi  = np.arctan2(e_y, e_x)
    energy = 0.5 * (vx**2 + vy**2) - 1 / R_vec
    a      = -1 / (2 * energy)

    return es, varpi, a


def orbit_average(f, r_grid, phi_grid, a, e, varpi):
    """
    Time-average of quantity f over an elliptical streamline (a, e, varpi).

    Uses the Keplerian time weight  <f> = \oint f r^2 dphi / \oint r^2 dphi.

    Parameters
    ----------
    f        : 2-D array (nr, nphi)
    r_grid   : 1-D array
    phi_grid : 1-D array
    a, e, varpi : ellipse parameters

    Returns
    -------
    float – orbit-averaged value
    """
    from scipy.interpolate import RegularGridInterpolator

    r_orbit = a * (1 - e**2) / (1 + e * np.cos(phi_grid - varpi))
    r_orbit = np.clip(r_orbit, r_grid[0], r_grid[-1])

    interp     = RegularGridInterpolator(
        (r_grid, phi_grid), f, method="linear",
        bounds_error=False, fill_value=np.nan,
    )
    f_on_orbit = interp(np.column_stack([r_orbit, phi_grid]))

    r2 = r_orbit**2
    return np.trapz(f_on_orbit * r2, phi_grid) / np.trapz(r2, phi_grid)


def b_laplace(s, m, alpha):
    """Laplace coefficient b_s^m(alpha)."""
    alpha = np.atleast_1d(alpha)
    psi   = np.linspace(0, 2 * np.pi, 2000)
    cos_mpsi = np.cos(m * psi)
    denom    = (1 - 2 * alpha[:, None] * np.cos(psi) + alpha[:, None]**2) ** s
    result   = np.trapezoid(cos_mpsi[None, :] / denom, psi, axis=1) / np.pi
    return result if alpha.size > 1 else result.item()
