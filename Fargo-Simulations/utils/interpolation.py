import numpy as np


def interpolate(r, phi, quantity, r_grid, phi_grid):
    """
    Bilinear interpolation of a 2-D array at a single (r, phi) point.
    Phi is treated as periodic on [-pi, pi), so the wraparound cell
    (and any phi below phi_grid[0]) interpolates correctly.

    Parameters
    ----------
    r, phi      : float  – target coordinates
    quantity    : 2-D array shaped (nr, nphi)
    r_grid      : 1-D array of radial cell centres
    phi_grid    : 1-D array of azimuthal cell centres

    Returns
    -------
    float – interpolated value
    """
    nphi = len(phi_grid)
    two_pi = 2 * np.pi

    if phi > np.pi:
        phi = phi - two_pi

    # radial interpolation (unchanged - not periodic)
    min_i = np.searchsorted(r_grid, r) - 1
    min_i = np.clip(min_i, 0, len(r_grid) - 2)
    max_i = min_i + 1
    y1 = r_grid[min_i]
    y2 = r_grid[max_i]

    # azimuthal interpolation - periodic-safe
    min_j = (np.searchsorted(phi_grid, phi) - 1) % nphi
    max_j = (min_j + 1) % nphi

    x1 = phi_grid[min_j]
    dx = (phi_grid[max_j] - x1) % two_pi   # cell width, correct even across the seam
    wx = (phi - x1) % two_pi               # offset into the cell, correct even across the seam

    f11 = quantity[min_i][min_j]
    f12 = quantity[max_i][min_j]
    f21 = quantity[min_i][max_j]
    f22 = quantity[max_i][max_j]

    return (
        1 / (dx * (y2 - y1))
        * (
            f11 * (dx - wx) * (y2 - r)
            + f12 * (dx - wx) * (r  - y1)
            + f21 * wx        * (y2 - r)
            + f22 * wx        * (r  - y1)
        )
    )


def interpolate_vectorized(r_array, phi_scalar, quantity, r_grid, phi_grid):
    """
    Bilinear interpolation at many r values but a single phi value.
    Phi is treated as periodic on [-pi, pi), so the wraparound cell
    (and any phi below phi_grid[0]) interpolates correctly.

    Parameters
    ----------
    r_array     : 1-D array of radial positions
    phi_scalar  : float – single azimuthal position
    quantity    : 2-D array shaped (nr, nphi)
    r_grid      : 1-D array of radial cell centres
    phi_grid    : 1-D array of azimuthal cell centres

    Returns
    -------
    1-D array – interpolated values at each r in r_array
    """
    nphi = len(phi_grid)
    two_pi = 2 * np.pi

    phi = phi_scalar
    if phi > np.pi:
        phi = phi - two_pi

    # radial interpolation (unchanged - not periodic)
    min_i = np.searchsorted(r_grid, r_array) - 1
    min_i = np.clip(min_i, 0, len(r_grid) - 2)
    max_i = min_i + 1
    y1 = r_grid[min_i]
    y2 = r_grid[max_i]

    # azimuthal interpolation - periodic-safe
    min_j = (np.searchsorted(phi_grid, phi) - 1) % nphi
    max_j = (min_j + 1) % nphi

    x1 = phi_grid[min_j]
    dx = (phi_grid[max_j] - x1) % two_pi   # cell width, correct even across the seam
    wx = (phi - x1) % two_pi               # offset into the cell, correct even across the seam

    f11 = quantity[min_i, min_j]
    f12 = quantity[max_i, min_j]
    f21 = quantity[min_i, max_j]
    f22 = quantity[max_i, max_j]

    return (
        1 / (dx * (y2 - y1))
        * (
            f11 * (dx - wx) * (y2 - r_array)
            + f12 * (dx - wx) * (r_array - y1)
            + f21 * wx        * (y2 - r_array)
            + f22 * wx        * (r_array - y1)
        )
    )