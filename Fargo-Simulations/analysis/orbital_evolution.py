import numpy as np
import matplotlib.pyplot as plt
from get_parameters import read_par
from utils.constants import R0_CGS, MSTAR_CGS, MSTAR, R0
from utils.interpolation import interpolate, interpolate_vectorized
from analysis.orbital import fit_ellipse_polar
from analysis.stokes import relative_velocity_calculator, Stokes_from_Particle
from core.processor import build_a_mesh
from scipy.signal import savgol_filter
import scipy.integrate as integrate
from scipy.ndimage import map_coordinates
import warnings


class da_dt_calculator:

    def __init__(self, data_loader, variables_calculator, simulation_streamlines, dust_num, dusttype):
        params = read_par(data_loader.path)
        self.data_loader          = data_loader
        self.variables_calculator = variables_calculator
        self.streamlines          = simulation_streamlines
        self.dust_num             = dust_num
        self.dusttype             = dusttype
        if dusttype == 'particle':
            self.particle_size = (
                float(params[f"PARTICLESIZE{self.dust_num}"]) * R0 / R0_CGS
            )
            self.rhosolid = (
                float(params["RHOSOLID"])
                / (MSTAR_CGS / R0_CGS**3)
                * (MSTAR / R0**3)
            )
        elif dusttype == 'stokes':
            try:
                self.stokes = 1 / float(params[f'INVSTOKES{dust_num}'])
            except:
                self.stokes = float(params[f'TS'])
        self.rvc         = relative_velocity_calculator(data_loader, variables_calculator, self.streamlines, dust_num, 'statler')
        self.stokes_calc = Stokes_from_Particle(self.data_loader, self.variables_calculator, self.streamlines, self.dust_num, self.dusttype)

    # ------------------------------------------------------------------
    # Helpers shared by the Statler-type calculations
    # ------------------------------------------------------------------
    def calc_big_QS_n(self, orbit):
        params = read_par(self.data_loader.path)
        St, _ = self.stokes_calc.compute_stokes_number(orbit)
        eps = float(params["EPSILON"])

        big_Q_n = eps * St / (1 + St**2)
        big_S_n = big_Q_n / St

        return big_Q_n, big_S_n

    def calc_xi(self, orbit):
        params = read_par(self.data_loader.path)
        aspect_ratio = float(params["ASPECTRATIO"])
        r_grid = self.data_loader.r_grid
        pres_map = self.variables_calculator.calculate_gas_pressure(orbit)
        dp_dr = self.variables_calculator.calculate_pressure_gradient(orbit)

        eta = aspect_ratio**2 / 2 * (r_grid[:, None] / pres_map) * dp_dr
        beta = np.sqrt(1 + 2*eta)
        dbeta_dr = np.gradient(beta, r_grid, axis=0)
        xi = beta * (0.5 + r_grid[:, None] / beta * dbeta_dr)
        return xi

    def calc_psi(self, orbit):
        params = read_par(self.data_loader.path)
        aspect_ratio = float(params["ASPECTRATIO"])
        r_grid = self.data_loader.r_grid
        pres_map = self.variables_calculator.calculate_gas_pressure(orbit)
        dp_dr = self.variables_calculator.calculate_pressure_gradient(orbit)

        eta = aspect_ratio**2 / 2 * (r_grid[:, None] / pres_map) * dp_dr
        beta = np.sqrt(1 + 2*eta)

        big_Q_n, big_S_n = self.calc_big_QS_n(orbit)
        xi = self.calc_xi(orbit)
        psi = 1 / ((big_S_n + beta) * (big_S_n + 2 * xi) + 1 / (big_Q_n)**2)

        return psi

    def calc_delta_v_gas(self, orbit):
        params = read_par(self.data_loader.path)
        aspect_ratio = float(params["ASPECTRATIO"])
        r_grid = self.data_loader.r_grid
        pres_map = self.variables_calculator.calculate_gas_pressure(orbit)
        dp_dr = self.variables_calculator.calculate_pressure_gradient(orbit)

        eta = aspect_ratio**2 / 2 * (r_grid[:, None] / pres_map) * dp_dr
        beta = np.sqrt(1 + 2*eta)

        big_Q_n, big_S_n = self.calc_big_QS_n(orbit)
        psi = self.calc_psi(orbit)
        xi = self.calc_xi(orbit)

        v_K = 1 / np.sqrt(r_grid[:, None])

        delta_vgas_r = -2 * beta * big_Q_n * psi * (beta - 1) * v_K
        delta_vgas_phi = -((big_S_n + 2 * xi) * big_S_n + big_Q_n**2) * psi * (beta - 1) * v_K
        return delta_vgas_r, delta_vgas_phi

    # ------------------------------------------------------------------
    # Semi-major axis maps
    # ------------------------------------------------------------------
    def calc_a_from_spec_energy(self, orbit, material):
        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid

        if material == 'gas':
            vr, vphi = self.data_loader.load_gasvel(orbit)
        else:
            vr, vphi = self.data_loader.load_dustvel(orbit, self.dust_num)

        energy = 0.5 * (vr**2 + vphi**2) - 1 / r_grid[:, None]
        a_map = -1 / (2 * energy)
        return a_map

    import warnings

    def calc_a_from_streamlines(self, orbit, material, return_orbits=False,
                                phi_indices=(0, 50, 100, 150)):
        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid
        n_phi = len(phi_grid)
        dphi = phi_grid[1] - phi_grid[0]
        r_inits = np.logspace(np.log10(r_grid[0]), np.log10(22), 500)
        n = len(r_inits)

        a_maps, e_maps = [], []
        a_all, e_all, r_ell_all, r_vec_all = [], [], [], []

        for p in phi_indices:
            r_vecs, phi_vecs, t_vecs, orbit_times = self.streamlines.get_streamline_vectorized(
                r_inits, int(p), orbit, material, self.dust_num
            )

            a_array = np.full(n, np.nan)
            e_array = np.full(n, np.nan)
            r_ellipses = np.full((n, n_phi), np.nan)
            r_aligned = np.full((n, n_phi), np.nan)

            for i, r_vec in enumerate(r_vecs):
                r_vec = np.asarray(r_vec)
                if len(r_vec) != n_phi:
                    raise ValueError(f"streamline has {len(r_vec)} points, expected {n_phi}")

                # true azimuths of this streamline: it starts at phi_grid[p]
                phi_stream = phi_grid[p] + dphi * np.arange(n_phi)
                a_fit, e_fit, varpi = fit_ellipse_polar(r_vec, phi_stream)
                if a_fit is None or e_fit is None:
                    continue

                a_array[i], e_array[i] = a_fit, e_fit
                r_ellipses[i, :] = a_fit * (1 - e_fit**2) / (
                    1 + e_fit * np.cos(phi_grid - varpi)
                )
                # align so that column j sits at phi_grid[j]
                r_aligned[i, :] = np.roll(r_vec, p)

            a_maps.append(build_a_mesh(r_grid, phi_grid, a_array, r_ellipses))
            e_maps.append(build_a_mesh(r_grid, phi_grid, e_array, r_ellipses))
            a_all.append(a_array)
            e_all.append(e_array)
            r_ell_all.append(r_ellipses)
            r_vec_all.append(r_aligned)

        # average the maps over starting azimuths, ignoring starts that left a cell empty
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            a_map = np.nanmean(np.stack(a_maps), axis=0)
            e_map = np.nanmean(np.stack(e_maps), axis=0)

        if return_orbits:
            # orbits from all starts are stacked: (len(phi_indices) * n, ...)
            return (a_map, e_map,
                    np.concatenate(a_all), np.concatenate(e_all),
                    np.concatenate(r_ell_all, axis=0), np.concatenate(r_vec_all, axis=0))
        return a_map
    # ------------------------------------------------------------------
    # Orbit averaging
    # ------------------------------------------------------------------
    def orbit_average(self, da_dt_map, a_map, e_map, r_orb, orbit):
        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid
        n_r = len(r_grid)

        a = np.asarray(a_map)
        e = np.asarray(e_map)
        r_orb = np.asarray(r_orb)
        v_phi_dust = self.data_loader.load_dustvel(orbit, self.dust_num)[1]

        # periodic in phi: pad one column on each side, then shift the phi index by +1
        da_dt_pad = np.pad(da_dt_map, ((0, 0), (1, 1)), mode='wrap')
        v_phi_pad = np.pad(v_phi_dust, ((0, 0), (1, 1)), mode='wrap')
        phi_idx = np.arange(len(phi_grid)) + 1

        da_dt_orbit = np.zeros(r_orb.shape)
        v_phi_orbit = np.zeros(r_orb.shape)
        for i in range(len(r_orb)):
            r_idx = np.interp(r_orb[i, :], r_grid, np.arange(n_r),
                            left=np.nan, right=np.nan)
            off = ~np.isfinite(r_idx)
            coords = [np.where(off, 0, r_idx), phi_idx]
            da_dt_orbit[i, :] = map_coordinates(da_dt_pad, coords, order=1, mode='nearest')
            v_phi_orbit[i, :] = map_coordinates(v_phi_pad, coords, order=1, mode='nearest')
            da_dt_orbit[i, off] = np.nan      # orbit leaves the grid -> NaN
            v_phi_orbit[i, off] = np.nan

        weight = np.abs(r_orb / v_phi_orbit)  # dt = r dphi / v_phi along the orbit
        da_dt_avg = np.sum(da_dt_orbit * weight, axis=1) / np.sum(weight, axis=1)

        #bad = (e >= 1) | (a <= 0) | ~np.isfinite(a) | ~np.isfinite(e)
        #da_dt_avg[bad] = np.nan
        return da_dt_avg

    # ------------------------------------------------------------------
    # Two-streamline method
    # ------------------------------------------------------------------
    def compute_da_dt_2streams(self, path, orbit, material):

        r_grid = self.data_loader.r_grid
        r_inits = r_grid
        r_inits = np.logspace(np.log10(r_grid[0]), np.log10(22), 1000)

        for phi_index in [0, 50, 100, 150]:
            r_vecs_fwd, phi_vecs_fwd, t_vecs_fwd, t_fwd = self.streamlines.get_streamline_vectorized(
                r_inits, phi_index, orbit, material, self.dust_num, direction=1
            )
            r_vecs_bwd, phi_vecs_bwd, t_vecs_bwd, t_bwd = self.streamlines.get_streamline_vectorized(
                r_inits, phi_index, orbit, material, self.dust_num, direction=-1
            )

            a_fwd = np.zeros(len(r_inits))
            a_bwd = np.zeros(len(r_inits))

            for i in range(len(r_inits)):
                a_fwd[i], _, _ = fit_ellipse_polar(r_vecs_fwd[i], phi_vecs_fwd[i])
                a_bwd[i], _, _ = fit_ellipse_polar(r_vecs_bwd[i], phi_vecs_bwd[i])

            a0_array = 0.5 * (a_fwd + a_bwd)
            da_dt_array = (a_fwd - a_bwd) / (np.mean(t_vecs_fwd, axis=1) + np.mean(t_vecs_bwd, axis=1))

        #r_vecs_fwd, phi_vecs_fwd, t_vecs_fwd, t_fwd = self.streamlines.get_streamline_vectorized(
        #    r_inits, 150, orbit, material, self.dust_num, direction=1
        #)
        #r_vecs_bwd, phi_vecs_bwd, t_vecs_bwd, t_bwd = self.streamlines.get_streamline_vectorized(
        #    r_inits, 150, orbit, material, self.dust_num, direction=-1
        #)

        #a_fwd = np.zeros(len(r_inits))
        #a_bwd = np.zeros(len(r_inits))

        #for i in range(len(r_inits)):
        #    a_fwd[i], _, _ = fit_ellipse_polar(r_vecs_fwd[i], phi_vecs_fwd[i])
        #    a_bwd[i], _, _ = fit_ellipse_polar(r_vecs_bwd[i], phi_vecs_bwd[i])

        #a0_array = 0.5 * (a_fwd + a_bwd)
        #da_dt_array = (a_fwd - a_bwd) / (np.mean(t_vecs_fwd, axis=1) + np.mean(t_vecs_bwd, axis=1))

        return a0_array, da_dt_array

    # ------------------------------------------------------------------
    # Drag-induced relative velocities
    # ------------------------------------------------------------------
    def velocity_from_epstein_drag(self, orbit, material, method, return_orbits=False):
        """
        Returns a_map, delta_v_r, delta_v_phi.
        If return_orbits is True, also returns the streamline-fit orbit data used
        to build a_map: a_orb (n_orb,), e_orb (n_orb,), r_orb (n_orb, n_phi).
        """
        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid
        params = read_par(self.data_loader.path)
        aspect_ratio = float(params["ASPECTRATIO"])
        v_th = np.sqrt(8 / np.pi) * self.variables_calculator.calculate_sound_speed()
        c_s = self.variables_calculator.calculate_sound_speed()

        if method == 'statler':
            pres_map, dp_dr, Sigma_g, _, a_list, e_list, r_vec_list = self.rvc.calculate_statler_components(orbit)

            a_map = build_a_mesh(r_grid, phi_grid, a_list, r_vec_list)
            e_map = build_a_mesh(r_grid, phi_grid, e_list, r_vec_list)
            # NOTE: this overwrites the statler a_map above (kept from the original code)
            a_map, e_map, a_orb, e_orb, r_orb, r_vec = self.calc_a_from_streamlines(orbit, 'gas', return_orbits=True)
            dp_dphi = np.gradient(pres_map, phi_grid, axis=1)

        else:
            Sigma_g = self.data_loader.load_gasdens(orbit)
            dp_dr = self.variables_calculator.calculate_pressure_gradient(orbit)
            pres_map = self.variables_calculator.calculate_gas_pressure(orbit)
            dp_dphi = np.gradient(pres_map, phi_grid, axis=1)
            #a_map = self.calc_a_from_spec_energy(orbit, 'dust')
            a_map, e_map, a_orb, e_orb, r_orb, r_vec = self.calc_a_from_streamlines(orbit, 'gas', return_orbits=True)

        gasdens = Sigma_g / (np.sqrt(2*np.pi) * aspect_ratio * r_grid[:, None])

        v_K = np.sqrt(1 / a_map)
        Omega_K = np.sqrt(1 / r_grid**3)

        ####### Regular method
        if self.dusttype == 'particle':
            particle_size = float(params[f"PARTICLESIZE{self.dust_num}"]) * R0 / R0_CGS
            rhosolid = float(params["RHOSOLID"]) / (MSTAR_CGS / R0_CGS**3) * (MSTAR / R0**3)
            St, stopping_time = self.stokes_calc.compute_stokes_number(orbit)
            stopping_time = St / Omega_K[:, None]
            print("Stopping time:", np.nanmin(stopping_time), np.nanmax(stopping_time))
            print(f"Stokes: {np.nanmin(St)}-{np.nanmax(St)}")
            eta = aspect_ratio**2 / 2 * (r_grid[:, None] / pres_map) * dp_dr
            beta = np.sqrt(1 + 2*eta)
            delta_v_gas_r, delta_v_gas_phi = self.calc_delta_v_gas(orbit)
            delta_v_r = (rhosolid * particle_size / Sigma_g / v_th / gasdens) * dp_dr
            delta_v_phi = (rhosolid * particle_size / Sigma_g / v_th / gasdens) * dp_dphi / r_grid[:, None]

        if self.dusttype == 'stokes':
            stopping_time = self.stokes / Omega_K[:, None]

            delta_v_r = stopping_time / Sigma_g * dp_dr
            delta_v_phi = stopping_time / Sigma_g / r_grid[:, None] * dp_dphi

        if return_orbits:
            return a_map, e_map, delta_v_r, delta_v_phi, a_orb, e_orb, r_orb, r_vec
        return a_map, e_map, delta_v_r, delta_v_phi

    # ------------------------------------------------------------------
    # Statler (already orbit-averaged)
    # ------------------------------------------------------------------
    def compute_da_dt_statler(self, orbit):
        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid

        a_map, e_map, delta_v_r, delta_v_phi = self.velocity_from_epstein_drag(orbit, 'dust', 'statler')
        _, _, _, _, a_list, e_list, r_vec_list = self.rvc.calculate_statler_components(orbit)

        # da/dt on the (r, phi) grid
        da_dr = np.gradient(a_map, r_grid, axis=0)
        da_dphi = np.gradient(a_map, phi_grid, axis=1)
        da_dt = da_dr * delta_v_r + da_dphi * delta_v_phi / r_grid[:, None]   # (n_r, n_phi)

        a = np.asarray(a_list)
        e = np.asarray(e_list)
        r_orb = np.asarray(r_vec_list)   # (n_orbits, n_phi); n_orbits need not equal n_r


        da_dt_avg = self.orbit_average(da_dt, a_map, e_map, r_orb, orbit)
        return a, da_dt_avg

    # ------------------------------------------------------------------
    # Orbit-averaged simulation (grad a . delta_v) method
    # ------------------------------------------------------------------
    def compute_da_dt_sim_avg(self, orbit, material='gas'):
        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid

        v_r_dust, v_phi_dust = self.data_loader.load_dustvel(orbit, self.dust_num)
        v_r_gas, v_phi_gas = self.data_loader.load_gasvel(orbit)
        delta_v_r = v_r_dust - v_r_gas
        delta_v_phi = v_phi_dust - v_phi_gas

        a_map, e_map, a, e, r_orb, r_vec = self.calc_a_from_streamlines(orbit, material, return_orbits=True)

        da_dr = np.gradient(a_map, r_grid, axis=0)
        da_dphi = np.gradient(a_map, phi_grid, axis=1)
        da_dt = da_dr * delta_v_r + da_dphi * delta_v_phi / r_grid[:, None]

        da_dt_avg = self.orbit_average(da_dt, a_map, e_map, r_vec, orbit)
        return a, da_dt_avg

    # ------------------------------------------------------------------
    # Orbit-averaged TVA (Epstein-drag relative velocities)
    # ------------------------------------------------------------------
    def compute_da_dt_drag_avg(self, orbit, material='gas'):
        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid

        a_map, e_map, delta_v_r, delta_v_phi, a, e, r_orb, r_vec = self.velocity_from_epstein_drag(
            orbit, material, 'drag', return_orbits=True
        )

        da_dr = np.gradient(a_map, r_grid, axis=0)
        da_dphi = np.gradient(a_map, phi_grid, axis=1)
        da_dt = da_dr * delta_v_r + da_dphi * delta_v_phi / r_grid[:, None]

        da_dt_avg = self.orbit_average(da_dt, a_map, e_map, r_orb, orbit)
        return a, da_dt_avg

    # ------------------------------------------------------------------
    # Axisymmetric (Llambay-style) velocities
    # ------------------------------------------------------------------
    def compute_da_dt_llambay(self, orbit):
        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid
        params = read_par(self.data_loader.path)
        aspect_ratio = float(params["ASPECTRATIO"])
        Sigma_g = self.data_loader.load_gasdens(orbit)
        Sigma_g = np.mean(Sigma_g, axis=1)
        Sigma_map = np.zeros((len(r_grid), len(phi_grid)))
        for i in range(len(phi_grid)):
            Sigma_map[:, i] = Sigma_g
        Sigma_g = Sigma_map
        c_s = self.variables_calculator.calculate_sound_speed()
        pres_map = Sigma_g * c_s**2
        dp_dr = np.gradient(pres_map, r_grid, axis=0)

        a_map = np.zeros((len(r_grid), len(phi_grid)))
        for i in range(len(phi_grid)):
            a_map[:, i] = r_grid

        v_K = np.sqrt(1 / a_map)
        Omega_K = np.sqrt(1 / r_grid**3)

        if self.dusttype == 'particle':
            St, stopping_time = self.stokes_calc.compute_stokes_number(orbit)
            print(f"Stokes: {np.nanmin(St)}-{np.nanmax(St)}")

        elif self.dusttype == 'stokes':
            St = self.stokes

        eta = aspect_ratio**2 / 2 * (r_grid[:, None] / pres_map) * dp_dr
        beta = np.sqrt(1 + 2*eta)

        delta_v_r = 2 * St / (1 + St**2) * (beta - 1) * v_K
        delta_v_phi = 1 / (1 + St**2) * (beta - 1) * v_K
        print(a_map.shape, delta_v_r.shape, delta_v_phi.shape)
        return a_map, delta_v_r, delta_v_phi

    # ------------------------------------------------------------------
    # Local (non-averaged) da/dt maps
    # ------------------------------------------------------------------
    def compute_da_dt_mass_flux(self, orbit, material, method):
        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid

        if method == 'drag':
            a_map, delta_v_r, delta_v_phi = self.velocity_from_epstein_drag(orbit, material, method)

        elif method == 'sim':
            if material == 'gas':
                v_r_gas, v_phi_gas = self.data_loader.load_gasvel(orbit)

            elif material == 'dust':
                v_r_dust, v_phi_dust = self.data_loader.load_dustvel(orbit, self.dust_num)
                v_r_gas, v_phi_gas = self.data_loader.load_gasvel(orbit)
            delta_v_r = v_r_dust - v_r_gas
            delta_v_phi = v_phi_dust - v_phi_gas
            a_map = self.calc_a_from_streamlines(orbit, 'dust')

        elif method == 'statler':
            # a needs to come from list, not specific energy
            a_map, delta_v_r, delta_v_phi = self.velocity_from_epstein_drag(orbit, material, method)

        elif method == 'axisymmetric':
            a_map, delta_v_r, delta_v_phi = self.compute_da_dt_llambay(orbit)
            da_dt = delta_v_r
            return a_map, da_dt, delta_v_r, delta_v_phi

        da_dr = np.gradient(a_map, r_grid, axis=0)
        da_dphi = np.gradient(a_map, phi_grid, axis=1)

        da_dt = da_dr * delta_v_r + da_dphi * delta_v_phi / r_grid[:, None]
        print(np.nanmin(da_dr), np.nanmax(da_dr))

        return a_map, da_dt, delta_v_r, delta_v_phi