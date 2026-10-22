import numpy as np
import matplotlib.pyplot as plt
from get_parameters import read_par
from utils.constants import R0_CGS, MSTAR_CGS, MSTAR, R0
from utils.interpolation import interpolate, interpolate_vectorized
from analysis.orbital import fit_ellipse_polar, fit_ellipses, ellipse_radii
from analysis.stokes import relative_velocity_calculator, Stokes_from_Particle
from core.processor import build_a_mesh
from scipy.signal import savgol_filter
import scipy.integrate as integrate

class da_dt_calculator:
    
    def __init__(self, data_loader, variables_calculator, simulation_streamlines, dust_num, dusttype):
        params = read_par(data_loader.path)
        self.data_loader          = data_loader
        self.variables_calculator = variables_calculator
        self.streamlines           = simulation_streamlines
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
        self.rvc              = relative_velocity_calculator(data_loader, variables_calculator, self.streamlines, dust_num, 'statler')
        self.stokes_calc      = Stokes_from_Particle(self.data_loader, self.variables_calculator, self.streamlines, self.dust_num, self.dusttype)

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

        eta = aspect_ratio**2 / 2 * (r_grid[:, None] / pres_map) * dp_dr# * (np.sqrt(2*np.pi) * aspect_ratio * r_grid[:, None])
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

        eta = aspect_ratio**2 / 2 * (r_grid[:, None] / pres_map) * dp_dr# * (np.sqrt(2*np.pi) * aspect_ratio * r_grid[:, None])
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

        eta = aspect_ratio**2 / 2 * (r_grid[:, None] / pres_map) * dp_dr# * (np.sqrt(2*np.pi) * aspect_ratio * r_grid[:, None])
        beta = np.sqrt(1 + 2*eta)

        big_Q_n, big_S_n = self.calc_big_QS_n(orbit)
        psi = self.calc_psi(orbit)
        xi = self.calc_xi(orbit)

        v_K = 1 / np.sqrt(r_grid[:, None])

        delta_vgas_r = -2 * beta * big_Q_n * psi * (beta - 1) * v_K
        delta_vgas_phi = -((big_S_n + 2 * xi) * big_S_n + big_Q_n**2) * psi * (beta - 1) * v_K
        return delta_vgas_r, delta_vgas_phi

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

    def calc_a_from_streamlines(self, orbit, material):
        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid
        r_inits = np.logspace(np.log10(r_grid[0]), np.log10(22), 500)
        a_map_final = np.zeros((len(r_grid), len(phi_grid)))
        #for j, phi_idx in enumerate(range(len(phi_grid))):
        for j, phi_idx in enumerate(range(1)):
            r_vecs, phi_vecs, t_vecs, orbit_times = self.streamlines.get_streamline_vectorized(r_inits, 0, orbit, material, self.dust_num)            
            a_array, e_array, varpi_array = fit_ellipses(r_vecs, phi_grid)
            r_ellipse = ellipse_radii(a_array, e_array, varpi_array, phi_grid)
            a_map = build_a_mesh(r_grid, phi_grid, a_array, r_ellipse)
            a_map_final += a_map
        a_map_final /= (j + 1)
        return a_map_final

    def compute_da_dt_2streams(self, path, orbit, material):

        r_grid = self.data_loader.r_grid
        r_inits = np.logspace(np.log10(r_grid[0]), np.log10(22), 1000)
        r_inits = r_grid

        r_vecs_fwd, phi_vecs_fwd, t_vecs_fwd, t_fwd = self.streamlines.get_streamline_vectorized(
            r_inits, 0, orbit, material, self.dust_num, direction=1
        )
        r_vecs_bwd, phi_vecs_bwd, t_vecs_bwd, t_bwd = self.streamlines.get_streamline_vectorized(
            r_inits, 0, orbit, material, self.dust_num, direction=-1
        )

        # every streamline shares the same phi samples, so one batched (cached) fit per direction
        a_fwd, _, _ = fit_ellipses(r_vecs_fwd, phi_vecs_fwd[0])
        a_bwd, _, _ = fit_ellipses(r_vecs_bwd, phi_vecs_bwd[0])

        a0_array = 0.5 * (a_fwd + a_bwd)
        da_dt_array = (a_fwd - a_bwd) / (np.mean(t_vecs_fwd, axis=1) + np.mean(t_vecs_bwd, axis=1))

        return a0_array, da_dt_array

    def velocity_from_epstein_drag(self, orbit, material, method):
        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid
        params = read_par(self.data_loader.path)
        aspect_ratio = float(params["ASPECTRATIO"])
        v_th = np.sqrt(8 / np.pi) * self.variables_calculator.calculate_sound_speed()
        c_s = self.variables_calculator.calculate_sound_speed()

        if method == 'statler':
            pres_map, dp_dr, Sigma_g, _, a_list, e_list, r_vec_list = self.rvc.calculate_statler_components(orbit)

            a_map = self.calc_a_from_streamlines(orbit, 'dust')   # (a mesh from the Statler lists was built here and immediately overwritten)
            dp_dphi = np.gradient(pres_map, phi_grid, axis=1)

        else:
            Sigma_g = self.data_loader.load_gasdens(orbit)
            dp_dr = self.variables_calculator.calculate_pressure_gradient(orbit)
            pres_map = self.variables_calculator.calculate_gas_pressure(orbit)
            dp_dphi = np.gradient(pres_map, phi_grid, axis=1)
            #a_map = self.calc_a_from_spec_energy(orbit, 'dust')
            a_map = self.calc_a_from_streamlines(orbit, 'dust')

        gasdens = Sigma_g / (np.sqrt(2*np.pi) * aspect_ratio * r_grid[:, None])

        v_K = np.sqrt(1 / a_map) #if method == 'statler' else np.sqrt(1 / r_grid[:, None])
        Omega_K = np.sqrt(1 / a_map**3) #if method == 'statler' else np.sqrt(1 / r_grid[:, None]**3)


        ####### Regular method
        if self.dusttype == 'particle':
            particle_size = float(params[f"PARTICLESIZE{self.dust_num}"]) * R0 / R0_CGS
            rhosolid = float(params["RHOSOLID"]) / (MSTAR_CGS / R0_CGS**3) * (MSTAR / R0**3)
            St, stopping_time = self.stokes_calc.compute_stokes_number(orbit)
            stopping_time = St / Omega_K
            print("Stopping time:", np.nanmin(stopping_time), np.nanmax(stopping_time))
            print(f"Stokes: {np.nanmin(St)}-{np.nanmax(St)}")
            eta = aspect_ratio**2 / 2 * (r_grid[:, None] / pres_map) * dp_dr# * (np.sqrt(2*np.pi) * aspect_ratio * r_grid[:, None])
            beta = np.sqrt(1 + 2*eta)
            delta_v_r = (rhosolid * particle_size / Sigma_g / v_th / gasdens) * dp_dr #/ (1 + St**2)
            #delta_v_r = stopping_time / Sigma_g * dp_dr
            delta_v_phi = (rhosolid * particle_size / Sigma_g / v_th / gasdens) * dp_dphi / r_grid[:, None]
            #delta_v_phi = np.zeros_like(delta_v_r)

        if self.dusttype == 'stokes':
            stopping_time = self.stokes / Omega_K

            delta_v_r = stopping_time / Sigma_g * dp_dr
            delta_v_phi = stopping_time / Sigma_g / r_grid[:, None] * dp_dphi

        return a_map, delta_v_r, delta_v_phi


    def compute_da_dt_statler(self, orbit):
        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid

        a_map, delta_v_r, delta_v_phi = self.velocity_from_epstein_drag(orbit, 'dust', 'statler')
        _, _, _, _, a_list, e_list, r_vec_list = self.rvc.calculate_statler_components(orbit)

        # da/dt on the (r, phi) grid
        da_dr = np.gradient(a_map, r_grid, axis=0)
        da_dphi = np.gradient(a_map, phi_grid, axis=1)
        da_dt = da_dr * delta_v_r + da_dphi * delta_v_phi / r_grid[:, None]   # (n_r, n_phi)

        a = np.asarray(a_list)
        e = np.asarray(e_list)
        r_orb = np.asarray(r_vec_list)   # expected (n_orbits, n_phi); n_orbits need not equal n_r

        print("da_dt:", da_dt.shape, "| a:", a.shape, "| e:", e.shape, "| r_orb:", r_orb.shape)
        assert r_orb.shape == (len(a), len(phi_grid)), (
            f"r_vec_list has shape {r_orb.shape}, expected ({len(a)}, {len(phi_grid)})"
        )

        # Sample da/dt along each orbit: at azimuth phi_j, interpolate in r to the orbit's radius.
        # Orbits that leave [r_grid[0], r_grid[-1]] become NaN and are dropped later by bin_own_range.
        da_dt_orbit = np.full(r_orb.shape, np.nan)
        for j in range(len(phi_grid)):
            da_dt_orbit[:, j] = np.interp(r_orb[:, j], r_grid, da_dt[:, j],
                                          left=np.nan, right=np.nan)

        # 1/(1 + e cos f)^2 = (r / [a(1 - e^2)])^2 on the ellipse, so varpi is not needed
        weight = (r_orb / (a * (1 - e**2))[:, None])**2

        # (1/2pi) * integral over a full circle on a uniform periodic grid = mean over phi
        integral = np.mean(da_dt_orbit * weight, axis=1)
        da_dt_avg = (1 - e**2)**1.5 * integral

        return a, da_dt_avg



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

        #a_map = self.calc_a_from_streamlines(orbit, 'dust')

        #a_map = self.calc_a_from_spec_energy(orbit, 'dust')

        a_map = np.zeros((len(r_grid), len(phi_grid)))
        for i in range(len(phi_grid)):
            a_map[:, i] = r_grid

        v_K = np.sqrt(1 / a_map)
        Omega_K = np.sqrt(1 / a_map**3)

        if self.dusttype =='particle':

            St, stopping_time = self.stokes_calc.compute_stokes_number(orbit)
            print(f"Stokes: {np.nanmin(St)}-{np.nanmax(St)}")

        elif self.dusttype == 'stokes':
            St = self.stokes

        eta = aspect_ratio**2 / 2 * (r_grid[:, None] / pres_map) * dp_dr 
        beta = np.sqrt(1 + 2*eta)

        #delta_v_gas_r, delta_v_gas_phi = self.calc_delta_v_gas(orbit)

        delta_v_r = 2 * St / (1 + St**2) * (beta - 1) * v_K# + (delta_v_gas_r + 2 * St * delta_v_gas_phi) / (1 + St**2)
        delta_v_phi = 1 / (1 + St**2) * (beta - 1) * v_K# + (2 * delta_v_gas_phi - St * delta_v_gas_r) / (2 * (1 + St**2))
        print(a_map.shape, delta_v_r.shape, delta_v_phi.shape)
        return a_map, delta_v_r, delta_v_phi

    
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
            #specific_energy = 0.5 * (v_r_dust**2 + v_phi_dust**2) - 1 / r_grid[:, None]
            delta_v_r = v_r_dust - v_r_gas
            delta_v_phi = v_phi_dust - v_phi_gas
            #a_map = -1 / (2 * specific_energy)
            a_map = self.calc_a_from_streamlines(orbit, 'dust')

        elif method == 'statler':
            # a needs to come from list, not specific energy
            a_map, delta_v_r, delta_v_phi = self.velocity_from_epstein_drag(orbit, material, method)

        elif method == 'axisymmetric':
            a_map, delta_v_r, delta_v_phi = self.compute_da_dt_llambay(orbit)

        da_dr = np.gradient(a_map, r_grid, axis=0)
        da_dphi = np.gradient(a_map, phi_grid, axis=1)

        v_r_dust, v_phi_dust = self.data_loader.load_dustvel(orbit, self.dust_num)
        v_r_gas, v_phi_gas = self.data_loader.load_gasvel(orbit)

        da_dt = da_dr * delta_v_r + da_dphi * delta_v_phi / r_grid[:, None]
        print(np.nanmin(da_dr), np.nanmax(da_dr))

        mask = (a_map > 0) & (a_map < 40)

        #a_map = a_map[mask].flatten()
        #da_dt = da_dt[mask].flatten()
        #delta_v_r = delta_v_r[mask].flatten()
        #delta_v_phi = delta_v_phi[mask].flatten()

        return a_map, da_dt, delta_v_r * da_dr, delta_v_phi