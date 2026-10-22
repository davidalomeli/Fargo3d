import numpy as np
from get_parameters import read_par
from utils.constants import R0_CGS, MSTAR_CGS, MSTAR, R0
from utils.interpolation import interpolate, interpolate_vectorized
from analysis.orbital import fit_ellipse_polar, fit_ellipses
from core.processor import SimulationDataProcessor
import matplotlib.pyplot as plt
from core.processor import build_a_mesh


class Stokes_from_Particle:
    """Compute Stokes number maps from particle size and gas properties."""

    def __init__(self, data_loader, variables_calculator, simulation_streamlines, dust_num, dusttype):
        params = read_par(data_loader.path)
        self.data_loader          = data_loader
        self.variables_calculator = variables_calculator
        self.simulation_streamlines = simulation_streamlines
        self.dust_num             = dust_num
        self.processor          = SimulationDataProcessor(self.data_loader, self.variables_calculator, self.simulation_streamlines)
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

    def compute_stopping_time_map(self, orbit, r_vecs):
        if self.dusttype == 'particle':
            params = read_par(self.data_loader.path)
            aspect_ratio = float(params["ASPECTRATIO"])
            r_grid, phi_grid = self.data_loader.r_grid, self.data_loader.phi_grid
            nr, nphi = len(r_grid), len(phi_grid)

            gas_dens = self.data_loader.load_gasdens(orbit) / (
                np.sqrt(2 * np.pi) * aspect_ratio * r_grid[:, None]
            )

            c_s = self.variables_calculator.calculate_sound_speed()  # shape (nr,)
            v_th = np.sqrt(8 / np.pi) * c_s
            v_th_2d = np.repeat(v_th, nphi, axis=1)

            stopping_time_map = (self.rhosolid * self.particle_size) / (gas_dens * v_th_2d)

        elif self.dusttype == 'stokes':
            r_grid, phi_grid = self.data_loader.r_grid, self.data_loader.phi_grid
            a_list, _, _ = fit_ellipses(r_vecs, phi_grid)
            a_map = build_a_mesh(r_grid, self.data_loader.phi_grid, a_list, r_vecs)
            frequency_map = np.sqrt(1 / a_map**3)

            stopping_time_map = self.stokes / frequency_map

        return stopping_time_map

    def compute_stokes_number(self, orbit):
        r_grid  = self.data_loader.r_grid
        nr, nphi = len(r_grid), len(self.data_loader.phi_grid)
        r_inits = np.logspace(np.log10(r_grid[0]), np.log10(22), 500)
        r_vecs  = self.simulation_streamlines.get_streamline_vectorized(
            r_inits, 0, orbit, "dust", self.dust_num
        )[0]

        stopping_time_map = self.compute_stopping_time_map(orbit, r_vecs)
        frequency_map      = np.zeros((nr, nphi))
        #count_map          = np.zeros_like(frequency_map)

        #phi_idx_all = np.arange(nphi)
        a_list, _, _ = fit_ellipses(r_vecs, self.data_loader.phi_grid)
        #    frequency   = np.sqrt(1 / a**3)  # scalar for this whole streamline

        #    r_idx = np.searchsorted(r_grid, r_vec)
        #    r_idx = np.clip(r_idx, 0, nr - 1)

        #    freq_vals = np.full(nphi, frequency)  # same scalar broadcast per-point
        #    np.add.at(frequency_map, (r_idx, phi_idx_all), freq_vals)
        #    np.add.at(count_map,     (r_idx, phi_idx_all), 1)

        a_map = build_a_mesh(r_grid, self.data_loader.phi_grid, a_list, r_vecs)
        frequency_map = np.sqrt(1 / a_map**3)

        return stopping_time_map * frequency_map, stopping_time_map


class relative_velocity_calculator:
    """TVA and Llambay+19 relative velocity predictions."""

    def __init__(self, data_loader, variables_calculator, streamlines, dust_num, gastype):
        self.data_loader          = data_loader
        self.variables_calculator = variables_calculator
        self.streamlines          = streamlines
        self.dust_num             = dust_num
        self.gastype              = gastype
        self.processor            = SimulationDataProcessor(
            self.data_loader, self.variables_calculator, self.streamlines
        )

    def calculate_relative_velocity(self, orbit):
        gas_vr, _  = self.data_loader.load_gasvel(orbit)
        dust_vr, _ = self.data_loader.load_dustvel(orbit, self.dust_num)
        return dust_vr - gas_vr 
    
    def calculate_axisymmetric_components(self, orbit):

        r_grid = self.data_loader.r_grid
        params = read_par(self.data_loader.path)
        aspect_ratio = float(params["ASPECTRATIO"])
        c_s          = self.variables_calculator.calculate_sound_speed()

        Sigma_1d = np.mean(self.data_loader.load_gasdens(orbit), axis=1)  # shape (nr,)
        Sigma    = np.tile(Sigma_1d[:, np.newaxis], (1, len(self.data_loader.phi_grid)))
        gas_dens = Sigma / (np.sqrt(2*np.pi) * aspect_ratio * r_grid[:, None])
        pres_map = c_s**2 * Sigma

        log_r    = np.log(r_grid)

        dp_dlnr          = np.zeros_like(pres_map)
        dp_dlnr[0,  :]   = (pres_map[1,  :] - pres_map[0,  :]) / (log_r[1]  - log_r[0])
        dp_dlnr[-1, :]   = (pres_map[-1, :] - pres_map[-2, :]) / (log_r[-1] - log_r[-2])
        d_log_r          = log_r[2:] - log_r[:-2]
        dp_dlnr[1:-1, :] = (pres_map[2:, :] - pres_map[:-2, :]) / d_log_r[:, None]

        dp_dr = dp_dlnr / r_grid[:, None]

        return pres_map, dp_dr, Sigma, gas_dens
    
    def calculate_aesymmetric_components(self, orbit):

        r_grid = self.data_loader.r_grid
        params = read_par(self.data_loader.path)
        aspect_ratio = float(params["ASPECTRATIO"])
        c_s          = self.variables_calculator.calculate_sound_speed()

        Sigma_ae_symmetric = self.make_ae_symmetric_disk(self.data_loader.path, orbit)
        gas_dens          = Sigma_ae_symmetric / (np.sqrt(2*np.pi) * aspect_ratio * r_grid[:, None])
        pres_map          = c_s**2 * Sigma_ae_symmetric
        log_r    = np.log(r_grid)

        dp_dlnr          = np.zeros_like(pres_map)
        dp_dlnr[0,  :]   = (pres_map[1,  :] - pres_map[0,  :]) / (log_r[1]  - log_r[0])
        dp_dlnr[-1, :]   = (pres_map[-1, :] - pres_map[-2, :]) / (log_r[-1] - log_r[-2])
        d_log_r          = log_r[2:] - log_r[:-2]
        dp_dlnr[1:-1, :] = (pres_map[2:, :] - pres_map[:-2, :]) / d_log_r[:, None]

        dp_dr = dp_dlnr / r_grid[:, None]

        return pres_map, dp_dr, Sigma_ae_symmetric, gas_dens
    

    def calculate_statler_components(self, orbit):
        r_grid = self.data_loader.r_grid
        params = read_par(self.data_loader.path)
        aspect_ratio = float(params['ASPECTRATIO'])
        c_s = self.variables_calculator.calculate_sound_speed()
        Sigma, a_list, e_list, r_vec_list = self.processor.statler_model(orbit, phi_index=0, plot='no')
        gas_dens = Sigma / (np.sqrt(2 * np.pi) * aspect_ratio * r_grid[:, None])
        pres_map = c_s**2 * Sigma
        dp_dr = np.gradient(pres_map, r_grid, axis = 0)
        return pres_map, dp_dr, Sigma, gas_dens, a_list, e_list, r_vec_list

    def make_ae_symmetric_disk(self, path, orbit):

        r_grid = self.data_loader.r_grid
        phi_grid = self.data_loader.phi_grid

        r_inits = np.logspace(np.log10(r_grid[0]), np.log10(28), 1000)
        phi_idx = 0

        r_vecs_gas = self.streamlines.get_streamline_vectorized(r_inits, phi_idx, orbit, 'gas', 1)[0]

        Sigma_gas = self.data_loader.load_gasdens(orbit)
        Sigma_gas_interp = interpolate_vectorized(r_vecs_gas, 0, Sigma_gas, r_grid, phi_grid)

        Sigma_ae_symmetric = np.zeros_like(Sigma_gas)
        counts = np.zeros_like(Sigma_gas)

        a_list = []
        e_list = []
        sigma_list = []

        a_all, e_all, varpi_all = fit_ellipses(r_vecs_gas, phi_grid)
        for i in range(len(r_inits)):
            a, e, varpi = a_all[i], e_all[i], varpi_all[i]
            r = a * (1 - e**2) / (1 + e * np.cos(phi_grid - varpi))
            r_idx = np.clip(np.searchsorted(r_grid, r) - 1, 0, len(r_grid) - 1)
            Sigma_mean_interp = np.mean(Sigma_gas_interp[i])
            Sigma_ae_symmetric[r_idx, :] += Sigma_mean_interp
            counts[r_idx, :] += 1

            a_list.append(a)
            e_list.append(e)
            sigma_list.append(Sigma_mean_interp)
        Sigma_ae_symmetric = np.where(counts > 0, Sigma_ae_symmetric / counts, np.nan)
        return Sigma_ae_symmetric

    def TVA(self, orbit, path, particle_type):
        params = read_par(path)
        c_s = self.variables_calculator.calculate_sound_speed()
        r_grid = self.data_loader.r_grid    
        if self.gastype == 'axisymmetric':
            _, dp_dr, Sigma, gas_dens = self.calculate_axisymmetric_components(orbit)

        elif self.gastype == 'aesymmetric':
            _, dp_dr, Sigma, gas_dens = self.calculate_aesymmetric_components(orbit)

        elif self.gastype == 'statler':
            _, dp_dr, Sigma, gas_dens, *_ = self.calculate_statler_components(orbit)

        else:
            aspect_ratio = float(params["ASPECTRATIO"])
            dp_dr        = self.variables_calculator.calculate_pressure_gradient(orbit)
            Sigma        = self.data_loader.load_gasdens(orbit)
            gas_dens     = Sigma / (np.sqrt(2 * np.pi) * aspect_ratio * r_grid[:, None])
 
        if particle_type == "particle":
            particle_size = float(params[f"PARTICLESIZE{self.dust_num}"]) * R0 / R0_CGS
            rhosolid      = float(params["RHOSOLID"]) * R0_CGS**3 / MSTAR_CGS * MSTAR / R0**3
            v_thermal     = np.sqrt(8 / np.pi) * c_s
            stopping_time = (rhosolid * particle_size) / (gas_dens * v_thermal)
 
        elif particle_type == "stokes":
            St            = 1 / float(params[f"INVSTOKES{self.dust_num}"])
            vr, vphi      = self.data_loader.load_dustvel(orbit, self.dust_num)
            specific_energy = 0.5 * (vr**2 + vphi**2) - 1 / r_grid[:, None]
            a_map         = -1 / (2 * specific_energy)
            OmegaK        = a_map ** (-1.5)
            stopping_time = St / OmegaK
 
        return stopping_time * (dp_dr / Sigma)

    def llambay_19(self, orbit, path, particle_type):
        params = read_par(path)
        c_s = self.variables_calculator.calculate_sound_speed()
        r_grid = self.data_loader.r_grid
        aspect_ratio = float(params["ASPECTRATIO"])

        if self.gastype == 'axisymmetric':
            pres_map, dp_dr, Sigma, gas_dens = self.calculate_axisymmetric_components(orbit)

        elif self.gastype == 'aesymmetric':
            pres_map, dp_dr, Sigma, gas_dens = self.calculate_aesymmetric_components(orbit)

        else:
            pres_map     = self.variables_calculator.calculate_gas_pressure(orbit)
            dp_dr        = self.variables_calculator.calculate_pressure_gradient(orbit)
            Sigma        = self.data_loader.load_gasdens(orbit)
            gas_dens     = Sigma / (np.sqrt(2 * np.pi) * aspect_ratio * r_grid[:, None])
 
 
        dlnp_dlnr = r_grid[:, None] / pres_map * dp_dr
        eta       = aspect_ratio**2 / 2 * dlnp_dlnr
        beta      = np.sqrt(1 + 2 * eta)
        v_K       = np.sqrt(1 / r_grid)


        if particle_type == 'particle':
            v_thermal = np.sqrt(8 / np.pi) * c_s
            particle_size = float(params[f"PARTICLESIZE{self.dust_num}"]) * R0 / R0_CGS
            rhosolid      = float(params["RHOSOLID"]) / (MSTAR_CGS / R0_CGS**3) * (MSTAR / R0**3)
            stopping_time = (rhosolid * particle_size) / (gas_dens * v_thermal)
            St            = stopping_time * np.sqrt(1 / r_grid**3)[:, None]
            #St = np.sqrt(8 / np.pi) * (rhosolid * particle_size)
            print(f"Stokes number range: {np.nanmin(St)} - {np.nanmax(St)}")
            fig, ax = plt.subplots()

            #pcm = ax.pcolormesh(self.data_loader.X, self.data_loader.Y, np.log10(St), shading='auto')
            #fig.colorbar(pcm, ax=ax, label='Stokes number')
            #ax.set_title('Stokes number map') 
            ax.plot(r_grid, St[:, 0])
            ax.set_xlabel('r')
            ax.set_ylabel('Stokes number')
            ax.set_yscale('log')
            ax.set_xscale('log')
            #plt.show()

        elif particle_type == 'stokes':
            St = 1 / float(params[f"INVSTOKES{self.dust_num}"])

        return 2 * St / (1 + St**2) * (beta - 1) * v_K[:, None], St
    
