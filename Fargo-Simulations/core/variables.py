import numpy as np
from get_parameters import read_par


class SimulationVariablesCalculator:
    """Derives physical quantities (pressure, sound speed, …) from raw fields."""

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def calculate_sound_speed(self):
        params       = read_par(self.data_loader.path)
        aspect_ratio = float(params["ASPECTRATIO"])
        v_K          = np.sqrt(1 / self.data_loader.r_grid[:, None])
        return aspect_ratio * v_K

    def calculate_gas_pressure(self, orbit):
        cs_map = self.calculate_sound_speed()
        Sigma  = self.data_loader.load_gasdens(orbit)
        return Sigma * cs_map ** 2
 
    def calculate_pressure_gradient(self, orbit):
        r_grid   = self.data_loader.r_grid
        pres_map = self.calculate_gas_pressure(orbit)

        dp_dr = np.gradient(pres_map, r_grid, axis=0)

        #log_r    = np.log(r_grid)
 
        #dp_dlnr          = np.zeros_like(pres_map)
        #dp_dlnr[0,  :]   = (pres_map[1,  :] - pres_map[0,  :]) / (log_r[1]  - log_r[0])
        #dp_dlnr[-1, :]   = (pres_map[-1, :] - pres_map[-2, :]) / (log_r[-1] - log_r[-2])
        #d_log_r          = log_r[2:] - log_r[:-2]
        #dp_dlnr[1:-1, :] = (pres_map[2:, :] - pres_map[:-2, :]) / d_log_r[:, None]
 
        return dp_dr
    
    def _cell_dr(self):
        log_r = np.log(self.data_loader.r_grid)
        dr        = np.zeros_like(self.data_loader.r_grid)
        dr[0]     = (log_r[1]  - log_r[0])  / 2
        dr[-1]    = (log_r[-1] - log_r[-2]) / 2
        dr[1:-1]  = (log_r[2:] - log_r[:-2]) / 2
        return dr

    def calculate_gas_mass(self, orbit):
        r_grid = self.data_loader.r_grid
        Sigma  = self.data_loader.load_gasdens(orbit)
        dr     = self._cell_dr()
        dphi   = self.data_loader.phi_grid[1] - self.data_loader.phi_grid[0]
        return Sigma * r_grid[:, None] ** 2 * dr[:, None] * dphi

    def calculate_dust_mass(self, orbit, dust_num):
        r_grid = self.data_loader.r_grid
        Sigma  = self.data_loader.load_dustdens(orbit, dust_num)
        dr     = self._cell_dr()
        dphi   = self.data_loader.phi_grid[1] - self.data_loader.phi_grid[0]
        return Sigma * r_grid[:, None] ** 2 * dr[:, None] * dphi
    
    def calculate_vr(self, orbit, material):
        r_grid = self.data_loader.r_grid
        if material == 'gas':   
            dp_dr = self.calculate_pressure_gradient(orbit)

            v_phi = np.sqrt()
            
