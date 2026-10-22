from core.streamlines import SimulationDataStreamlines
from core.variables import SimulationVariablesCalculator
import numpy as np
import matplotlib.pyplot as plt
from analysis.stokes import relative_velocity_calculator


class FFT_1D_Analyzer:
    """1-D Fourier analysis of disk density fields in azimuth."""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.variables = SimulationVariablesCalculator(self.data_loader)
        self.streamlines = SimulationDataStreamlines(self.data_loader, self.variables)
        self.rvc = relative_velocity_calculator(self.data_loader, self.variables, self.streamlines, 1, 'statler')

    def transform(self, orbit, dust_num):
        nphi    = self.data_loader.nphi
        gasdens = self.data_loader.load_gasdens(orbit)
        Sigma_m = np.fft.fft(gasdens, axis=1)
        return np.abs(Sigma_m) / nphi, np.angle(Sigma_m)

    def construct_features(self, amplitude, phase, m):
        A0             = amplitude[:, 0]
        Am             = amplitude[:, m]
        spiral_strength = Am / A0
        return spiral_strength, phase[:, m]

    def inverse_transform(self, amplitude, phase, m):
        phi_grid = self.data_loader.phi_grid
        Am       = amplitude[:, m]
        phase_m  = phase[:, m]
        return Am[:, None] * np.cos(m * (phi_grid + np.pi) + phase_m[:, None])

    def fft_amplitude_spectrum(self, orbit, dust_num, m_max=15):
        """
        Global Fourier amplitude spectrum A_m vs m, mass-weighted and
        integrated over radius, normalized to the m=0 (axisymmetric) mode.
        """
        path     = self.data_loader.path
        phi_grid = self.data_loader.phi_grid
        r_grid   = self.data_loader.r_grid
        nr, nphi = len(r_grid), len(phi_grid)

        fig, ax  = plt.subplots(1, 1 + dust_num, figsize=(12, 6), squeeze=False)
        prefix   = ["gas", "dust1", "dust2", "dust3"]

        for num, pre in enumerate(prefix[: dust_num + 1]):
            dens = np.fromfile(f"{path}{pre}dens{orbit}.dat").reshape(nr, nphi)

            if pre == "gas":
                pres_map, dp_dr, dens, gas_dens, a_list, e_list, r_vec_list = self.rvc.calculate_statler_components(orbit)

            # FFT along phi at each radius -> Sigma_m(r)
            Sigma_m_r = np.fft.fft(dens, axis=1)  # shape (nr, nphi)

            # mass-weighted radial integral: integrate Sigma_m(r) * r dr over the disk,
            # this collapses the per-radius spectrum into one complex number per m
            dr = np.gradient(r_grid)
            Sigma_m_total = np.sum(Sigma_m_r * r_grid[:, None] * dr[:, None], axis=0)  # shape (nphi,)
            print(Sigma_m_total)

            A0 = np.abs(Sigma_m_total[0])
            m_vals = np.arange(1, m_max + 1)
            Am = np.array([np.abs(Sigma_m_total[m]) / A0 for m in m_vals])

            ax[0, num].plot(m_vals, Am, '-o', ms=3)
            ax[0, num].set_xlabel(r'$m$')
            ax[0, num].set_ylabel(r'Amplitude, $|A_m|/|A_0|$')
            ax[0, num].set_title(pre)
            ax[0, num].grid(True)
        plt.tight_layout()
        plt.show()

