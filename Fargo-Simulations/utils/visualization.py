import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from get_parameters import read_par


# Custom astronomical colormap
astro_blue = LinearSegmentedColormap.from_list(
    "astro_blue",
    [
        "#000000",
        "#0a0f2a",
        "#162a5c",
        "#2b5da8",
        "#4fa3d9",
        "#a6d5e3",
        "#f2f2f2",
    ],
)


cmap_dust = LinearSegmentedColormap.from_list(
    "astro_orange",
    [
        "#000000",  # black
        "#2a150a",  # very dark brown-orange
        "#5c2e16",  # deep burnt orange
        "#a85a2b",  # medium orange
        "#d98f4f",  # warm orange
        "#e3c5a6",  # light peach
        "#f2f2f2"   # near white
    ]
)


class animate_dust_gas:
    """Animate gas density, dust density, and pressure gradient side-by-side."""

    def __init__(self, data_loader, variables_calculator, dust_num):
        self.data_loader          = data_loader
        self.variables_calculator = variables_calculator
        self.dust_num             = dust_num

    def create_animation(self, orbit_list):
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))

        gas_dens  = self.data_loader.load_gasdens(orbit_list[0])
        dust_dens = self.data_loader.load_dustdens(orbit_list[0], self.dust_num)

        pcm_gas = ax[0].pcolormesh(
            self.data_loader.X, self.data_loader.Y,
            np.log10(gas_dens), cmap=astro_blue, vmin=-8, vmax=-4
        )
        pcm_dust = ax[1].pcolormesh(
            self.data_loader.X, self.data_loader.Y,
            np.log10(dust_dens), cmap=cmap_dust, vmin=-12, vmax=-6,
        )
     

        fig.colorbar(pcm_gas,      ax=ax[0], label=r'$\log_{10}(\Sigma_{\rm gas})$')
        fig.colorbar(pcm_dust,     ax=ax[1], label=r'$\log_{10}(\Sigma_{\rm dust})$')

        def _title(panel, label):
            panel.set_title(f"{label}")

        _title(ax[0], "Gas Density")
        _title(ax[1], "Dust Density")

        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        plt.tight_layout()


        def update(frame):
            o = orbit_list[frame]
            gas_dens          = self.data_loader.load_gasdens(o)
            dust_dens         = self.data_loader.load_dustdens(o, self.dust_num)

            pcm_gas.set_array(np.log10(gas_dens).ravel())
            pcm_dust.set_array(np.log10(dust_dens).ravel())

            _title(ax[0], "Gas Density")
            _title(ax[1], "Dust Density")

            ax[0].set_aspect('equal')
            ax[1].set_aspect('equal')
            ax[1].set_title(f"Orbit: {o}")

            plt.tight_layout()


            return pcm_gas, pcm_dust

        anim = FuncAnimation(fig, update, frames=len(orbit_list), blit=False)
        anim.save('density.gif', writer='ffmpeg', fps=2, dpi=300)
        return HTML(anim.to_jshtml())

    def radial_vel_animation(self, orbit_list, save = True):
        params = read_par(self.data_loader.path)
        NINTERM = float(params['NINTERM'])
        DT = float(params['DT'])
        fig, ax = plt.subplots(1, 3, figsize=(22, 6))

        Sigma_0 = self.data_loader.load_gasdens(orbit_list[0])

        v_r_gas, _ = self.data_loader.load_gasvel(orbit_list[0])
        pcm = ax[0].pcolormesh(
            self.data_loader.X, self.data_loader.Y,
            v_r_gas, cmap=astro_blue, vmin=-1e-4, vmax=1e-4,
        )

        #also line plot of v_r vs r in the second panel
        r_grid = self.data_loader.r_grid

        ax[1].plot([], [])
        ax[1].set_xlabel("r")
        ax[1].set_ylabel("v_r")
        ax[1].set_ylim(-1e-5, 1e-5)
        ax[1].axvline([0], color='r', linestyle='--')
        ax[1].set_xscale('log')

        ax[2].plot([], [])  # Placeholder for the third panel

        ax[2].set_xlabel("r")
        ax[2].set_ylabel("Sigma - Sigma_0")
        #ax[2].set_ylim(-1e-5, 1e-5)
        ax[2].axvline([0], color='r', linestyle='--')
        ax[2].set_xscale('log')

        fig.colorbar(pcm, ax=ax[0], label=r'$v_r$')

        def _title(panel, orbit):
            panel.set_title(f"Gas Radial Velocity, orbit: {orbit}")

        _title(ax[0], orbit_list[0])

        def update(frame):
            o = orbit_list[frame]
            v_r_gas, _ = self.data_loader.load_gasvel(o)
            pcm.set_array(v_r_gas.ravel())
            _title(ax[0], o)
            ax[1].clear()
            r_f = (3/2 * o * NINTERM * DT * 0.05 + 0.3**(3/2))**(2/3)
            ax[1].axvline([r_f], color='r', linestyle='--', alpha = 0.5)
            ax[1].plot(r_grid, v_r_gas[:, 0], label="v_r")
            print(v_r_gas[:10, 0])
            ax[1].set_xscale('log')

            ax[1].set_title(f"Radial Velocity Profile, t = {o * NINTERM * DT:.2f}, Snapshot {o}")
            ax[1].set_ylim(-1e-5, 1e-5)
            ax[1].set_xlim(0.3, 30)

            Sigma = self.data_loader.load_gasdens(o)
            ax[2].clear()
            ax[2].plot(r_grid, (Sigma[:, 0] - Sigma_0[:, 0]) / Sigma_0[:, 0], label="(Sigma - Sigma_0) / Sigma_0")
            ax[2].set_title(f"Surface Density Difference, t = {o * NINTERM * DT:.2f}, Snapshot {o}")
            ax[2].set_xscale('log')
            ax[2].set_xlabel("r")
            ax[2].set_ylabel("(Sigma - Sigma_0) / Sigma_0")
            ax[2].set_xlim(0.3, 30)


            return pcm,

        anim = FuncAnimation(fig, update, frames=len(orbit_list), blit=False)

        if save:
            anim.save('radial_velocity_animation.gif', writer='ffmpeg', fps=2)

        return HTML(anim.to_jshtml())
