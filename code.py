import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # For LogNorm

# Define path
path = 'outputs/fargo_multifluid_outflow_feel_short/'

# Load grid data
phi = np.loadtxt(path + 'domain_x.dat')[:-1]  # Azimuthal grid (theta)
r = np.loadtxt(path + 'domain_y.dat')[3:-4]  # Radial grid (r)

nphi = len(phi)
nr = len(r)

# Create meshgrid for Cartesian coordinates
PHI, R = np.meshgrid(phi, r)
X = np.cos(PHI) * R
Y = np.sin(PHI) * R

# Load density data
gas_density = np.fromfile(path + 'gasdens21.dat').reshape(nr, nphi)
#dust_density = np.fromfile(path + 'dust1dens21.dat').reshape(nr, nphi)
#density = gas_density + dust_density  # Total density

# Define a shared color scale using LogNorm
#vmin = min(gas_density.min(), dust_density.min())  # Minimum density value
#vmax = max(gas_density.max(), dust_density.max())  # Maximum density value
#norm = colors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)  # Avoid log(0) issues

# --- Plot Gas Density ---
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, np.log10(gas_density), cmap="viridis")  # Use LogNorm
plt.colorbar(label="Density")
plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gas Density")
plt.savefig('Image.png')
plt.show()
