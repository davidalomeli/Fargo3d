"""
Quick-start example — replace PATH and orbit numbers as needed.
"""

from core.data_loader  import SimulationDataLoader
from core.variables    import SimulationVariablesCalculator
from core.streamlines  import SimulationDataStreamlines
from core.processor    import SimulationDataProcessor
from analysis.stokes   import relative_velocity_calculator, Stokes_from_Particle
from analysis.fft_analysis import FFT_1D_Analyzer
from utils.visualization   import animate_dust_gas

PATH  = "/project/phys-k/Fargo3d/fargo_triple_dust_rhodamporig/"
ORBIT = 100
DUST  = 1

loader   = SimulationDataLoader(PATH)
calc     = SimulationVariablesCalculator(loader)
streams  = SimulationDataStreamlines(loader, calc)
proc     = SimulationDataProcessor(loader, calc, streams)

# Statler reconstruction
Sigma, ratio = proc.statler_a2(orbit=ORBIT, phi_index=0, plot="yes")

# Relative velocity
rel_vel = relative_velocity_calculator(loader, calc, dust_num=DUST)
vr_tva  = rel_vel.TVA(ORBIT, PATH, "stokes")

# FFT
fft = FFT_1D_Analyzer(loader)
amp, phase = fft.transform(ORBIT, DUST)
strength, phase_m = fft.construct_features(amp, phase, m=1)

# Animation
anim = animate_dust_gas(loader, calc, dust_num=DUST)
html = anim.create_animation(orbit_list=[100, 110, 120])
