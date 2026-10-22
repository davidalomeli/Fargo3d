import sys, time, pickle
repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
import matplotlib; matplotlib.use("Agg")
import numpy as np
from core.data_loader import SimulationDataLoader
from core.variables import SimulationVariablesCalculator
from core.streamlines import SimulationDataStreamlines
from analysis.stokes import relative_velocity_calculator, Stokes_from_Particle
from analysis.orbital_evolution import da_dt_calculator
PATH = "/home/claude/work/sim/"; ORBIT = 20
loader = SimulationDataLoader(PATH); var = SimulationVariablesCalculator(loader)
st = SimulationDataStreamlines(loader, var)
res, times = {}, {}
def stage(name, fn):
    t = time.perf_counter(); res[name] = fn(); times[name] = time.perf_counter() - t
    print(f"  {name:30s} {times[name]:7.2f}s", flush=True)

for gastype in ("axisymmetric", "aesymmetric", "statler"):
    rvc = relative_velocity_calculator(loader, var, st, 1, gastype)
    stage(f"TVA[{gastype},particle]", lambda: rvc.TVA(ORBIT, PATH, "particle"))
    stage(f"TVA[{gastype},stokes]",   lambda: rvc.TVA(ORBIT, PATH, "stokes"))
    if gastype != "statler":
        stage(f"llambay[{gastype}]",  lambda: rvc.llambay_19(ORBIT, PATH, "stokes"))
sp = Stokes_from_Particle(loader, var, st, 1, "particle")
stage("stokes_number[particle]", lambda: sp.compute_stokes_number(ORBIT))
ss = Stokes_from_Particle(loader, var, st, 1, "stokes")
stage("stokes_number[stokes]",   lambda: ss.compute_stokes_number(ORBIT))
dd = da_dt_calculator(loader, var, st, 1, "stokes")
for m in ("sim", "drag", "statler", "axisymmetric"):
    stage(f"mass_flux[{m},stokes]", lambda m=m: dd.compute_da_dt_mass_flux(ORBIT, "dust", m))
stage("da_dt_statler[stokes]",   lambda: dd.compute_da_dt_statler(ORBIT))
print(f"  TOTAL                          {sum(times.values()):7.2f}s")
pickle.dump({"res": res, "times": times}, open(out, "wb"))
