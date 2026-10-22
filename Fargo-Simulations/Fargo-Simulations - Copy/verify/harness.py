import sys, time, cProfile, pstats, io, pickle
repo, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]   # mode: stokes | particle
sys.path.insert(0, repo)
import matplotlib; matplotlib.use("Agg")
import numpy as np
from core.data_loader import SimulationDataLoader
from core.variables import SimulationVariablesCalculator
from core.streamlines import SimulationDataStreamlines
from core.processor import SimulationDataProcessor
from analysis.orbital_evolution import da_dt_calculator

PATH = "/home/claude/work/sim/"
ORBIT = 20
loader = SimulationDataLoader(PATH)
var = SimulationVariablesCalculator(loader)
st = SimulationDataStreamlines(loader, var)
proc = SimulationDataProcessor(loader, var, st)
dd = da_dt_calculator(loader, var, st, 1, mode)

res, times = {}, {}
def stage(name, fn):
    t = time.perf_counter(); res[name] = fn(); times[name] = time.perf_counter() - t
    print(f"  {name:28s} {times[name]:7.2f}s", flush=True)

pr = cProfile.Profile(); pr.enable()
stage("get_mu_a_flux",        lambda: proc.get_mu_a_flux(ORBIT, 0))
stage("statler_model",        lambda: proc.statler_model(ORBIT, 0, "no"))
stage("da_dt_statler",        lambda: dd.compute_da_dt_statler(ORBIT))
for m in ("sim", "drag", "statler"):
    stage(f"mass_flux[{m}]",  lambda m=m: dd.compute_da_dt_mass_flux(ORBIT, "dust", m))
stage("2streams[dust]",       lambda: dd.compute_da_dt_2streams(PATH, ORBIT, "dust"))
stage("2streams[gas]",        lambda: dd.compute_da_dt_2streams(PATH, ORBIT, "gas"))
pr.disable()
print(f"  TOTAL                        {sum(times.values()):7.2f}s")

s = io.StringIO(); ps = pstats.Stats(pr, stream=s)
def nc(pattern):
    return sum(v[1] for k, v in ps.stats.items() if pattern in k[0] + ":" + k[2])
counts = {
  "curve_fit calls":               nc("_minpack_py.py:curve_fit"),
  "get_streamline_vectorized":     nc("get_streamline_vectorized"),
  "statler_model":                 nc(":statler_model"),
  "compute_stokes_number":         nc("compute_stokes_number"),
  "scalar interpolate() calls":    sum(v[1] for k, v in ps.stats.items() if k[2] == "interpolate" and "interpolation.py" in k[0]),
  "griddata calls":                nc("_ndgriddata.py:griddata"),
  "np.fromfile":                   nc("fromfile"),
  "read_par":                      nc("get_parameters.py"),
}
print("  call counts:", counts)
pickle.dump({"res": res, "times": times, "counts": counts}, open(out, "wb"))
