# Continue Calving Relaxation Run (2001-2100) with PDD + MAR correction + calving
# Loads from: Step2.2_relaxation_calving_100yr.nc
# SMB forcing: MAR-corrected mean of GISS 1901-2000
# Von Mises calving enabled

import os
import sys
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from scipy.interpolate import griddata

# Add ISSM paths
sys.path.append("/home/yanmeiti/issm/Functions/")
sys.path.append(os.getenv('ISSM_DIR') + '/bin')
sys.path.append(os.getenv('ISSM_DIR') + '/lib')
sys.path.append(os.getenv('ISSM_DIR') + '/share')
sys.path.append(os.getenv('ISSM_DIR') + '/share/proj')

if 'SMBd18opdd' in sys.modules:
    sys.modules.pop('SMBd18opdd', None)

my_function_path = '/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Functions/'
if my_function_path not in sys.path:
    sys.path.insert(0, my_function_path)

from model import *
from loadmodel import loadmodel
from InterpFromGridToMesh import InterpFromGridToMesh
from ll2xy import ll2xy
from solve import solve
from export_netCDF import export_netCDF
from generic import generic
from xy2ll import xy2ll
from cfl_step_v2 import cfl_step
from calvingvonmises import calvingvonmises

# =============================================================
# 1. Load previous calving relaxation result
# =============================================================
print("Loading model...")
loadname = '/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Models_tym_v2/Step2.2_relaxation_calving_100yr.nc'
md = loadmodel(loadname)
print("Model loaded.")

# update geometry and initialization from last relaxation step
md.geometry.thickness = md.results.TransientSolution[-1].Thickness.copy()
md.geometry.base      = md.results.TransientSolution[-1].Base.copy()
md.geometry.base      = np.maximum(md.geometry.base, md.geometry.bed)
md.geometry.surface   = md.geometry.base + md.geometry.thickness

md.initialization.vel = md.results.TransientSolution[-1].Vel.copy()
md.initialization.vx  = md.results.TransientSolution[-1].Vx.copy()
md.initialization.vy  = md.results.TransientSolution[-1].Vy.copy()
md.initialization.pressure = md.results.TransientSolution[-1].Pressure.copy()
if hasattr(md.results.TransientSolution[-1], 'MaskIceLevelset'):
    md.mask.ice_levelset = md.results.TransientSolution[-1].MaskIceLevelset.copy()


print(f"Surface range: {md.geometry.surface.min():.1f} - {md.geometry.surface.max():.1f} m")
print(f"Velocity range: {md.initialization.vel.min():.1f} - {md.initialization.vel.max():.1f} m/yr")

# =============================================================
# 2. Climate forcing: MAR-corrected GISS 1901-2000 climatology
# =============================================================
print("\nLoading climate data with MAR correction...")

md.smb = SMBd18opdd()

# ── STEP 1: MAR 1981-2010 monthly climatology → presentday baseline ──
MAR_file = "/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Data/MAR/MAR_1981_2010_monthly_climatology_for_ISSM.nc"
ds_MAR  = xr.open_dataset(MAR_file)
tt_MAR  = ds_MAR["TT"].values
pr_MAR  = ds_MAR["PR"].values
sh_MAR  = ds_MAR["SH"].values
lat_MAR = ds_MAR["LAT"].values
lon_MAR = ds_MAR["LON"].values
ds_MAR.close()

[md.mesh.lat, md.mesh.long] = xy2ll(md.mesh.x, md.mesh.y, +1)
points_MAR = np.column_stack((lat_MAR.ravel(), lon_MAR.ravel()))

md.smb.precipitations_presentday = np.full((md.mesh.numberofvertices, 12), np.nan)
md.smb.temperatures_presentday   = np.full((md.mesh.numberofvertices, 12), np.nan)

for i in range(12):
    md.smb.precipitations_presentday[:, i] = griddata(
        points_MAR, pr_MAR[i].ravel(), (md.mesh.lat, md.mesh.long), method="nearest")
    md.smb.temperatures_presentday[:, i] = griddata(
        points_MAR, tt_MAR[i].ravel(), (md.mesh.lat, md.mesh.long), method="nearest")

MAR_dem    = griddata(points_MAR, sh_MAR.ravel(), (md.mesh.lat, md.mesh.long), method="nearest")
md.smb.s0p = np.maximum(MAR_dem, 0)
md.smb.s0t = np.maximum(MAR_dem, 0)

print(f"  MAR T range: {np.nanmin(md.smb.temperatures_presentday):.2f} - {np.nanmax(md.smb.temperatures_presentday):.2f} K")
print(f"  MAR P range: {np.nanmin(md.smb.precipitations_presentday):.4f} - {np.nanmax(md.smb.precipitations_presentday):.4f} m/yr")

# ── STEP 2: Load full GISS historical data (1850-2014) ──
tas_file = "/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/GISS_Data/ssp245/r1i1p1f2/historic/tas_historical_r1i1p1f2_185001-201412.nc"
pr_file  = "/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/GISS_Data/ssp245/r1i1p1f2/historic/pr_historical_r1i1p1f2_185001-201412.nc"

nc_tas   = Dataset(tas_file)
lat_giss = nc_tas.variables["lat"][:]
lon_giss = nc_tas.variables["lon"][:]
tas_raw  = nc_tas.variables["tas"][:]
nc_tas.close()

nc_pr  = Dataset(pr_file)
pr_raw = nc_pr.variables["pr"][:]
nc_pr.close()

lon_giss  = (lon_giss + 180) % 360 - 180
lat_idx   = np.where((lat_giss >= 50) & (lat_giss <= 90))[0]
lon_idx   = np.where((lon_giss >= -80) & (lon_giss <= -5))[0]
i_lat0, i_lat1 = lat_idx.min(), lat_idx.max() + 1
i_lon0, i_lon1 = lon_idx.min(), lon_idx.max() + 1

lat_sub   = lat_giss[i_lat0:i_lat1]
lon_sub   = lon_giss[i_lon0:i_lon1]
tas_sub   = tas_raw[:, i_lat0:i_lat1, i_lon0:i_lon1]
pr_sub    = pr_raw[:,  i_lat0:i_lat1, i_lon0:i_lon1]
pr_reunit = pr_sub * 3.154e7 * (1/1000) * (md.materials.rho_freshwater / md.materials.rho_ice)

print(f"GISS data loaded: {tas_sub.shape[0]} months")

# ── STEP 3: Interpolate GISS to mesh ──
num_vertices = md.mesh.numberofvertices
nt     = tas_sub.shape[0]
x_axis = lon_sub.astype(float)
y_axis = lat_sub.astype(float)

temp_series   = np.zeros((num_vertices, nt))
precip_series = np.zeros((num_vertices, nt))

print(f"Interpolating {nt} months to mesh...")
for k in range(nt):
    if k % 120 == 0:
        print(f"  month {k}/{nt}")
    temp_series[:, k]   = InterpFromGridToMesh(x_axis, y_axis, tas_sub[k], md.mesh.long, md.mesh.lat, 0)
    precip_series[:, k] = InterpFromGridToMesh(x_axis, y_axis, pr_reunit[k], md.mesh.long, md.mesh.lat, 0)
print("Interpolation done.")

# ── STEP 4: GISS 1981-2010 reference climatology ──
years_giss  = 1850 + np.arange(nt) // 12
months_giss = np.arange(nt) % 12

ref_mask         = (years_giss >= 1981) & (years_giss <= 2010)
temp_ref         = temp_series[:, ref_mask].reshape(num_vertices, -1, 12)
precip_ref       = precip_series[:, ref_mask].reshape(num_vertices, -1, 12)
temp_giss_clim   = np.mean(temp_ref,   axis=1)
precip_giss_clim = np.mean(precip_ref, axis=1)
precip_giss_clim[precip_giss_clim <= 0] = 1e-6

# ── STEP 5: Apply MAR correction to 1901-2000 monthly series ──
mask_period = (years_giss >= 1901) & (years_giss <= 2000)
idx_period  = np.where(mask_period)[0]
nt_period   = len(idx_period)

tmp_period = np.zeros((num_vertices, nt_period))
pre_period = np.zeros((num_vertices, nt_period))

for j, k in enumerate(idx_period):
    m       = months_giss[k]
    dT      = temp_series[:, k] - temp_giss_clim[:, m]
    p_ratio = precip_series[:, k] / precip_giss_clim[:, m]
    tmp_period[:, j] = md.smb.temperatures_presentday[:, m] + dT
    pre_period[:, j] = md.smb.precipitations_presentday[:, m] * p_ratio

pre_period[pre_period <= 0] = 0.1

# ── STEP 6: Average to 1901-2000 monthly climatology ──
months_period = months_giss[idx_period]
for m in range(12):
    m_mask = months_period == m
    md.smb.temperatures_presentday[:, m]   = np.mean(tmp_period[:, m_mask], axis=1)
    md.smb.precipitations_presentday[:, m] = np.mean(pre_period[:, m_mask], axis=1)

print("1901-2000 MAR-corrected climatology ready.")
print(f"  T range: {md.smb.temperatures_presentday.min():.2f} - {md.smb.temperatures_presentday.max():.2f} K")
print(f"  P range: {md.smb.precipitations_presentday.min():.4f} - {md.smb.precipitations_presentday.max():.4f} m/yr")

# =============================================================
# 3. SMB parameters
# =============================================================
md.smb.isd18opd = 1
md.smb.rlaps    = 6.0
md.smb.desfac   = 1
md.smb.rlapslgm = 5.5

md.smb.issetpddfac  = 1
md.smb.pddfac_snow  = 4.0 * np.ones(md.mesh.numberofvertices)
md.smb.pddfac_ice   = 8.0 * np.ones(md.mesh.numberofvertices)
md.smb.isprecipscaled      = 0
md.smb.istemperaturescaled = 0

md.smb.precipitations_reconstructed = np.zeros((md.mesh.numberofvertices + 1, 12))
md.smb.temperatures_reconstructed   = np.zeros((md.mesh.numberofvertices + 1, 12))
md.smb.precipitations_reconstructed[:-1, :] = md.smb.precipitations_presentday
md.smb.temperatures_reconstructed[:-1, :]   = md.smb.temperatures_presentday
md.smb.precipitations_reconstructed[-1, :]  = np.linspace(1/12, 12, 12)
md.smb.temperatures_reconstructed[-1, :]    = np.linspace(1/12, 12, 12)

delta18o = np.loadtxt('/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Data/delta18o.data')
md.smb.delta18o = delta18o

print(f"\nSMB set. rlaps={md.smb.rlaps}, pddfac_snow={md.smb.pddfac_snow[0]}, pddfac_ice={md.smb.pddfac_ice[0]}")

# =============================================================
# 4. Transient settings
# =============================================================
print("\nSetting up transient...")

# FIX: use actual model velocity (from relaxation) for CFL, not MEaSUREs obs
time_step = cfl_step(md, md.initialization.vx, md.initialization.vy)
print(f"CFL time step: {time_step:.3f} yr")

md.timestepping.start_time = 2001.0
md.timestepping.final_time = 2100.0
md.timestepping.time_step  = 0.05

md.transient.isthermal      = 0
md.transient.ismovingfront  = 1
md.inversion.iscontrol      = 0
md.transient.requested_outputs = ['default', 'IceVolume', 'TotalSmb', 'SmbMassBalance', 'IceVolumeAboveFloatation']
md.settings.output_frequency = 20   # every 20 steps × 0.05 yr = 1 yr

# =============================================================
# 5. Von Mises calving
# =============================================================
md.levelset.spclevelset = np.full(md.mesh.numberofvertices, np.nan)
pos = np.union1d(np.where(md.mesh.vertexonboundary > 0)[0],
                 np.where(md.mask.ice_levelset > 0)[0])
md.levelset.spclevelset[pos] = md.mask.ice_levelset[pos]

md.levelset.reinit_frequency = 5
md.levelset.stabilization    = 1
md.thermal.reltol             = 0.01
md.thermal.penalty_lock       = 2
md.thermal.penalty_threshold  = 10

md.calving = calvingvonmises()
md.calving.stress_threshold_floatingice = 200e3   # Pa
md.calving.stress_threshold_groundedice = 1e6
md.calving.min_thickness                = 2
md.frontalforcings.meltingrate = np.zeros(md.mesh.numberofvertices)
md.levelset.kill_icebergs      = 1

# =============================================================
# 6. Solve
# =============================================================
md.miscellaneous.name = 'GrIS_relax_calving_200yr_v2'
md.cluster = generic('name', 'localhost', 'np', 8)
md.verbose  = verbose('solution', True, 'module', False, 'convergence', False)
md.settings.solver_residue_threshold = 1e-4
md.stressbalance.maxiter = 100

print("Starting calving relaxation continuation (2001-2100)...")
md = solve(md, 'Transient')
print("Relaxation completed.")

# =============================================================
# 7. Save
# =============================================================
savename = '/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Models_tym_v2/Step2.2_relaxation_calving_200yr.nc'
export_netCDF(md, savename)
print(f"Model saved to {savename}")

# =============================================================
# 8. Quick diagnostics
# =============================================================
n  = len(md.results.TransientSolution)
V0 = md.results.TransientSolution[0].Vel
V1 = md.results.TransientSolution[-1].Vel
H1 = md.results.TransientSolution[-1].Thickness

print("\n--- Final diagnostics ---")
print(f"Velocity start: max={np.nanmax(V0):.1f}  p99={np.nanpercentile(V0,99):.1f} m/yr")
print(f"Velocity end:   max={np.nanmax(V1):.1f}  p99={np.nanpercentile(V1,99):.1f} m/yr")
print(f"Thickness end:  max={np.nanmax(H1):.1f}  min={np.nanmin(H1):.1f} m")
print(f"SMB end: min={md.results.TransientSolution[-1].SmbMassBalance.min():.2f}  max={md.results.TransientSolution[-1].SmbMassBalance.max():.2f} m/yr")

vol = [md.results.TransientSolution[i].IceVolume[0] for i in range(n)]
rate_early = (vol[min(9,n-1)] - vol[0]) / min(9,n-1) if n > 1 else 0
rate_late  = (vol[-1] - vol[max(-10,-n)]) / min(9,n-1) if n > 1 else 0
print(f"Ice volume change rate - early: {rate_early:.3e}  late: {rate_late:.3e} m³/yr")
if rate_early != 0:
    print(f"Convergence ratio: {abs(rate_late/rate_early):.3f}")
print("Done.")
