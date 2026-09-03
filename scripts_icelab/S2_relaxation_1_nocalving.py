# First Relaxation Run (1850-1900 Climatology) with PDD + MAR correction
# SMB forcing: MAR-corrected mean of GISS 1850-1900
# No calving

import os
import sys
import numpy as np
from netCDF4 import Dataset
import datetime
import xarray as xr
from scipy.interpolate import griddata

# Add ISSM paths
sys.path.append("/home/yanmeiti/issm/Functions/")
sys.path.append(os.getenv('ISSM_DIR') + '/bin')
sys.path.append(os.getenv('ISSM_DIR') + '/lib')
sys.path.append(os.getenv('ISSM_DIR') + '/share')
sys.path.append(os.getenv('ISSM_DIR') + '/share/proj')

if 'SMBd18opdd' in sys.path:
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

# =============================================================
# 1. Load Model
# =============================================================
print("Loading model...")
loadname = '/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Models_tym_v2/Step1.5_StressBalance_PostInversion_v2.nc'
md = loadmodel(loadname)
print("Model loaded.")

print("\n--- Checking Initial State (Post-Inversion) ---")
vel       = md.initialization.vel
thickness = md.geometry.thickness
friction  = md.friction.coefficient
print(f"Initial Velocity (m/yr) - Min: {np.min(vel):.2f}, Max: {np.max(vel):.2f}, Mean: {np.mean(vel):.2f}")
print(f"Initial Thickness (m)   - Min: {np.min(thickness):.2f}, Max: {np.max(thickness):.2f}")
print(f"Friction Coefficient    - Min: {np.min(friction):.2f}, Max: {np.max(friction):.2f}")
print(f"StressbalanceSolution.Vel - Min: {np.min(md.results.StressbalanceSolution.Vel):.2f}, Max: {np.max(md.results.StressbalanceSolution.Vel):.2f}")

# =============================================================
# 2. Fix ocean geometry inconsistencies
# =============================================================
ice_mask  = md.mask.ice_levelset < 0
ocean_noice = (~ice_mask) & (md.mask.ocean_levelset < 0)

eps = 1e-3
bad_ocean = ocean_noice & (md.geometry.base <= md.geometry.bed)
md.geometry.base[bad_ocean] = md.geometry.bed[bad_ocean] + eps
md.geometry.thickness[ocean_noice] = np.maximum(
    md.geometry.surface[ocean_noice] - md.geometry.base[ocean_noice], 0.0
)
md.geometry.surface[ocean_noice] = md.geometry.base[ocean_noice] + md.geometry.thickness[ocean_noice]

# =============================================================
# 3. Climate forcing: MAR-corrected GISS 1850-1900 climatology
# =============================================================
print("\nLoading climate data with MAR correction...")

md.smb = SMBd18opdd()

# ── STEP 1: MAR 1981-2010 monthly climatology → presentday baseline ──
MAR_file = "/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Data/MAR/MAR_1981_2010_monthly_climatology_for_ISSM.nc"
ds_MAR = xr.open_dataset(MAR_file)
tt_MAR  = ds_MAR["TT"].values   # (12, Y, X), K
pr_MAR  = ds_MAR["PR"].values   # (12, Y, X), mWE/yr
sh_MAR  = ds_MAR["SH"].values   # (Y, X), m
lat_MAR = ds_MAR["LAT"].values
lon_MAR = ds_MAR["LON"].values
ds_MAR.close()

[md.mesh.lat, md.mesh.long] = xy2ll(md.mesh.x, md.mesh.y, +1)
points_MAR = np.column_stack((lat_MAR.ravel(), lon_MAR.ravel()))

md.smb.precipitations_presentday = np.full((md.mesh.numberofvertices, 12), np.nan)
md.smb.temperatures_presentday   = np.full((md.mesh.numberofvertices, 12), np.nan)

for i in range(12):
    md.smb.precipitations_presentday[:, i] = griddata(
        points_MAR, pr_MAR[i].ravel(), (md.mesh.lat, md.mesh.long), method="nearest"
    )
    md.smb.temperatures_presentday[:, i] = griddata(
        points_MAR, tt_MAR[i].ravel(), (md.mesh.lat, md.mesh.long), method="nearest"
    )

MAR_dem    = griddata(points_MAR, sh_MAR.ravel(), (md.mesh.lat, md.mesh.long), method="nearest")
md.smb.s0p = np.maximum(MAR_dem, 0)
md.smb.s0t = np.maximum(MAR_dem, 0)

print("MAR climatology interpolated to mesh.")
print(f"  MAR T range: {np.nanmin(md.smb.temperatures_presentday):.2f} - {np.nanmax(md.smb.temperatures_presentday):.2f} K")
print(f"  MAR P range: {np.nanmin(md.smb.precipitations_presentday):.4f} - {np.nanmax(md.smb.precipitations_presentday):.4f} m/yr")

# ── STEP 2: Load full GISS historical data (1850-2014) ───────────────
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

lon_giss = (lon_giss + 180) % 360 - 180

lat_idx = np.where((lat_giss >= 50) & (lat_giss <= 90))[0]
lon_idx = np.where((lon_giss >= -80) & (lon_giss <= -5))[0]
i_lat0, i_lat1 = lat_idx.min(), lat_idx.max() + 1
i_lon0, i_lon1 = lon_idx.min(), lon_idx.max() + 1

lat_sub   = lat_giss[i_lat0:i_lat1]
lon_sub   = lon_giss[i_lon0:i_lon1]
tas_sub   = tas_raw[:, i_lat0:i_lat1, i_lon0:i_lon1]
pr_sub    = pr_raw[:,  i_lat0:i_lat1, i_lon0:i_lon1]
pr_reunit = pr_sub * 3.154e7 * (1/1000) * (md.materials.rho_freshwater / md.materials.rho_ice)

print(f"GISS data loaded: {tas_sub.shape[0]} months")

# ── STEP 3: Interpolate GISS to mesh ─────────────────────────────────
num_vertices = md.mesh.numberofvertices
nt           = tas_sub.shape[0]
x_axis       = lon_sub.astype(float)
y_axis       = lat_sub.astype(float)

temp_series   = np.zeros((num_vertices, nt))
precip_series = np.zeros((num_vertices, nt))

print(f"Interpolating {nt} months to mesh...")
for k in range(nt):
    if k % 120 == 0:
        print(f"  month {k}/{nt}")
    temp_series[:, k]   = InterpFromGridToMesh(x_axis, y_axis, tas_sub[k], md.mesh.long, md.mesh.lat, 0)
    precip_series[:, k] = InterpFromGridToMesh(x_axis, y_axis, pr_reunit[k], md.mesh.long, md.mesh.lat, 0)
print("Interpolation done.")

# ── STEP 4: GISS 1981-2010 reference climatology ─────────────────────
years_giss  = 1850 + np.arange(nt) // 12
months_giss = np.arange(nt) % 12

ref_mask         = (years_giss >= 1981) & (years_giss <= 2010)
temp_ref         = temp_series[:, ref_mask].reshape(num_vertices, -1, 12)
precip_ref       = precip_series[:, ref_mask].reshape(num_vertices, -1, 12)
temp_giss_clim   = np.mean(temp_ref,   axis=1)
precip_giss_clim = np.mean(precip_ref, axis=1)
precip_giss_clim[precip_giss_clim <= 0] = 1e-6

# ── STEP 5: Apply MAR correction to 1850-1900 monthly series ─────────
mask_period = (years_giss >= 1850) & (years_giss <= 1899)
idx_period  = np.where(mask_period)[0]
nt_period   = len(idx_period)

tmp_period = np.zeros((num_vertices, nt_period))
pre_period = np.zeros((num_vertices, nt_period))

for j, k in enumerate(idx_period):
    m = months_giss[k]
    dT      = temp_series[:, k] - temp_giss_clim[:, m]
    p_ratio = precip_series[:, k] / precip_giss_clim[:, m]
    tmp_period[:, j] = md.smb.temperatures_presentday[:, m] + dT
    pre_period[:, j] = md.smb.precipitations_presentday[:, m] * p_ratio

pre_period[pre_period <= 0] = 0.1

print("MAR-corrected 1850-1900 T/P:")
print(f"  T: {np.nanmin(tmp_period):.2f} - {np.nanmax(tmp_period):.2f} K")
print(f"  P: {np.nanmin(pre_period):.4f} - {np.nanmax(pre_period):.4f} m/yr")

# ── STEP 6: Average to 12-month climatology ───────────────────────────
months_period = months_giss[idx_period]
for m in range(12):
    m_mask = months_period == m
    md.smb.temperatures_presentday[:, m]   = np.mean(tmp_period[:, m_mask], axis=1)
    md.smb.precipitations_presentday[:, m] = np.mean(pre_period[:, m_mask], axis=1)

print("1850-1900 MAR-corrected climatology ready.")
print(f"  Final T range: {md.smb.temperatures_presentday.min():.2f} - {md.smb.temperatures_presentday.max():.2f} K")
print(f"  Final P range: {md.smb.precipitations_presentday.min():.4f} - {md.smb.precipitations_presentday.max():.4f} m/yr")

# =============================================================
# 4. SMB parameters
# =============================================================
md.smb.isd18opd = 1
md.smb.delta18o = np.array([[-40.0110], [0.0]])
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
md.smb.precipitations_reconstructed[-1, :] = np.linspace(1/12, 12, 12)
md.smb.temperatures_reconstructed[-1, :]   = np.linspace(1/12, 12, 12)

delta18o = np.loadtxt('/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Data/delta18o.data')
md.smb.delta18o = delta18o

print("\nSMB parameters set.")
print(f"  rlaps: {md.smb.rlaps},  pddfac_snow: {md.smb.pddfac_snow[0]},  pddfac_ice: {md.smb.pddfac_ice[0]}")
print(f"  s0t range: {md.smb.s0t.min():.1f} - {md.smb.s0t.max():.1f} m")

# =============================================================
# 5. Relaxation setup and run (1850-1900)
# =============================================================
print("\nSetting up Relaxation...")

time_step = cfl_step(md, md.results.StressbalanceSolution.Vx, md.results.StressbalanceSolution.Vy)
print(f"CFL time step: {time_step:.3f} yr")

md.timestepping.start_time = 1850.0
md.timestepping.final_time = 1900.0
md.timestepping.time_step  = 0.05

md.calving.calvingrate = np.zeros(md.mesh.numberofvertices)
md.basalforcings.floatingice_melting_rate = np.zeros(md.mesh.numberofvertices)
md.basalforcings.groundedice_melting_rate = np.zeros(md.mesh.numberofvertices)

Hmin = 20.0
md.masstransport.spcthickness = np.nan * np.ones(md.mesh.numberofvertices)
H0   = md.geometry.thickness
thin = H0 <= Hmin
md.masstransport.spcthickness[thin] = H0[thin]
print(f"Applied spcthickness on {thin.sum()} vertices (H <= {Hmin} m)")

md.inversion.iscontrol = 0
md.transient.requested_outputs = ['default', 'IceVolume', 'TotalSmb', 'SmbMassBalance']
md.settings.output_frequency = 50  # every 50 steps = every 5 yr at dt=0.02

md.miscellaneous.name = 'relaxation_v2'
md.cluster = generic('name', 'localhost', 'np', 8)
md.verbose  = verbose('solution', True, 'module', False, 'convergence', False)
md.settings.solver_residue_threshold = 1e-4
md.stressbalance.maxiter = 100

print("Starting Relaxation (1850-1900)...")
md = solve(md, 'Transient')
print("Relaxation completed.")

# =============================================================
# 6. Save
# =============================================================
savename = '/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Models_tym_v2/Step2.1_relaxation_nocalving_v2.nc'
export_netCDF(md, savename)
print(f"Model saved to {savename}")

# =============================================================
# 7. Quick diagnostics
# =============================================================
V0 = np.sqrt(md.results.TransientSolution[0].Vx**2  + md.results.TransientSolution[0].Vy**2)
V1 = np.sqrt(md.results.TransientSolution[-1].Vx**2 + md.results.TransientSolution[-1].Vy**2)
H0 = md.results.TransientSolution[0].Thickness
H1 = md.results.TransientSolution[-1].Thickness

print("\n--- Final diagnostics ---")
print(f"Velocity  start: max={np.nanmax(V0):.1f}  p99={np.nanpercentile(V0,99):.1f} m/yr")
print(f"Velocity  end:   max={np.nanmax(V1):.1f}  p99={np.nanpercentile(V1,99):.1f} m/yr")
print(f"Thickness start: max={np.nanmax(H0):.1f}  min={np.nanmin(H0):.1f} m")
print(f"Thickness end:   max={np.nanmax(H1):.1f}  min={np.nanmin(H1):.1f} m")
print(f"SMB end: min={md.results.TransientSolution[-1].SmbMassBalance.min():.2f}  max={md.results.TransientSolution[-1].SmbMassBalance.max():.2f} m/yr")
print(" This experiment is Done.")
