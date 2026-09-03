# Forward Simulation 1850-2500 (SSP2-4.5, r3i1p1f2)
# 50-year segmented solves: each segment reads overlapping GISS/SMR forcing chunks
#   1850-1900, 1900-1950, ..., 2450-2500

import os
import sys
import numpy as np
from netCDF4 import Dataset, num2date
import xarray as xr
import h5py
from scipy.interpolate import griddata

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
from xy2ll import xy2ll
from solve import solve
from export_netCDF import export_netCDF
from generic import generic
from cfl_step_v2 import cfl_step
from calvingvonmises import calvingvonmises
from reinitializelevelset import reinitializelevelset

# =============================================================
# Configuration — change MEMBER to switch ensemble
# =============================================================
MEMBER   = 'r9i1p1f2'
MODELS   = '/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Models_tym_v2'
GISS_BASE = f'/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/GISS_Data/ssp245/{MEMBER}'
SMR_BASE  = f'/home/yanmeiti/issm/TF/giss-forcing-pipeline/giss_data/{MEMBER}/output'
MAR_FILE  = '/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Data/MAR/MAR_1981_2010_monthly_climatology_for_ISSM.nc'
DELTA18O  = '/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Data/delta18o.data'
RELAX_NC  = f'{MODELS}/Step2.2_relaxation_calving_200yr.nc'   # ice model state (same for all members)

H = f'{GISS_BASE}/historic/tas_Amon_GISS-E2-1-G_historical_{MEMBER}_gn_'
P = f'{GISS_BASE}/historic/pr_Amon_GISS-E2-1-G_historical_{MEMBER}_gn_'
F = f'{GISS_BASE}/future/tas_Amon_GISS-E2-1-G_ssp245_{MEMBER}_gn_'
Q = f'{GISS_BASE}/future/pr_Amon_GISS-E2-1-G_ssp245_{MEMBER}_gn_'

# 50-year segmented periods. Each period is one solve and one restart/output file.
# giss_files: list of (tas_file, pr_file, epoch_year)
ALL_GISS_FILES = [
    (1850, 1900, f'{H}185001-190012.nc', f'{P}185001-190012.nc', 1850.0),
    (1901, 1950, f'{H}190101-195012.nc', f'{P}190101-195012.nc', 1850.0),
    (1951, 2000, f'{H}195101-200012.nc', f'{P}195101-200012.nc', 1850.0),
    (2001, 2014, f'{H}200101-201412.nc', f'{P}200101-201412.nc', 1850.0),
    (2015, 2050, f'{F}201501-205012.nc', f'{Q}201501-205012.nc', 2015.0),
    (2051, 2100, f'{F}205101-210012.nc', f'{Q}205101-210012.nc', 2015.0),
    (2101, 2150, f'{F}210101-215012.nc', f'{Q}210101-215012.nc', 2015.0),
    (2151, 2200, f'{F}215101-220012.nc', f'{Q}215101-220012.nc', 2015.0),
    (2201, 2250, f'{F}220101-225012.nc', f'{Q}220101-225012.nc', 2015.0),
    (2251, 2300, f'{F}225101-230012.nc', f'{Q}225101-230012.nc', 2015.0),
    (2301, 2350, f'{F}230101-235012.nc', f'{Q}230101-235012.nc', 2015.0),
    (2351, 2400, f'{F}235101-240012.nc', f'{Q}235101-240012.nc', 2015.0),
    (2401, 2450, f'{F}240101-245012.nc', f'{Q}240101-245012.nc', 2015.0),
    (2451, 2500, f'{F}245101-250012.nc', f'{Q}245101-250012.nc', 2015.0),
]

def _smr_path(base, member, y0, y1):
    """Return SMR file path, trying hyphen separator first then underscore."""
    p_hyphen = f'{base}/SMR_lenearts_E2-1-G_{member}_{y0}-{y1}.nc'
    p_under  = f'{base}/SMR_lenearts_E2-1-G_{member}_{y0}_{y1}.nc'
    if os.path.exists(p_hyphen):
        return p_hyphen
    if os.path.exists(p_under):
        return p_under
    return p_hyphen  # fall back so error message shows expected name

ALL_SMR_FILES = [
    (1850, 2014, _smr_path(SMR_BASE, MEMBER, 1850, 2014), 1850.0),
    (2015, 2100, _smr_path(SMR_BASE, MEMBER, 2015, 2100), 2015.0),
    (2101, 2300, _smr_path(SMR_BASE, MEMBER, 2101, 2300), 2101.0),
    (2301, 2500, _smr_path(SMR_BASE, MEMBER, 2301, 2500), 2301.0),
]

def _overlaps(file_start, file_end, period_start, period_final):
    return (file_start < period_final) and (file_end >= period_start)

def _select_giss_files(period_start, period_final):
    selected = []
    for y0, y1, tas_f, pr_f, epoch in ALL_GISS_FILES:
        if _overlaps(y0, y1, period_start, period_final):
            selected.append((tas_f, pr_f, epoch))
    return selected

def _select_smr_files(period_start, period_final):
    selected = []
    for y0, y1, smr_f, epoch in ALL_SMR_FILES:
        if _overlaps(y0, y1, period_start, period_final):
            selected.append((smr_f, epoch))
    return selected

MAJOR_PERIODS = []
period_edges = list(range(1850, 2501, 50))
previous_save = RELAX_NC
for start, final in zip(period_edges[:-1], period_edges[1:]):
    label = f'{start}-{final}'
    save_as = f'{MODELS}/{MEMBER}_forward_{label}_lenearts.nc'
    MAJOR_PERIODS.append({
        'label'      : label,
        'start_time' : float(start),
        'final_time' : float(final),
        'load_from'  : previous_save,
        'save_as'    : save_as,
        'smr_files'  : _select_smr_files(start, final),
        'smr_years'  : (start, final),
        'giss_files' : _select_giss_files(start, final),
    })
    previous_save = save_as

def output_complete(nc_path, final_time, tol=1.0e-6):
    if not os.path.exists(nc_path) or os.path.getsize(nc_path) <= 0:
        return False
    try:
        with Dataset(nc_path, 'r') as ds:
            time = ds.groups['results'].groups['TransientSolution'].variables['time']
            if len(time) == 0:
                return False
            return float(time[-1]) >= final_time - tol
    except Exception as exc:
        print(f'  Existing output is not readable/complete: {os.path.basename(nc_path)} ({exc})')
        return False


# =============================================================
# STEP 0: Load reference model for mesh info + MAR climatology
# =============================================================
print('='*60)
print(f'Member: {MEMBER}')
print('Loading reference model and MAR climatology...')
REFERENCE_NC = next(
    (period['save_as'] for period in MAJOR_PERIODS
     if output_complete(period['save_as'], period['final_time'])),
    RELAX_NC,
)
print(f'Reference model: {os.path.basename(REFERENCE_NC)}')
md0 = loadmodel(REFERENCE_NC)
[md0.mesh.lat, md0.mesh.long] = xy2ll(md0.mesh.x, md0.mesh.y, +1)
num_vertices = md0.mesh.numberofvertices
rho_ratio    = md0.materials.rho_freshwater / md0.materials.rho_ice

ds_MAR  = xr.open_dataset(MAR_FILE)
tt_MAR  = ds_MAR['TT'].values
pr_MAR  = ds_MAR['PR'].values
sh_MAR  = ds_MAR['SH'].values
lat_MAR = ds_MAR['LAT'].values
lon_MAR = ds_MAR['LON'].values
ds_MAR.close()

pts = np.column_stack((lat_MAR.ravel(), lon_MAR.ravel()))
precip_presentday = np.full((num_vertices, 12), np.nan)
temp_presentday   = np.full((num_vertices, 12), np.nan)
for i in range(12):
    precip_presentday[:, i] = griddata(pts, pr_MAR[i].ravel(), (md0.mesh.lat, md0.mesh.long), method='nearest')
    temp_presentday[:, i]   = griddata(pts, tt_MAR[i].ravel(), (md0.mesh.lat, md0.mesh.long), method='nearest')
MAR_dem = griddata(pts, sh_MAR.ravel(), (md0.mesh.lat, md0.mesh.long), method='nearest')
print('MAR done.')

# =============================================================
# STEP 1: GISS 1981-2010 reference climatology
# Same approach as S3: load combined historical file, use integer year index
# =============================================================
print('Computing 1981-2010 reference climatology...')

hist_combined_tas = f'{GISS_BASE}/historic/tas_historical_{MEMBER}_185001-201412.nc'
hist_combined_pr  = f'{GISS_BASE}/historic/pr_historical_{MEMBER}_185001-201412.nc'

nc = Dataset(hist_combined_tas)
lat_g   = nc.variables['lat'][:]
lon_g   = nc.variables['lon'][:]
tas_all = nc.variables['tas'][:]   # (1980, lat, lon)
nc.close()
nc = Dataset(hist_combined_pr)
pr_all  = nc.variables['pr'][:]
nc.close()

lon_g_conv = (lon_g + 180) % 360 - 180
la_idx = np.where((lat_g >= 50) & (lat_g <= 90))[0]
lo_idx = np.where((lon_g_conv >= -80) & (lon_g_conv <= -5))[0]
la0, la1 = la_idx.min(), la_idx.max() + 1
lo0, lo1 = lo_idx.min(), lo_idx.max() + 1
x_ref = lon_g_conv[lo0:lo1].astype(float)
y_ref = lat_g[la0:la1].astype(float)

# integer year from 1850, same as S3
nt_hist   = tas_all.shape[0]
years_all = 1850 + np.arange(nt_hist) // 12
ref_mask  = (years_all >= 1981) & (years_all <= 2010)
print(f'  Reference months: {ref_mask.sum()} (should be 360)')

tas_ref = tas_all[ref_mask, la0:la1, lo0:lo1]
pr_ref  = pr_all[ref_mask,  la0:la1, lo0:lo1] * 3.154e7 / 1000 * rho_ratio
del tas_all, pr_all

t_ref = np.zeros((num_vertices, ref_mask.sum()))
p_ref = np.zeros((num_vertices, ref_mask.sum()))
for k in range(ref_mask.sum()):
    if k % 60 == 0:
        print(f'    {k}/{ref_mask.sum()}')
    t_ref[:, k] = InterpFromGridToMesh(x_ref, y_ref, tas_ref[k], md0.mesh.long, md0.mesh.lat, 0)
    p_ref[:, k] = InterpFromGridToMesh(x_ref, y_ref, pr_ref[k],  md0.mesh.long, md0.mesh.lat, 0)
del tas_ref, pr_ref

temp_giss_clim   = np.mean(t_ref.reshape(num_vertices, -1, 12), axis=1)
precip_giss_clim = np.mean(p_ref.reshape(num_vertices, -1, 12), axis=1)
precip_giss_clim[precip_giss_clim <= 0] = 1e-6
del t_ref, p_ref, md0
print('Reference climatology done.')

# =============================================================
# Helper: build T/P arrays from list of GISS files
# =============================================================
def build_tp_arrays(giss_files, mesh_long, mesh_lat):
    """Read GISS files one by one, interpolate, apply MAR correction, concatenate."""
    tmp_list = []
    pre_list = []
    time_list = []

    for fi, (tas_f, pr_f, epoch_yr) in enumerate(giss_files):
        print(f'  [{fi+1}/{len(giss_files)}] {os.path.basename(tas_f)}')

        nc = Dataset(tas_f)
        lat_f  = nc.variables['lat'][:]
        lon_f  = nc.variables['lon'][:]
        time_var = nc.variables['time']
        t_days = time_var[:]
        units = time_var.units
        calendar = getattr(time_var, 'calendar', 'standard')
        dates = num2date(t_days, units=units, calendar=calendar)
        tas    = nc.variables['tas'][:]
        nc.close()
        nc = Dataset(pr_f)
        pr = nc.variables['pr'][:]
        nc.close()

        years = np.array([d.year for d in dates], dtype=int)
        months = np.array([d.month - 1 for d in dates], dtype=int)
        time_c = years.astype(float) + (months + 0.5) / 12.0
        nt     = len(time_c)

        lon_c2 = (lon_f + 180) % 360 - 180
        la = np.where((lat_f >= 50) & (lat_f <= 90))[0]
        lo = np.where((lon_c2 >= -80) & (lon_c2 <= -5))[0]
        la0_, la1_ = la.min(), la.max()+1
        lo0_, lo1_ = lo.min(), lo.max()+1
        x_ax = lon_c2[lo0_:lo1_].astype(float)
        y_ax = lat_f[la0_:la1_].astype(float)

        tas_s = tas[:, la0_:la1_, lo0_:lo1_]
        pr_s  = pr[:,  la0_:la1_, lo0_:lo1_] * 3.154e7 / 1000 * rho_ratio
        del tas, pr

        ts = np.zeros((num_vertices, nt))
        ps = np.zeros((num_vertices, nt))
        print(f'    Interpolating {nt} months...')
        for k in range(nt):
            if k % 120 == 0:
                print(f'      {k}/{nt}')
            ts[:, k] = InterpFromGridToMesh(x_ax, y_ax, tas_s[k], mesh_long, mesh_lat, 0)
            ps[:, k] = InterpFromGridToMesh(x_ax, y_ax, pr_s[k],  mesh_long, mesh_lat, 0)
        del tas_s, pr_s

        tmp_c = np.zeros((num_vertices, nt))
        pre_c = np.zeros((num_vertices, nt))
        for k in range(nt):
            m = months[k]
            tmp_c[:, k] = temp_presentday[:, m]   + (ts[:, k] - temp_giss_clim[:, m])
            pre_c[:, k] = precip_presentday[:, m] * (ps[:, k] / precip_giss_clim[:, m])
        pre_c[pre_c <= 0] = 0.1
        del ts, ps

        tmp_list.append(tmp_c)
        pre_list.append(pre_c)
        time_list.append(time_c)
        print(f'    {time_c[0]:.2f}–{time_c[-1]:.2f}')

    tmp = np.concatenate(tmp_list, axis=1)
    pre = np.concatenate(pre_list, axis=1)
    t   = np.concatenate(time_list)
    return tmp, pre, t

# =============================================================
# Helper: build SMR annual field on mesh
# =============================================================
def build_smr(smr_files, smr_year_range):
    yr0, yr1 = smr_year_range
    uniq = np.arange(yr0, yr1)
    ny = len(uniq)
    smr_mesh = np.zeros((num_vertices, ny))

    x_smr = y_smr = None
    year_to_index = {int(yr): i for i, yr in enumerate(uniq)}

    def decode_smr_time(time_var, epoch_yr):
        t_raw = np.asarray(time_var[:], dtype=float)
        units = getattr(time_var, 'units', '').lower()
        dt = float(t_raw[1] - t_raw[0]) if len(t_raw) > 1 else np.nan

        if 'day' in units or (np.isfinite(dt) and dt > 10.0):
            return epoch_yr + t_raw / 365.25

        if 'month' in units or (np.isfinite(dt) and dt <= 10.0):
            # Most newer SMR files use global month indices since 1850-01.
            t_global = 1850.0 + t_raw / 12.0
            if np.any((t_global >= yr0 - 1.0) & (t_global <= yr1 + 1.0)):
                return t_global
            return epoch_yr + t_raw / 12.0

        raise ValueError(f'Cannot decode SMR time axis: units={units!r}, dt={dt!r}')

    # Stream SMR files instead of concatenating all monthly grids in memory.
    for smr_file, epoch_yr in smr_files:
        print(f'    SMR file: {os.path.basename(smr_file)}')
        nc = Dataset(smr_file)
        if x_smr is None:
            x_smr = nc.variables['x'][:].astype(float)
            y_smr = nc.variables['y'][:].astype(float)

        t = decode_smr_time(nc.variables['time'], epoch_yr)
        yi_int = t.astype(int)
        smrv = nc.variables['submarine_melt']

        years_here = [yr for yr in np.unique(yi_int) if int(yr) in year_to_index]
        for count, yr in enumerate(years_here, start=1):
            if count == 1 or count % 25 == 0 or count == len(years_here):
                print(f'      year {int(yr)} ({count}/{len(years_here)})')
            mask = np.where(yi_int == yr)[0]
            if mask.size == 0:
                continue
            f = np.nanmean(smrv[mask, :, :], axis=0)
            f = np.where(np.isfinite(f), f, 0.0)
            smr_mesh[:, year_to_index[int(yr)]] = InterpFromGridToMesh(
                x_smr, y_smr, f, md_ref.mesh.x, md_ref.mesh.y, 0)
        nc.close()

    return np.clip(smr_mesh, 0.0, 500.0), uniq + 0.5

# =============================================================
# STEP 4: MAIN LOOP over 50-year segmented periods -- generate NC first
# =============================================================
for period in MAJOR_PERIODS:
    label      = period['label']
    start_time = period['start_time']
    final_time = period['final_time']
    save_as    = period['save_as']

    print('='*60)
    print(f'[{label}]  {start_time}-{final_time}')
    print('='*60)

    if output_complete(save_as, final_time):
        print(f'  Output exists, skip solve: {os.path.basename(save_as)}')
        continue

    if not os.path.exists(period['load_from']):
        raise FileNotFoundError(
            f'Missing restart/input file for {label}: {period["load_from"]}'
        )

    # -- Load model state -------------------------------------------------
    print(f'  Loading: {os.path.basename(period["load_from"])}')
    md = loadmodel(period['load_from'])
    [md.mesh.lat, md.mesh.long] = xy2ll(md.mesh.x, md.mesh.y, +1)
    md_ref = md   # alias for SMR build helper

    md.geometry.thickness      = md.results.TransientSolution[-1].Thickness.copy()
    md.geometry.base           = md.results.TransientSolution[-1].Base.copy()
    md.geometry.base           = np.maximum(md.geometry.base, md.geometry.bed)
    md.geometry.surface        = md.geometry.base + md.geometry.thickness
    md.initialization.vel      = md.results.TransientSolution[-1].Vel.copy()
    md.initialization.vx       = md.results.TransientSolution[-1].Vx.copy()
    md.initialization.vy       = md.results.TransientSolution[-1].Vy.copy()
    md.initialization.pressure = md.results.TransientSolution[-1].Pressure.copy()
    if hasattr(md.results.TransientSolution[-1], 'MaskIceLevelset'):
        md.mask.ice_levelset = md.results.TransientSolution[-1].MaskIceLevelset.copy()
        # md.mask.ice_levelset = reinitializelevelset(md, md.mask.ice_levelset)
    print(f'  Vel max: {md.initialization.vel.max():.1f} m/yr')

    # -- Build T/P arrays from GISS files --------------------------------
    print('  Building T/P (reading GISS files chunk by chunk)...')
    tmp, pre, time_decimal = build_tp_arrays(
        period['giss_files'], md.mesh.long, md.mesh.lat)
    nt = tmp.shape[1]
    print(f'  T/P: {nt} months, {time_decimal[0]:.2f}-{time_decimal[-1]:.2f}')

    md.smb = SMBd18opdd()
    md.smb.isd18opd = 1
    md.smb.delta18o = np.loadtxt(DELTA18O)
    md.smb.rlaps    = 6.0
    md.smb.desfac   = 1
    md.smb.rlapslgm = 5.5
    md.smb.issetpddfac  = 1
    md.smb.pddfac_snow  = 4.0 * np.ones(num_vertices)
    md.smb.pddfac_ice   = 8.0 * np.ones(num_vertices)
    md.smb.isprecipscaled      = 0
    md.smb.istemperaturescaled = 0
    md.smb.s0p = np.maximum(MAR_dem, 0)
    md.smb.s0t = np.maximum(MAR_dem, 0)
    md.smb.precipitations_presentday = precip_presentday.copy()
    md.smb.temperatures_presentday   = temp_presentday.copy()

    md.smb.temperatures_reconstructed   = np.zeros((num_vertices + 1, nt))
    md.smb.precipitations_reconstructed = np.zeros((num_vertices + 1, nt))
    md.smb.temperatures_reconstructed[:-1, :]   = tmp
    md.smb.precipitations_reconstructed[:-1, :] = pre
    md.smb.temperatures_reconstructed[-1, :]    = time_decimal
    md.smb.precipitations_reconstructed[-1, :]  = time_decimal
    del tmp, pre

    # -- SMR --------------------------------------------------------------
    print('  Building SMR...')
    smr_mesh, smr_time = build_smr(period['smr_files'], period['smr_years'])
    md.frontalforcings.meltingrate = np.zeros((num_vertices + 1, len(smr_time)))
    md.frontalforcings.meltingrate[:-1, :] = smr_mesh
    md.frontalforcings.meltingrate[-1, :]  = smr_time

    # -- Basal + transient + calving -------------------------------------
    md.basalforcings.floatingice_melting_rate = 10.0 * np.ones(num_vertices)
    md.basalforcings.groundedice_melting_rate = np.zeros(num_vertices)

    print(f'  CFL: {cfl_step(md, md.initialization.vx, md.initialization.vy):.3f} yr')
    md.timestepping.start_time = start_time
    md.timestepping.final_time = final_time
    md.timestepping.time_step  = 1.0 / 24.0
    md.transient.isthermal     = 0
    md.transient.ismovingfront = 1
    md.inversion.iscontrol     = 0
    md.transient.requested_outputs = [
        'default', 'Thickness', 'MaskIceLevelset', 'MaskOceanLevelset',
        'CalvingCalvingrate', 'CalvingMeltingrate', 'GroundinglineMassFlux',
        'SmbMassBalance', 'SmbMelt', 'SmbAccumulation', 'TotalSmb',
        'IceVolume', 'IceVolumeAboveFloatation',
        'GroundedArea', 'FloatingArea',
        'IceVolumeScaled', 'IceVolumeAboveFloatationScaled',
        'GroundedAreaScaled', 'FloatingAreaScaled', 'TotalSmbScaled',
    ]
    # time_step = 1/24 yr, so output_frequency=2 writes every 1/12 yr (monthly).
    # Halving the old 0.05 yr step is strictly more conservative for CFL, at the
    # cost of ~20% more solve steps per segment.
    md.settings.output_frequency = 2

    md.levelset.spclevelset = np.full(num_vertices, np.nan)
    pos = np.union1d(np.where(md.mesh.vertexonboundary > 0)[0],
                     np.where(md.mask.ice_levelset > 0)[0])
    md.levelset.spclevelset[pos] = md.mask.ice_levelset[pos]
    md.levelset.reinit_frequency = 5
    md.levelset.stabilization    = 1
    md.thermal.reltol            = 0.01
    md.thermal.penalty_lock      = 2
    md.thermal.penalty_threshold = 10
    md.calving = calvingvonmises()
    md.calving.stress_threshold_floatingice = 200e3
    md.calving.stress_threshold_groundedice = 1e6
    md.calving.min_thickness                = 2
    md.levelset.kill_icebergs               = 1

    md.miscellaneous.name = f'GrIS_{MEMBER}_{label}'
    md.cluster = generic('name', 'localhost', 'np', 16)
    md.verbose  = verbose('solution', True, 'module', False, 'convergence', False)
    md.settings.solver_residue_threshold = 1e-4
    md.stressbalance.maxiter = 100

    # -- Solve + save -----------------------------------------------------
    print(f'  Solving {start_time}-{final_time}...')
    md = solve(md, 'Transient')
    print('  Solve complete.')

    export_netCDF(md, save_as)
    print(f'  Saved: {os.path.basename(save_as)}')

    V = md.results.TransientSolution[-1].Vel
    print(f'  Vel: max={np.nanmax(V):.1f}  p99={np.nanpercentile(V,99):.1f} m/yr')
    print(f'  SMB: {md.results.TransientSolution[-1].TotalSmb[0]:.1f} Gt/yr')
    del md

print('='*60)
print(f'All requested NC file is present or solved! ({MEMBER})')

missing_outputs = [p['save_as'] for p in MAJOR_PERIODS if not output_complete(p['save_as'], p['final_time'])]
if missing_outputs:
    raise FileNotFoundError('Missing/incomplete NC output(s): ' + ', '.join(missing_outputs))

# =============================================================
# STEP 5: Post-process FW flux after all NC files exist
# =============================================================
def _nc_mesh(handle):
    x = handle['mesh/x'][:]
    y = handle['mesh/y'][:]
    elements = handle['mesh/elements'][:].astype(int)
    if elements.min() == 1:
        elements -= 1
    return x, y, elements


def _front_length_weights(elements, x, y, ice_flag):
    edges = np.vstack((elements[:, [0, 1]], elements[:, [1, 2]], elements[:, [2, 0]]))
    edges.sort(axis=1)
    crosses = ice_flag[edges[:, 0]] ^ ice_flag[edges[:, 1]]
    front_edges = np.unique(edges[crosses], axis=0)

    weights = np.zeros_like(x, dtype=np.float64)
    if front_edges.size == 0:
        return weights
    i = front_edges[:, 0]
    j = front_edges[:, 1]
    length = np.hypot(x[i] - x[j], y[i] - y[j])
    np.add.at(weights, i, length / 2.0)
    np.add.at(weights, j, length / 2.0)
    weights[~ice_flag] = 0.0
    return weights


def _read_total_smb(group, k):
    if 'TotalSmb' in group:
        return float(np.asarray(group['TotalSmb'][k]).squeeze())
    if 'TotalSmbScaled' in group:
        return float(np.asarray(group['TotalSmbScaled'][k]).squeeze())
    return np.nan


all_years = []
all_fw_frontal = []
all_fw_calving = []
all_fw_smb = []
seen_years = set()
Gt = 1e-12

for period in MAJOR_PERIODS:
    nc_path = period['save_as']
    print('='*60)
    print(f'Post-processing FW flux: {os.path.basename(nc_path)}')
    with h5py.File(nc_path, 'r') as nc:
        x, y, elements = _nc_mesh(nc)
        rho_ice = float(nc['materials/rho_ice'][()])
        sol = nc['results/TransientSolution']
        times = np.asarray(sol['time'][:], dtype=np.float64)

        for k, year in enumerate(times):
            year_key = round(float(year), 6)
            if year_key in seen_years:
                continue
            seen_years.add(year_key)

            ice_flag = np.asarray(sol['MaskIceLevelset'][k, :] < 0)
            H        = np.nan_to_num(np.asarray(sol['Thickness'][k, :], dtype=np.float64), copy=False)
            smr_v    = np.nan_to_num(np.asarray(sol['CalvingMeltingrate'][k, :], dtype=np.float64), copy=False)
            cr_v     = np.nan_to_num(np.asarray(sol['CalvingCalvingrate'][k, :], dtype=np.float64), copy=False)
            vw       = _front_length_weights(elements, x, y, ice_flag)

            all_years.append(float(year))
            all_fw_frontal.append(float(np.sum(smr_v * H * vw)) * rho_ice * Gt)
            all_fw_calving.append(float(np.sum(cr_v  * H * vw)) * rho_ice * Gt)
            all_fw_smb.append(_read_total_smb(sol, k))

            if len(all_years) == 1 or len(all_years) % 25 == 0:
                print(f'  n={len(all_years):4d}, year={float(year):.2f}')

fw_npz = f'{MODELS}/FW_flux_1850-2500_{MEMBER}_lenearts.npz'
np.savez(fw_npz,
         years      = np.array(all_years),
         fw_frontal = np.array(all_fw_frontal),
         fw_calving = np.array(all_fw_calving),
         fw_smb     = np.array(all_fw_smb))
print('='*60)
print(f'FW flux saved: {fw_npz}')
print(f'  n={len(all_years)}, years {all_years[0]:.1f}-{all_years[-1]:.1f}')
print('Done.')
