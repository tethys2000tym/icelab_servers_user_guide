# Parameterization file for Greenland 1850 setup (v2)
# Called by parameterize() in Setup_BedMachine_v2.py
#
# Fixes vs v1:
#   - base bug fixed: grounded ice base = bed (not floatation_base)
#   - min_thickness: 10 → 20 m
#   - friction lower bound: 5 → 20
#   - removed dead assignment (friction.coefficient vs md.friction.coefficient)
#   - removed broken reinitializelevelset call
#
# Surface: modern BedMachine (friction inversion uses modern MEaSUREs obs)
# 1850 surface is applied in Setup_BedMachine_v2.py Step 1.6

import os
import sys
sys.path.append("/home/yanmeiti/issm/Functions/")
sys.path.append(os.getenv('ISSM_DIR') + '/bin')
sys.path.append(os.getenv('ISSM_DIR') + '/lib')

import numpy as np
from netCDF4 import Dataset
from model import *
from InterpFromGridToMesh import InterpFromGridToMesh
from InterpFromMeshToMesh2d import InterpFromMeshToMesh2d
from loadmodel import loadmodel
from sethydrostaticmask import setmask
from SetMarineIceSheetBC import SetMarineIceSheetBC
from slope import slope
from averaging import averaging
from paterson import paterson

DATA = '/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Data'

md.mesh.epsg = 3413
md.miscellaneous.name = 'GRIS_BedMachine_S1_v2'

# =============================================================
# Bed and surface from BedMachine v5 (modern)
# =============================================================
print('   Loading BedMachine')
nc = Dataset(f'{DATA}/BedMachineGreenland-v5.nc', 'r')
topg    = np.squeeze(nc.variables['bed'][:].data)
surface = np.squeeze(nc.variables['surface'][:].data)
x3      = np.squeeze(nc.variables['x'][:].data)
y3      = np.squeeze(nc.variables['y'][:].data)
nc.close()

md.geometry.bed     = InterpFromGridToMesh(
    x3.astype(float), np.flipud(y3.astype(float)),
    np.flipud(topg),    md.mesh.x, md.mesh.y, 0)
md.geometry.surface = InterpFromGridToMesh(
    x3.astype(float), np.flipud(y3.astype(float)),
    np.flipud(surface), md.mesh.x, md.mesh.y, 0)
del topg, surface

# =============================================================
# MEaSUREs velocity observations
# =============================================================
print('   Loading MEaSUREs velocity')
nc = Dataset(f'{DATA}/icevel_gris_measures.nc', 'r')
x1   = nc.variables['x'][:].data
y1   = nc.variables['y'][:].data
velx = nc.variables['velx'][:].data
vely = nc.variables['vely'][:].data
vel  = nc.variables['vel'][:].data
nc.close()

md.inversion.vx_obs  = InterpFromGridToMesh(x1, y1, np.flipud(velx.T), md.mesh.x, md.mesh.y, 0)
md.inversion.vy_obs  = InterpFromGridToMesh(x1, y1, np.flipud(vely.T), md.mesh.x, md.mesh.y, 0)
md.inversion.vel_obs = InterpFromGridToMesh(x1, y1, np.flipud(vel.T),  md.mesh.x, md.mesh.y, 0)
md.initialization.vx  = md.inversion.vx_obs.copy()
md.initialization.vy  = md.inversion.vy_obs.copy()
md.initialization.vz  = np.zeros(md.mesh.numberofvertices)
md.initialization.vel = md.inversion.vel_obs.copy()
del velx, vely, vel

# =============================================================
# Geometry: base, thickness, ocean mask
# =============================================================
print('   Setting up geometry')
ice_mask   = md.mask.ice_levelset < 0
noice_mask = ~ice_mask
nvert      = md.mesh.numberofvertices
rho_i      = md.materials.rho_ice
rho_w      = md.materials.rho_water

md.geometry.base       = np.zeros(nvert)
md.mask.ocean_levelset = np.zeros(nvert)

# non-ice: base = bed, thickness = 0
md.geometry.base[noice_mask]      = md.geometry.bed[noice_mask]
md.geometry.thickness             = md.geometry.surface - md.geometry.bed
md.geometry.thickness[noice_mask] = 0.0
md.geometry.surface[noice_mask]   = md.geometry.bed[noice_mask]

# FIX: grounded → base = bed;  floating → base = floatation_base
floatation_base = md.geometry.surface * rho_i / (rho_i - rho_w)
grounded_mask   = ice_mask & (floatation_base <= md.geometry.bed)
floating_mask   = ice_mask & (floatation_base >  md.geometry.bed)

md.geometry.base[grounded_mask] = md.geometry.bed[grounded_mask]
md.geometry.base[floating_mask] = floatation_base[floating_mask]

md.geometry.thickness = md.geometry.surface - md.geometry.base

# FIX: minimum thickness 20 m
min_thickness = 20
thin = ice_mask & (md.geometry.thickness < min_thickness)
md.geometry.thickness[thin] = min_thickness
md.geometry.surface = md.geometry.base + md.geometry.thickness

# ocean mask
md = setmask(md)

# fix residual grounded base inconsistencies after setmask
pos_fix = np.where((md.mask.ocean_levelset > 0) & (md.geometry.base != md.geometry.bed))[0]
if len(pos_fix) > 0:
    print(f'   Correcting {len(pos_fix)} grounded base inconsistencies')
    md.geometry.base[pos_fix]      = md.geometry.bed[pos_fix]
    md.geometry.thickness[pos_fix] = md.geometry.surface[pos_fix] - md.geometry.base[pos_fix]
    thin2 = pos_fix[md.geometry.thickness[pos_fix] < min_thickness]
    md.geometry.thickness[thin2] = min_thickness
    md.geometry.surface[pos_fix] = md.geometry.base[pos_fix] + md.geometry.thickness[pos_fix]

# hydrostatic base for floating ice
pos_float = np.where(md.mask.ocean_levelset < 0)[0]
md.geometry.base[pos_float]    = -rho_i / rho_w * md.geometry.thickness[pos_float]
md.geometry.surface[pos_float] = md.geometry.base[pos_float] + md.geometry.thickness[pos_float]

print(f'   Grounded: {grounded_mask.sum()}  Floating: {floating_mask.sum()}')
print(f'   Thickness: min={md.geometry.thickness[ice_mask].min():.1f}  max={md.geometry.thickness.max():.1f} m')

# =============================================================
# Friction coefficient (SIA-based initial guess)
# =============================================================
print('   Computing friction coefficient initial guess')
sx, sy, s = slope(md)
sslope    = np.squeeze(averaging(md, s, 0))

obs_vel              = md.inversion.vel_obs.copy()
obs_vel[obs_vel <= 0] = 1.0
obs_vel              = np.maximum(obs_vel, 0.1)

denom = (rho_i * md.geometry.thickness + rho_w * md.geometry.bed) * obs_vel / md.constants.yts
denom[denom == 0] = 1e-10
term  = rho_i * md.geometry.thickness * sslope / denom
term[term < 0] = 0

md.friction.coefficient = np.sqrt(term)

# FIX: lower bound 20, upper bound 250
md.friction.coefficient[md.friction.coefficient < 20]  = 20
md.friction.coefficient[md.friction.coefficient > 250] = 250
md.friction.coefficient[pos_float]                     = 0.0

print(f'   Friction: min={md.friction.coefficient[ice_mask].min():.1f}  max={md.friction.coefficient.max():.1f}')

# =============================================================
# Temperature, rheology, pressure
# =============================================================
print('   Loading IGS model for temperature')
md2 = loadmodel(f'{DATA}/igs3_relax_setup1_relax1_part3_results.nc')

md.initialization.temperature = InterpFromMeshToMesh2d(
    md2.mesh.elements, md2.mesh.x, md2.mesh.y,
    md2.initialization.temperature, md.mesh.x, md.mesh.y)
md.initialization.pressure = md.materials.rho_ice * md.constants.g * md.geometry.thickness

md.materials.rheology_n = 3 * np.ones(md.mesh.numberofelements)
md.materials.rheology_B = paterson(md.initialization.temperature)
md.friction.q           = np.ones(md.mesh.numberofelements)
md.friction.p           = np.ones(md.mesh.numberofelements)

# =============================================================
# Geothermal heat flux
# =============================================================
print('   Loading geothermal heat flux')
nc    = Dataset(f'{DATA}/geothermal_heat_flow_map_10km.nc', 'r')
XGEOT = nc.variables['X'][:].data
YGEOT = nc.variables['Y'][:].data
GHF   = nc.variables['GHF'][:].data / 1000
nc.close()
md.basalforcings.geothermalflux = InterpFromGridToMesh(
    XGEOT.astype(float), YGEOT.astype(float), GHF, md.mesh.x, md.mesh.y, 0)

# =============================================================
# Boundary conditions
# =============================================================
print('   Setting boundary conditions')
md.stressbalance.referential = np.nan * np.ones((nvert, 6))
md.stressbalance.spcvx       = np.nan * np.ones(nvert)
md.stressbalance.spcvy       = np.nan * np.ones(nvert)
md.stressbalance.spcvz       = np.nan * np.ones(nvert)

md.geometry.hydrostatic_ratio             = np.ones(nvert)
md                                        = SetMarineIceSheetBC(md)
md.basalforcings.floatingice_melting_rate = np.zeros(nvert)
md.basalforcings.groundedice_melting_rate = np.zeros(nvert)
md.thermal.spctemperature                 = md.initialization.temperature.copy()
md.masstransport.spcthickness             = np.nan * np.ones(nvert)
