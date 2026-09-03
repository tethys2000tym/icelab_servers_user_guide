# Setup script for Greenland 1850 initialization (v2)
# Steps 1.2 → 1.5
#
# 1.2  Parameterize (modern BedMachine surface, fixed geometry + friction)
# 1.3  Initial stress balance
# 1.4  Basal friction inversion (min_parameters = 20)
# 1.5  Post-inversion stress balance
#
# Output: Step1.5_StressBalance_PostInversion_v2.nc

import os, sys

issm_dir = os.getenv('ISSM_DIR')
sys.path.insert(0, '/home/yanmeiti/issm/Functions')
sys.path.insert(0, issm_dir + '/bin')
sys.path.insert(0, issm_dir + '/lib')
sys.path.insert(0, issm_dir + '/share')
sys.path.insert(0, issm_dir + '/share/proj')

import numpy as np
from netCDF4 import Dataset
from model import *
from loadmodel import loadmodel
from export_netCDF import export_netCDF
from parameterize import parameterize
from setflowequation import setflowequation
from solve import solve
from generic import generic
from ContourToMesh import ContourToMesh
from InterpFromGridToMesh import InterpFromGridToMesh
from verbose import verbose

SCRIPTS = '/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/scripts_tym_v2'
MODELS  = '/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Models_tym_v2'
DATA    = '/home/yanmeiti/issm/Greenland/ISSM_GrIS_Yanmei/Data'

# =============================================================
# Step 1.2 – Parameterization (modern surface)
# =============================================================
print('\n===== Step 1.2: Parameterization =====')
md = loadmodel(f'{MODELS}/Step1.1_1850mesh.nc')

md.mask.ice_levelset = np.ones(md.mesh.numberofvertices)
inn = ContourToMesh(md.mesh.elements, md.mesh.x, md.mesh.y,
                    f'{DATA}/innerDomainLIA.exp', 'node', 1)
md.mask.ice_levelset[np.where(inn == 1)] = -1

md = parameterize(md, f'{SCRIPTS}/S0_par_v2.py')
md = setflowequation(md, 'SSA', 'all')

print(f'base<bed:    {np.sum(md.geometry.base < md.geometry.bed)}')
print(f'bed>surface: {np.sum(md.geometry.bed > md.geometry.surface)}')
print(f'friction NaN: {np.nansum(np.isnan(md.friction.coefficient))}')
print(f'Thickness min={md.geometry.thickness.min():.1f}  max={md.geometry.thickness.max():.1f} m')

export_netCDF(md, f'{MODELS}/Step1.2_parameters_v2.nc')
print('Step 1.2 saved.')

# =============================================================
# Step 1.3 – Initial Stress Balance
# =============================================================
print('\n===== Step 1.3: Initial Stress Balance =====')
md = loadmodel(f'{MODELS}/Step1.2_parameters_v2.nc')

# cap friction at margin only; reduce at fast-flow areas
margin = (md.mask.ice_levelset > -200) & (md.mask.ice_levelset < 200)
md.friction.coefficient[margin & (md.friction.coefficient > 80)] = 80
md.friction.coefficient[md.inversion.vel_obs > 40  & (md.friction.coefficient > 20)] *= 0.95
md.friction.coefficient[md.inversion.vel_obs > 100 & (md.friction.coefficient > 30)] *= 0.70
md.friction.coefficient[md.friction.coefficient < 20] = 20
md.friction.coefficient[md.mask.ocean_levelset < 0]   = 0.0

md.inversion.iscontrol = 0
md.stressbalance.restol = 0.01
md.stressbalance.reltol = 0.1
md.stressbalance.abstol = np.nan
md.stressbalance.loadingforce = np.zeros((md.mesh.numberofvertices, 3))

md.miscellaneous.name = 'GRIS_Step1.3_v2'
md.cluster = generic('name', 'localhost', 'np', 8)
md.verbose  = verbose('solution', True, 'module', False, 'convergence', False)
md = solve(md, 'Stressbalance')

V = md.results.StressbalanceSolution.Vel
print(f'SB Vel: max={np.nanmax(V):.1f}  p99={np.nanpercentile(V,99):.1f}  p99.9={np.nanpercentile(V,99.9):.1f} m/yr')

export_netCDF(md, f'{MODELS}/Step1.3_stressbalance_v2.nc')
print('Step 1.3 saved.')

# =============================================================
# Step 1.4 – Basal Friction Inversion
# =============================================================
print('\n===== Step 1.4: Basal Inversion =====')
md = loadmodel(f'{MODELS}/Step1.3_stressbalance_v2.nc')

nc = Dataset(f'{DATA}/BedMachineGreenland-v5.nc', 'r')
mask_bm = np.squeeze(nc.variables['mask'][:].data)
x3      = np.squeeze(nc.variables['x'][:].data)
y3      = np.squeeze(nc.variables['y'][:].data)
nc.close()
mask_bm = InterpFromGridToMesh(
    x3.astype(float), np.flipud(y3.astype(float)),
    np.flipud(mask_bm), md.mesh.x, md.mesh.y, 0)

md.inversion.iscontrol           = 1
md.inversion.nsteps              = 200
md.inversion.step_threshold      = 0.99 * np.ones(md.inversion.nsteps)
md.inversion.maxiter_per_step    = 5    * np.ones(md.inversion.nsteps)

md.inversion.cost_functions = [101, 103, 501]
md.inversion.cost_functions_coefficients = np.ones((md.mesh.numberofvertices, 3))
md.inversion.cost_functions_coefficients[:, 0] = 350
md.inversion.cost_functions_coefficients[:, 1] = 0.2
md.inversion.cost_functions_coefficients[:, 2] = 2e-6

md.inversion.control_parameters = ['FrictionCoefficient']
md.inversion.gradient_scaling    = 50 * np.ones((md.inversion.nsteps, 1))

# FIX: min_parameters = 20 (was 5)
md.inversion.min_parameters = 20  * np.ones((md.mesh.numberofvertices, 1))
md.inversion.max_parameters = 250 * np.ones((md.mesh.numberofvertices, 1))

# exclude no-data and non-glacier areas from cost function
pos_nofit = np.where(
    (mask_bm >= 2.5) | (mask_bm <= 1.5) | (md.inversion.vel_obs <= 1)
)[0]
md.inversion.cost_functions_coefficients[pos_nofit, :] = 0
pos_float = np.where(md.mask.ocean_levelset < 0)[0]
md.inversion.cost_functions_coefficients[pos_float, :] = 0

md.stressbalance.restol = 0.01
md.stressbalance.reltol = 0.1
md.stressbalance.abstol = np.nan

md.miscellaneous.name = 'GRIS_Step1.4_v2'
md.cluster = generic('name', 'localhost', 'np', 8)
md.verbose  = verbose('solution', True, 'control', True, 'module', False)
md = solve(md, 'Stressbalance')

md.initialization.vx    = md.results.StressbalanceSolution.Vx.copy()
md.initialization.vy    = md.results.StressbalanceSolution.Vy.copy()
md.friction.coefficient = md.results.StressbalanceSolution.FrictionCoefficient.copy()

print(f'Post-inversion friction: min={md.friction.coefficient.min():.1f}  max={md.friction.coefficient.max():.1f}')

export_netCDF(md, f'{MODELS}/Step1.4_BasalInversion_v2.nc')
print('Step 1.4 saved.')

# =============================================================
# Step 1.5 – Post-Inversion Stress Balance (modern surface)
# =============================================================
print('\n===== Step 1.5: Post-Inversion Stress Balance =====')
md = loadmodel(f'{MODELS}/Step1.4_BasalInversion_v2.nc')

md.inversion.iscontrol = 0
md.stressbalance.restol = 0.01
md.stressbalance.reltol = 0.1
md.stressbalance.abstol = np.nan
md.stressbalance.loadingforce = np.zeros((md.mesh.numberofvertices, 3))

md.miscellaneous.name = 'GRIS_Step1.5_v2'
md.cluster = generic('name', 'localhost', 'np', 8)
md.verbose  = verbose('solution', True, 'module', False, 'convergence', False)
md = solve(md, 'Stressbalance')

V = md.results.StressbalanceSolution.Vel
print(f'Post-inversion Vel: max={np.nanmax(V):.1f}  p99={np.nanpercentile(V,99):.1f}  p99.9={np.nanpercentile(V,99.9):.1f} m/yr')

export_netCDF(md, f'{MODELS}/Step1.5_StressBalance_PostInversion_v2.nc')
print('Step 1.5 saved.')
print('\nAll steps completed. Next: run Setup_BedMachine_v3.py for 1850 surface + SB')
