#!/usr/bin/env python
# coding: utf-8
# ISMIP7_ssp.py
#
# Full pipeline for one ISMIP7 SSP experiment

import os
import sys
import glob
from pathlib import Path

import numpy as np
import xarray as xr

# ISSM 4.24 still calls np.in1d, which was removed in NumPy 2.4. 
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# ---------------------------------------------------------------------------
# CONFIG — scenario / model / paths
# ---------------------------------------------------------------------------
MODEL    = "CESM2-WACCM"
SCENARIO = "SSP126"          
EXPERIMENT = "C005"          # ISMIP7 experiment id for this scenario

BASE_DIR   = "/home/yanmeiti/ISMIP7"
DATA_DIR   = os.path.join(BASE_DIR, "Data", MODEL, SCENARIO)
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")

# Historical ISSM results to start from
HISTORICAL_MODEL = os.path.join(
    BASE_DIR, "Data", MODEL, "issmgris_update2_Setup2_DH02_historical_results.nc"
)

# ocean variables
TF_DIR  = os.path.join(DATA_DIR, "tf") 
SGD_DIR = os.path.join(DATA_DIR, "sgd")

# submarine-melt inputs to calculate the submarine melt rate
SUBMARINE_STATIC_DIR = "/home/yanmeiti/Submarine_Melting/03_submarine_melt"
A_FILE    = os.path.join(SUBMARINE_STATIC_DIR, "basin_submerged_area_ismip.nc")
BED_FILE  = os.path.join(SUBMARINE_STATIC_DIR, "topg_ismip.nc")
MASK_FILE = os.path.join(SUBMARINE_STATIC_DIR, "subglacial_discharge_basins_ismip.nc")


SKIP_EXISTING_SMR = True

MELT_OUT_DIR = os.path.join(OUTPUT_DIR, "melt", SCENARIO)

# Prepare-stage output (forcing + initial conditions, no solve yet)
PREPARE_NC = os.path.join(OUTPUT_DIR, f"icelab_ismip7_{EXPERIMENT}.Transient_Prepare.nc")

# Per-segment transient outputs go here
RESULTS_DIR = os.path.join(OUTPUT_DIR, "Results", EXPERIMENT)

os.makedirs(MELT_OUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# ISSM import path, this is important
# ---------------------------------------------------------------------------
sys.path.append(os.getenv("ISSM_DIR") + "/bin")
sys.path.append(os.getenv("ISSM_DIR") + "/lib")
sys.path.append(os.getenv("ISSM_DIR") + "/share")
sys.path.append(os.getenv("ISSM_DIR") + "/share/proj")

my_function_path = "/home/yanmeiti/ISMIP7/Functions"
if my_function_path not in sys.path:
    sys.path.insert(0, my_function_path)

from triangle import triangle
from model import *
from netCDF4 import Dataset
from InterpFromGridToMesh import InterpFromGridToMesh
from bamg import bamg
from xy2ll import xy2ll
from plotmodel import plotmodel
from export_netCDF import export_netCDF
from loadmodel import loadmodel
from setmask import setmask
from parameterize import parameterize
from setflowequation import setflowequation
from socket import gethostname
from ll2xy import ll2xy
from BamgTriangulate import BamgTriangulate
from InterpFromMeshToMesh2d import InterpFromMeshToMesh2d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from ContourToMesh import ContourToMesh
import rasterio
from paterson import paterson
from solve import solve
from cfl_step import cfl_step
from m1qn3inversion import m1qn3inversion
from reinitializelevelset import reinitializelevelset
from SMBgradients import SMBgradients
from generic import generic


CLUSTER_NP = 24


def output_complete(nc_path, final_time, tol=1.0e-6):
    """True if nc_path exists and its TransientSolution reaches final_time."""
    if not os.path.exists(nc_path) or os.path.getsize(nc_path) <= 0:
        return False
    try:
        with Dataset(nc_path, "r") as ds:
            time = ds.groups["results"].groups["TransientSolution"].variables["time"]
            if len(time) == 0:
                return False
            return float(time[-1]) >= final_time - tol
    except Exception as exc:
        print(f"  Existing output is not readable/complete: {os.path.basename(nc_path)} ({exc})")
        return False


# ===========================================================================
# STAGE 0: Compute submarine melt rate 
# — Xu & Rignot (2012) parameterization from TF/SGD forcing. Doesn't touch
# ===========================================================================
print("=" * 60)
print("STAGE 0: Compute submarine melt rate")
print("=" * 60)

with xr.open_dataset(A_FILE) as ds:
    _A = ds["submergedarea"].load()
with xr.open_dataset(BED_FILE) as ds:
    _bed = ds["bed"].load()
_h = xr.where(_bed < 0, -_bed, 0)

with xr.open_dataset(MASK_FILE) as ds:
    _mask = ((ds["basin"] > 0) & (ds["convexhullmask"] == 0)).load()

# Xu & Rignot (2012) parameterization coefficients
_a_xr, _b_xr, _alpha_xr, _beta_xr = 3e-4, 0.15, 0.39, 1.18

_tf_files = sorted(glob.glob(os.path.join(TF_DIR, "tf_*.nc")))
_q_files = sorted(glob.glob(os.path.join(SGD_DIR, "sgd_*.nc")))
if not _tf_files or not _q_files:
    raise FileNotFoundError(
        f"No TF/Q files found (TF_DIR={TF_DIR} -> {len(_tf_files)} files, "
        f"SGD_DIR={SGD_DIR} -> {len(_q_files)} files)"
    )

print(f"TF files: {len(_tf_files)}, Q files: {len(_q_files)}")

if len(_q_files) > len(_tf_files):
    # Repeat the last TF file to cover extra Q periods
    _tf_files = _tf_files + [_tf_files[-1]] * (len(_q_files) - len(_tf_files))
elif len(_tf_files) != len(_q_files):
    print(
        f"WARNING: more TF than Q files ({len(_tf_files)} vs {len(_q_files)}). "
        "Processing only paired files."
    )

def _smr_output_valid(path):
    #True if path exists AND actually has the submarine_melt variable.
    
    if not os.path.exists(path):
        return False
    try:
        with Dataset(path, "r") as ds:
            return "submarine_melt" in ds.variables
    except Exception:
        return False


_n_done = 0
_n_skipped = 0

for _tf_file, _q_file in zip(_tf_files, _q_files):
    _out_file = os.path.basename(_tf_file).replace("tf_", "SMR_")
    _out_path = os.path.join(MELT_OUT_DIR, _out_file)

    if SKIP_EXISTING_SMR and _smr_output_valid(_out_path):
        _n_skipped += 1
        continue

    print(f"\n{'=' * 60}")
    print(f"TF : {os.path.basename(_tf_file)}")
    print(f"Q  : {os.path.basename(_q_file)}")

    _ds_tf = xr.open_dataset(_tf_file, decode_times=False, chunks={"time": 12})
    _ds_q = xr.open_dataset(_q_file, decode_times=False, chunks={"time": 12})

    _TF = _ds_tf["tf"]
    _Q = _ds_q["sgd"]

    _n_tf = _TF.sizes["time"]
    _n_q = _Q.sizes["time"]

    if _n_tf != _n_q:
        _n = min(_n_tf, _n_q)
        print(f"  WARNING: time length mismatch (TF={_n_tf}, Q={_n_q}), trimming both to {_n}")
        _TF = _TF.isel(time=slice(0, _n))
        _Q = _Q.isel(time=slice(0, _n))

    # TF file has no coordinates; borrow spatial + time coords from Q
    _TF = _TF.assign_coords(y=_Q.y, x=_Q.x, time=_Q.time)

    # q: subglacial discharge flux per unit submerged area [m/day]
    _q = 86400.0 * _Q / _A
    _q = xr.where(_mask, _q, 0).fillna(0).clip(min=0)

    # Only positive TF drives melting
    _TF_pos = xr.where(_TF > 0, _TF, 0)

    # Xu & Rignot parameterization [m/day]
    _m = (_a_xr * _h * (_q ** _alpha_xr) + _b_xr) * (_TF_pos ** _beta_xr)

    # Zero out outside mask and where bed is above sea level
    _m = xr.where(_mask & (_h > 0), _m, 0)

    # Remove any remaining non-finite or negative values
    _m = xr.where(np.isfinite(_m) & (_m > 0), _m, 0)

    # Convert to m/yr
    _m = (_m * 365.0).transpose("time", "y", "x").astype("float32")

    _ds_out = xr.Dataset({"submarine_melt": _m})
    _ds_out["submarine_melt"].attrs = {
        "units": "m/yr",
        "standard_name": "submarine_melt_rate",
        "long_name": "Submarine melt rate (Xu & Rignot 2012 parameterization)",
        "forcing_TF": os.path.basename(_tf_file),
        "forcing_Q": os.path.basename(_q_file),
    }

    print(f"  Saving -> {_out_path}")
    _ds_out.to_netcdf(_out_path)

    _ds_tf.close()
    _ds_q.close()
    _n_done += 1
    print("  Done.")

print(
    f"\nSMR stage complete. ({_n_done} computed, {_n_skipped} already existed and were skipped)"
)

# ===========================================================================
# STAGE 1: Prepare — build forcing + initial conditions 
# ===========================================================================
# PREPARE_NC was never solved, so it has no TransientSolution
if os.path.exists(PREPARE_NC):
    print(f"Prepare-stage output already exists, skipping prepare: {PREPARE_NC}")
else:
    print("=" * 60)
    print("STAGE 1: Prepare (forcing + initial conditions)")
    print("=" * 60)

    # --- 1a. Load historical model ----------------------------------------
    print(f"Loading historical model: {HISTORICAL_MODEL}")
    md = loadmodel(HISTORICAL_MODEL)

    md.mask.ice_levelset = md.results.TransientSolution[-1].MaskIceLevelset.copy()
    md.mask.ice_levelset = reinitializelevelset(md, md.mask.ice_levelset)

    md.geometry.thickness = md.results.TransientSolution[-1].Thickness.copy()
    md.geometry.base = md.results.TransientSolution[-1].Base.copy()
    md.geometry.base = np.maximum(md.geometry.base, md.geometry.bed)
    md.geometry.surface = md.geometry.base + md.geometry.thickness
    md.initialization.vel = md.results.TransientSolution[-1].Vel.copy()
    md.initialization.vx = md.results.TransientSolution[-1].Vx.copy()
    md.initialization.vy = md.results.TransientSolution[-1].Vy.copy()
    md.initialization.pressure = md.results.TransientSolution[-1].Pressure.copy()
    md.mask.ocean_levelset = md.results.TransientSolution[-1].MaskOceanLevelset.copy()

    # --- 1b. SMB forcing (acabf) -------------------------------------------
    print("Processing SMB forcing (acabf)...")
    acabf_dir = Path(DATA_DIR) / "acabf"
    files = sorted(acabf_dir.glob("acabf*.nc"), key=lambda f: int(f.stem.split("_")[-1]))

    smb_2015_2300 = np.zeros((md.mesh.numberofvertices, 1))

    for f in files:
        with Dataset(f, mode="r") as ncdata:
            smb_cesm2 = np.squeeze(ncdata.variables["acabf"][:])
            x_cesm2 = np.squeeze(ncdata.variables["x"][:])
            y_cesm2 = np.squeeze(ncdata.variables["y"][:])

        smb_yeari = np.nanmean(smb_cesm2, axis=0)
        smb_yeari_mesh = InterpFromGridToMesh(
            x_cesm2, y_cesm2, smb_yeari, md.mesh.x, md.mesh.y, 0
        )
        smb_yeari_mesh = smb_yeari_mesh[:, np.newaxis]
        smb_2015_2300 = np.concatenate((smb_2015_2300, smb_yeari_mesh), axis=1)

    smb_2015_2300 = smb_2015_2300[:, 1:] * 31556926 * (1 / 1000) * (
        md.materials.rho_freshwater / md.materials.rho_ice
    )

    # --- 1c. SMB elevation correction (dacabfdz) ----------------------------
    print("Processing SMB elevation correction (dacabfdz)...")
    dacabfdz_dir = Path(DATA_DIR) / "dacabfdz"
    files = sorted(
        [f for f in dacabfdz_dir.glob("dacabfdz*.nc") if not f.name.startswith("._")],
        key=lambda f: int(f.stem.split("_")[-1]),
    )

    smbcorr_2015_2300 = np.zeros((md.geometry.thickness.shape[0], 1))

    for f in files:
        with Dataset(f, mode="r") as ncdata:
            x_cesm2 = np.squeeze(ncdata.variables["x"][:])
            y_cesm2 = np.squeeze(ncdata.variables["y"][:])
            smb_yeari = np.squeeze(ncdata.variables["dacabfdz"][:])

        smb_yeari_mesh = InterpFromGridToMesh(
            x_cesm2, y_cesm2, smb_yeari, md.mesh.x, md.mesh.y, 0
        )
        smb_yeari_mesh = smb_yeari_mesh[:, np.newaxis]
        smbcorr_2015_2300 = np.concatenate((smbcorr_2015_2300, smb_yeari_mesh), axis=1)

    smbcorr_2015_2300 = smbcorr_2015_2300[:, 1:] * 31556926 * (1 / 1000) * (
        md.materials.rho_freshwater / md.materials.rho_ice
    )

    # --- 1d. Set up SMB on the model ----------------------------------------
    last_row = np.arange(316, 316 + smb_2015_2300.shape[1])
    smb_correction_in = np.concatenate((smbcorr_2015_2300, last_row[:, np.newaxis].T), axis=0)
    smb_in = np.concatenate((smb_2015_2300, last_row[:, np.newaxis].T), axis=0)

    surface_reference = md.geometry.surface.copy()

    md.smb = SMBgradients()
    md.smb.href = surface_reference
    md.smb.smbref = smb_in.copy()
    md.smb.b_pos = smb_correction_in.copy()
    md.smb.b_neg = smb_correction_in.copy()

    # --- 1e. Submarine melt (read SMR_*.nc from Compute_SMR_ISMIP7.py) ------
    print("Applying submarine melt to model...")
    melt_files = sorted(glob.glob(os.path.join(MELT_OUT_DIR, "*.nc")))
    print("Number of melt files:", len(melt_files))
    if not melt_files:
        raise FileNotFoundError(
            f"No melt-rate files found in {MELT_OUT_DIR} — run Compute_SMR_ISMIP7.py first."
        )

    with Dataset(melt_files[0]) as nc:
        x_smr = nc.variables["x"][:]
        y_smr = nc.variables["y"][:]

    smr_all = []
    years = []
    for f in melt_files:
        year = int(os.path.basename(f).split("_")[-1].replace(".nc", ""))
        years.append(year)
        with Dataset(f) as nc:
            smr = np.nanmean(nc.variables["submarine_melt"][:], axis=0)
        smr_all.append(smr)

    smr_all = np.array(smr_all)
    smr_raw = smr_all.copy()
    years = np.array(years)

    if not np.array_equal(years, np.arange(years[0], years[0] + len(years))):
        print(
            "WARNING: melt files are not a contiguous run of consecutive years "
            f"({years[0]}-{years[-1]}, {len(years)} files) — check melt_files coverage/ordering."
        )

    uniq_years = years
    nt_ann = len(uniq_years)
    time_smr_ann = uniq_years + 0.5

    nv = md.mesh.numberofvertices
    smr_mesh_ann = np.zeros((nv, nt_ann))

    print(f"Interpolating SMR ({nt_ann} annual steps) to mesh...")
    for yi, yr in enumerate(uniq_years):
        if yi % 20 == 0:
            print(f" year {yr}")
        field_ann = smr_raw[yi]
        field_ann = np.where(np.isfinite(field_ann), field_ann, 0.0)
        field_ann = np.maximum(field_ann, 0.0)
        smr_mesh_ann[:, yi] = InterpFromGridToMesh(
            x_smr.astype(float), y_smr.astype(float), field_ann, md.mesh.x, md.mesh.y, 0
        )

    smr_mesh_ann = np.clip(smr_mesh_ann, 0.0, 500.0)

    if nt_ann != smb_2015_2300.shape[1]:
        raise ValueError(
            f"SMR file count ({nt_ann}) != SMB file count ({smb_2015_2300.shape[1]}) — "
            f"did Compute_SMR_ISMIP7.py finish? Check {MELT_OUT_DIR}."
        )

    time_future = np.linspace(316, 316 + smb_2015_2300.shape[1] - 1, smb_2015_2300.shape[1])
    smr_future = np.vstack((smr_mesh_ann, time_future))
    smr_hist = md.frontalforcings.meltingrate.copy()
    md.frontalforcings.meltingrate = np.concatenate((smr_hist, smr_future), axis=1)

    print(
        f"SMR range: {md.frontalforcings.meltingrate.min():.2f} - "
        f"{md.frontalforcings.meltingrate.max():.2f} m/yr"
    )

    # --- 1f. Time stepping + requested outputs + save -----------------------
    # final_time derived the same way as last_row above (was hardcoded 601,
    # i.e. 316 + 285 — only correct for SSP126's 286-year span).
    md.timestepping.time_step = 0.03
    md.timestepping.start_time = 316
    md.timestepping.final_time = 316 + smb_2015_2300.shape[1] - 1

    md.miscellaneous.name = f"ISMIP7_ISSM_UB_{EXPERIMENT}"

    md.transient.requested_outputs = [
        "default", "Thickness", "MaskIceLevelset", "MaskOceanLevelset",
        "CalvingCalvingrate", "CalvingMeltingrate", "SmbMassBalance", "TotalSmb",
        "IceVolume", "IceVolumeAboveFloatation", "GroundedArea", "FloatingArea",
        "IceVolumeScaled", "IceVolumeAboveFloatationScaled",
        "GroundedAreaScaled", "FloatingAreaScaled", "TotalSmbScaled",
        "BasalforcingsGroundediceMeltingRate", "TotalFloatingBmb", "TotalGroundedBmb",
        "TotalCalvingFluxLevelset", "TotalCalvingMeltingFluxLevelset",
        "GroundinglineMassFlux",
    ]

    print(f"Saving prepared model -> {PREPARE_NC}")
    export_netCDF(md, PREPARE_NC)
    print("Prepare stage done.")
    del md

# ===========================================================================
# STAGE 2: Solve — one solve() call for the full 2015-2300 range, one output file.
# calving / levelset / thermal are not here, they are already set on the historical model
# ===========================================================================
print("=" * 60)
print(f"STAGE 2: Solve ({EXPERIMENT}, {SCENARIO})")
print("=" * 60)


RESULTS_NC = os.path.join(
    RESULTS_DIR, os.path.basename(PREPARE_NC).replace("_Prepare.nc", "_results.nc")
)

# Read directly off PREPARE_NC
with Dataset(PREPARE_NC, "r") as _ds:
    FULL_START = float(_ds.groups["timestepping"].variables["start_time"][:])
    FULL_FINAL = float(_ds.groups["timestepping"].variables["final_time"][:])
print(f"Full range from {os.path.basename(PREPARE_NC)}: {FULL_START}-{FULL_FINAL}")

if output_complete(RESULTS_NC, FULL_FINAL):
    print(f"Output already exists and is complete, nothing to solve: {RESULTS_NC}")
else:

    print("Loading model...")
    md = loadmodel(PREPARE_NC)
    print("Finished loading model!")

    md.cluster = generic("name", "localhost", "np", CLUSTER_NP)
    md.verbose = verbose("solution", True, "module", False, "convergence", False)

    print("md.timestepping.start_time =", md.timestepping.start_time)
    print("md.timestepping.final_time =", md.timestepping.final_time)
    print("md.timestepping.time_step =", md.timestepping.time_step)

    print("Solving transient...")
    md = solve(md, "Transient")
    print("Finished solving transient!")

    print("Saving netcdf...")
    if os.path.exists(RESULTS_NC):
        os.remove(RESULTS_NC)
    export_netCDF(md, RESULTS_NC)
    print("Finished saving netcdf!")

    print("IceVol_first=", md.results.TransientSolution[0].IceVolume)
    print("IceVol_last=", md.results.TransientSolution[-1].IceVolume)
    del md

print("=" * 60)
print("Done.")
