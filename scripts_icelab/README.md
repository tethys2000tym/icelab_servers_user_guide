# Simulation Workflow

A multi-century GrIS run (e.g. 1850-2500) isn't one script — it's a chain of stages, each one solving a model state and saving it as a NetCDF "restart" file that the next stage loads. This page walks through that chain in order, using the actual scripts in this folder.

**Input data:** none of the raw input data (BedMachine geometry, MEaSUREs velocity, GISS/SMR forcing, ocean fields, etc.) is included in this repo — some files are pretty huge. Just reach out if you need any data.

## The pipeline, stage by stage

```
1. Parameterize + 1850 setup
        │
        ▼
2a. Relaxation — no calving
        │
        ▼
2b. Relaxation — calving on 
        │
        ▼
3. Forward simulation (1850-2500)
```

### 1. Parameterize + 1850 setup

- **1.1 Mesh** — create the triangle mesh for domain.
- **1.2 Parameterize** — assigns geometry (BedMachine), geothermal heat flux, initial velocity (MEaSUREs), material properties. This step internally calls ISSM's `parameterize()` on a separate "par file" (`S0_par*.py` in this folder is that kind of file — see note below).
- **1.3 Initial stress balance** — solves velocity for the as-given geometry/friction.
- **1.4 Basal friction inversion** — inverts basal friction coefficients against observed surface velocity.
- **1.5 Post-inversion stress balance** — re-solves stress balance with the inverted friction. This is the state every later stage starts from.

Each sub-step saves its own `Step1.*_..._v2.nc` and the next sub-step reloads it — so you can rerun from any sub-step without redoing the earlier ones.

> **Note on `S0_par*.py`:** these are *parameterization files*, not scripts you run directly.

### 2a. Relaxation — no calving

This lets the ice sheet relax toward equilibrium with the mass balance forcing before calving is switched on (calving can destabilize a front that hasn't settled yet). 

### 2b. Relaxation — calving on

Continues from a calving-enabled relaxation run (Von Mises calving switched on) and extends it to 2001-2100 with MAR-corrected 1901-2000 climatology.

### 3. Forward simulation 

Loads relaxation result and runs 1850-2500 forward simulation.
