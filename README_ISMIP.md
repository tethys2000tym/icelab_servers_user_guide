# ISSM Runbook — ismip server (ismip.caset.buffalo.edu)

This reflects what actually works on this machine for me, not a generic tutorial.

| | |
|---|---|
| **Host** | ismip |
| **ISSM version** | 4.24 |
| **CPU** | 48 cores / 96 threads |
| **Memory** | 251 GB |

> [!NOTE]
> **ISSM itself doesn't need installing.** It's a pre-built system installed at `/opt/ISSM-Linux-Python-3`. Part A below helps set up a Python environment that fits this ISSM module. (The base Python environment on ismip is 3.13, the python enviroment of ismip server ISSM is 3.11. So you need to create a Python 3.11 environment to fit the ISSM module.)


## Part A — One-time environment setup

Do this once per account, the first time you use ISSM on this server.

### 1. Confirm ISSM is in place

`ISSM_DIR` is already set system-wide in `/etc/environment`, so every login shell picks it up:

```
$ echo $ISSM_DIR
/opt/ISSM-Linux-Python-3
```

### 2. Install Miniconda (if your account doesn't have it yet)

```bash
wget https://repo.anaconda.com/miniconda3/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

Set `~/.condarc` to strict conda-forge priority — mixing in the defaults channel invites ABI conflicts:

```yaml
cat > ~/.condarc << 'EOF'
channels:
  - conda-forge
  - defaults
channel_priority: strict
EOF
```

### 3. Create a dedicated ISSM environment

Python 3.11 plus the usual scientific stack, matching the `issm_py311` env already in use on this box:

```bash
conda create -n issm_py311 python=3.11 \
    numpy scipy netCDF4 xarray h5py pandas matplotlib cftime dask \
    -c conda-forge -y

source "$(conda info --base)/etc/profile.d/conda.sh"   # make sure is activated
conda activate issm_py311
```

### 4. Put issm.exe on PATH

> [!WARNING]
> The easiest thing to trip on: `/etc/environment` sets `ISSM_DIR` but never adds `$ISSM_DIR/bin` to `PATH`. ISSM's `generic` cluster class runs `which issm.exe` at init time to locate the executable, and fails outright if it can't find it.

Add this to `~/.bashrc` (right after the conda-init block) and it's set for good:

```bash
echo 'export PATH="$ISSM_DIR/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda activate issm_py311
```

### 5. Verify

```
$ which issm.exe
/opt/ISSM-Linux-Python-3/bin/issm.exe

$ python -c "from issmversion import issmversion; issmversion()"
Ice-sheet and Sea-level System Model (ISSM) Version 4.24
Build date: Fri Aug 29 06:12:59 PDT 2025
```

A `ModuleNotFoundError: No module named 'model'` here means the script's `sys.path` isn't set up — not an environment problem. See Part B, step 2.

## Part B — Every time you run a simulation

Write the script, launch it, watch it, and clean up after — all four matter.

### 1. Activate the environment

```bash
conda activate issm_py311
```

Real-world report: if plain `conda activate` doesn't do anything (or errors with `CommandNotFoundError: 'activate' is not a conda command`), that shell hasn't had conda initialized in it — run `source "$(conda info --base)/etc/profile.d/conda.sh"` first, then activate.

### 2. Lines every script needs up top

ISSM's Python modules aren't a pip package, so they won't show up in site-packages on their own:

```python
import os, sys, numpy as np

sys.path.append(os.getenv('ISSM_DIR') + '/bin')
sys.path.append(os.getenv('ISSM_DIR') + '/lib')
sys.path.append(os.getenv('ISSM_DIR') + '/share')
sys.path.append(os.getenv('ISSM_DIR') + '/share/proj')

# ISSM 4.24 still calls np.in1d, which NumPy 2.4 removed
if not hasattr(np, 'in1d'):
    np.in1d = np.isin
```

Insert your own helper modules (e.g. a custom `generic` or `export_netCDF`) at the front with `sys.path.insert(0, ...)` so they shadow ISSM's built-in versions.

> [!WARNING]
> The ISSM 4.24 build on this server is missing a few internal modules (`generic_static`, `contourlevelzero`, `isoline`) — not something your own script imports, but something `solve.py`/`reinitializelevelset.py` need internally. Shows up as `ModuleNotFoundError: No module named 'generic_static'` or similar. You don't have to fix this problem, the copies already live in `~/Functions/`, just copy them over and add that path to the front of `sys.path` (see my script).

### 3. Configure the local cluster

```python
from generic import generic

md.cluster = generic('name', 'localhost', 'np', 8)   # np = number of MPI ranks
md.settings.solver_residue_threshold = 1e-4          # relax the residual tolerance for long runs
md = solve(md, 'Transient')
```

The solver is PETSc + MUMPS direct (LU). A MUMPS `INFO(1) = -9` error means it ran out of workspace — raise `-mat_mumps_icntl_14` in the toolkits file from 120 to 150 or 200.

Real-world report: if the literal string `'localhost'` doesn't work in your environment, use the actual hostname instead:

```python
import socket
md.cluster = generic('name', socket.gethostname(), 'np', 8)
```

### 4. Run inside tmux, tee the log

There's no working job scheduler here (Slurm is installed but its only node reports `inval`), so everything runs in the foreground — and a dropped SSH session kills it:

```bash
tmux new -s issm_run
cd ~/issm/Greenland/Scripts
python your_script.py 2>&1 | tee -a ../Model/run.log
# Ctrl-b then d to detach safely
```
