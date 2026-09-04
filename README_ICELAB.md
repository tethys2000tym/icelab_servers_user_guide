Are you ready to run ISSM on icelab? Let's set up the environment! Detailed tutorial is in UB Box: https://buffalo.box.com/s/0jbmm0cu86hfbab5hzrl9tkx4avs0ugc

Running on a different lab server? See [ISMIP_SERVER.md](ISMIP_SERVER.md) for the `ismip` setup guide.

## Getting Started on icelab

### 1. Access

icelab server (`icelab.caset.buffalo.edu`) is a dedicated Linux server hosted by CASET for our icelab. Unlike a shared cluster, there's no job scheduler — you run jobs directly in your shell.

- Accounts are provisioned by CASET — ask a current lab member to have you added.
- icelab is only reachable from the UB network. Off campus, connect to the UB VPN first (Cisco AnyConnect, `vpn.buffalo.edu`).
- Log in with:
  ```bash
  ssh your_ub_username@icelab.caset.buffalo.edu
  ```
  (I highy suggest using VSCode to access icelab.)

### 2. Directory structure

| Path | Purpose |
|---|---|
| `/home/your_username` | Your personal home directory — scripts and small files |
| `/data` | Shared lab storage — large datasets, model output, shared conda environments |
| `/data/envs` | Recommended location for shared conda environments (e.g. the ISSM Python environment) |
| `/data/ISSM-Linux-examples` | ISSM example files (Python and MATLAB), pre-loaded |
| `/opt/ISSM-Linux-MATLAB` | ISSM MATLAB installation (pre-installed) |
| `scripts_icelab/` (this repo) | The pipeline scripts (`S0`–`S5`) and workflow notebooks |
| `functions/` (this repo) | Shared helper functions the scripts import (SMB parameterizations, level-set reinitialization, NetCDF export, CFL step, etc.) — a copy of the lab's `issm/Functions/` directory |

### 3. Setting up ISSM — Python (Start from here!😄)

ISSM is pre-installed on icelab. To use it from Python, create a conda environment and point Python at the ISSM directories.

```bash
# one-time setup
conda init                                   # then close and reopen your terminal
conda config --add channels conda-forge
conda create -p /data/envs/my-issm python=3.11 numpy scipy matplotlib netCDF4 xarray h5py

# each session
conda activate /data/envs/my-issm
python
```

Inside Python Script:

```python
import os, sys
sys.path.append(os.getenv('ISSM_DIR') + '/bin')
sys.path.append(os.getenv('ISSM_DIR') + '/lib')
sys.path.append(os.getenv('ISSM_DIR') + '/share')
sys.path.append(os.getenv('ISSM_DIR') + '/share/proj')
```

(The scripts in this repo already include these `sys.path.append(...)` lines at the top — just make sure `ISSM_DIR` is set in your environment and that any lab-specific paths, e.g. `/home/yanmeiti/issm/Functions/`, are updated to your own account/path.)

The scripts also import shared helper functions that live in `functions/` in this repo. On icelab you can point `sys.path` at your own copy of `functions/` from this repo, see the shared python scripts.

### 4. Setting up ISSM — MATLAB (needs a graphical session via VNC, not a good choice to me, which crashed a lot of time😢 So I am using the python-version ISSM)

```bash
# on icelab, in a VNC desktop terminal
matlab
```

In the MATLAB command window:

```matlab
addpath /opt/ISSM-Linux-MATLAB/bin/
addpath /opt/ISSM-Linux-MATLAB/lib/
addpath /opt/ISSM-Linux-MATLAB/share/
addpath /opt/ISSM-Linux-MATLAB/share/proj/
```

Use MATLAB's "Set Path" dialog and "Save path for future sessions" to avoid re-adding these every time.

To set up VNC (required for MATLAB's GUI):

```bash
# on icelab
vncserver -xstartup /usr/bin/startlxde -geometry 1152x720 :4   # pick a free display number

# on your local machine, in a new terminal
ssh -L5904:localhost:5904 your_username@icelab.caset.buffalo.edu
# then connect a VNC client to localhost:5904
```

Port must be 5901 or higher; display number = port − 5900. Stop the server with `vncserver -kill :4`.

### 5. Running long jobs ?

Forward simulations can run for days. Always start them as background sessions — otherwise a dropped SSH connection kills the process.

**tmux (recommended):**

```bash
tmux new -s issm-run     #open a tmux session
conda activate /data/envs/my-issm      #activate your issm
python scripts_icelab/S3_forward_1850-2500_v3_new.py      #submit your issm scripts!
# detach with Ctrl+B then D — the job keeps running
# later, from anywhere:
tmux attach -t issm-run  #check your tmux window
```

**or nohup:**

```bash
nohup python scripts_icelab/S3_forward_1850-2500_v3_new.py > run.log 2>&1 &.        #submit your script by nohup
tail -f run.log.        #check your script progress
```

It's a good idea to test on a short time segment before committing to a full multi-century run.

### 6. Common issues

| Problem | Fix |
|---|---|
| SSH connection refused / timed out | You're off-campus and not on VPN — connect to the UB VPN first |
| SSH permission denied | Your account hasn't been added to icelab yet — ask CASET/the lab |
| `conda: command not found` | Run `conda init`, then close and reopen your terminal |
| ISSM import error in Python | Make sure your conda env is activated and the `ISSM_DIR` `sys.path` lines are present |
| VNC black screen / no desktop | The VNC server isn't running — SSH in and start it (see above) |
| Can't write files in `/data` | Write to your own subdirectory, e.g. `/data/your_username` |

For further server issues (accounts, hardware, software), open a ticket with CASET support: https://www.caset.buffalo.edu/

## Additional resources

- ISSM documentation: https://issm.jpl.nasa.gov/documentation/
- ISSM tutorials: https://issm.jpl.nasa.gov/documentation/tutorials/
- UB Caset ticket: https://www.caset.buffalo.edu/
