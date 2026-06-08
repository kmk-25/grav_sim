import os, sys, glob
import pickle
import numpy as np
import h5py
from tqdm import tqdm

_lib_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(_lib_dir, '../../kernel_results')
out_path = os.path.join(raw_path, 'kernel_table.h5')

# ── discover files ────────────────────────────────────────────────────────────
raw_filenames = sorted(glob.glob(os.path.join(raw_path, 'rbead_*.p')))
if not raw_filenames:
    print(f'No .p files found in {raw_path}')
    sys.exit(1)
print(f'Found {len(raw_filenames)} file(s) in {raw_path}')

# ── first pass: collect unique parameter values ───────────────────────────────
rbeads, seps, heights = set(), set(), set()
rho_bead = None
attractor_params = None
r_vals = None
yposvec = None

for fil in tqdm(raw_filenames, desc='scanning'):
    with open(fil, 'rb') as f:
        d = pickle.load(f)
    rbeads.add(d['rbead'])
    seps.add(d['sep'])
    heights.add(d['height'])
    if rho_bead is None:
        rho_bead       = d['rho_bead']
        attractor_params = d['attractor_params']
        r_vals         = d['r_vals']
        yposvec        = d['yposvec']

rbeads  = np.sort(np.array(list(rbeads)))
seps    = np.sort(np.array(list(seps)))
heights = np.sort(np.array(list(heights)))

N_rb, N_sep, N_h  = len(rbeads), len(seps), len(heights)
N_ypos, N_r       = len(yposvec), len(r_vals)

print(f'rbeads  : {N_rb}  | seps : {N_sep}  | heights : {N_h}')
print(f'yposvec : {N_ypos} pts  | r_vals : {N_r} pts')

# ── allocate output array and grid-coverage tracker ──────────────────────────
kernel     = np.full((N_rb, N_sep, N_h, N_ypos, N_r, 3), np.nan)
grid_check = np.zeros((N_rb, N_sep, N_h), dtype=int)

# ── second pass: fill array ───────────────────────────────────────────────────
for fil in tqdm(raw_filenames, desc='loading'):
    with open(fil, 'rb') as f:
        d = pickle.load(f)

    i_rb  = np.argmin(np.abs(rbeads  - d['rbead']))
    i_sep = np.argmin(np.abs(seps    - d['sep']))
    i_h   = np.argmin(np.abs(heights - d['height']))

    kernel[i_rb, i_sep, i_h] = d['table']
    grid_check[i_rb, i_sep, i_h] += 1

# ── coverage report ───────────────────────────────────────────────────────────
missing = np.argwhere(grid_check != 1)
if len(missing):
    print(f'\nWARNING: {len(missing)} missing/duplicate combo(s):')
    for idx in missing:
        print(f'  rbead={rbeads[idx[0]]:.3e}  sep={seps[idx[1]]:.3e}  '
              f'height={heights[idx[2]]:.3e}  (count={grid_check[tuple(idx)]})')
else:
    print('Grid complete — no missing combos.')

# ── write HDF5 ────────────────────────────────────────────────────────────────
with h5py.File(out_path, 'w') as hf:
    # coordinate axes
    hf.create_dataset('rbeads',  data=rbeads)
    hf.create_dataset('seps',    data=seps)
    hf.create_dataset('heights', data=heights)
    hf.create_dataset('r_vals',  data=r_vals)
    hf.create_dataset('yposvec', data=yposvec)

    # main data: (N_rb, N_sep, N_h, N_ypos, N_r, 3)
    hf.create_dataset('kernel', data=kernel, compression='gzip', compression_opts=4)

    # scalar metadata
    hf.attrs['rho_bead'] = rho_bead

    # attractor_params dict — store each entry as a root attribute
    for k, v in attractor_params.items():
        try:
            hf.attrs[f'attractor_{k}'] = v
        except Exception:
            pass  # skip any non-scalar values h5py can't serialize

print(f'\nSaved to {out_path}')
print(f'kernel shape: {kernel.shape}  (rb, sep, height, ypos, r, xyz)')
print(f'File size: {os.path.getsize(out_path) / 1e6:.1f} MB')
