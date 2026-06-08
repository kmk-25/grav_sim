
import numpy as np
from math import pi

import matplotlib.pyplot as plt
import pickle as pickle

import scipy.interpolate as interp
import scipy.signal as signal
import scipy.optimize as opti
import scipy, sys, time, os, itertools

import build_attractor_v2_density as density
import bead_util as bu

from numba import jit
from datetime import date

from tqdm import tqdm
from joblib import Parallel, delayed

ncore = 20
verbose = False

### Parameter list to simulate
# rbeads = np.array([2.35e-6])  # Bangs
# rbeads = np.array([3.78e-6])#, 2.32e-6])  # German
rbeads = np.array([4.99e-6, 3.78e-6])
#seps = np.arange(1.0e-6, 25.5e-6, 1.0e-6)
seps = np.arange(1.0e-6, 20.5e-6, 1.0e-6)
# seps = np.arange(2.0e-6, 10.5e-6, 1.0e-6)
#heights = np.arange(-25.0e-6, 25.5e-6, 1.0e-6)
heights = [0]

### Attractor properties in case they need to be adjusted
density.attractor_params['include_bridge'] = False
density.attractor_params['width_goldfinger'] = 25.0e-6
density.attractor_params['width_siliconfinger'] = 25.0e-6
# density.attractor_params['height'] = 8.0e-6
density.attractor_params['height'] = 10.0e-6
density.attractor_params['total_width'] = \
                density.attractor_params['n_goldfinger'] \
                        * density.attractor_params['width_goldfinger'] \
              + (density.attractor_params['n_goldfinger'] - 1) \
                        * density.attractor_params['width_siliconfinger'] \
              + 2.0 * density.attractor_params['width_outersilicon']

density.attractor_params['black_height'] = 3.0e-6
density.attractor_params['include_black'] = True
density.attractor_params['just_black'] = False
density.attractor_params['total_height'] = \
                density.attractor_params['height'] \
                        + 2*density.attractor_params['black_height']*density.attractor_params['include_black']

### Whether or not to include the outer silicon edge at the limits
### of y (I think it amounts to a 12um wide strip of silicon) so it
### shouldn't change too much. It does increase computation time by 
### a factor of a few, but is more complete to include
include_edge = True

### End values for the ranges are very important, see the function
### docstring (and consider how to define the CENTERS of cubic unit
### cells, such that the unit cells themselves actually span the 
### physical space that you want).
dxyz = (1)*1e-6

def shell_sum(pos, xx, yy, zz, m, r, rb):
	"""Sum m_i * sep_i * r'_i over attractor points within the shell r-rb < r' < r+rb.

	pos  : [x, y, z] bead position
	xx, yy, zz : 1D coordinate arrays from build_3d_array
	m    : 3D mass array (rho * cell_volume), same shape as meshgrid of xx,yy,zz
	r    : shell radius to evaluate
	rb   : half-width of the shell (bead radius)
	"""
	xsep, ysep, zsep = np.meshgrid(pos[0] - xx, pos[1] - yy, pos[2] - zz, indexing='ij')
	r_prime = np.sqrt(xsep**2 + ysep**2 + zsep**2)
	mask = (r_prime > r - rb) & (r_prime < r + rb)
	mw = m[mask]
	rp = r_prime[mask]
	sep_arr = np.array([xsep[mask], ysep[mask], zsep[mask]])
	linsum  = np.sum(mw * sep_arr / rp,    axis=1)
	quadsum = np.sum(mw * sep_arr / rp**3, axis=1)
	return np.array([linsum, quadsum])

def outersum(pos, xx, yy, zz, m, r, rb, rho_bead):
	result = shell_sum(pos, xx, yy, zz, m, r, rb)
	return pi * rho_bead * r * (result[0] - result[1] * (r**2 - rb**2))


def _kernel_at_pos(pos, xx, yy, zz, m, r_vals, rb, rho_bead):
	"""Compute outersum at all r_vals for a single position.

	Builds the full separation arrays once, sorts by distance, then uses
	searchsorted for O(log N) shell slicing per r — avoids re-scanning the
	full attractor for every r value.
	"""
	Xsep, Ysep, Zsep = np.meshgrid(pos[0] - xx, pos[1] - yy, pos[2] - zz, indexing='ij')
	r_prime  = np.sqrt(Xsep**2 + Ysep**2 + Zsep**2).ravel()
	sort_idx = np.argsort(r_prime)
	r_s = r_prime[sort_idx]
	m_s = m.ravel()[sort_idx]
	x_s = Xsep.ravel()[sort_idx]
	y_s = Ysep.ravel()[sort_idx]
	z_s = Zsep.ravel()[sort_idx]

	out = np.zeros((len(r_vals), 3))
	for j, r in enumerate(r_vals):
		lo = np.searchsorted(r_s, r - rb, side='left')
		hi = np.searchsorted(r_s, r + rb, side='right')
		if lo >= hi:
			continue
		mw      = m_s[lo:hi]
		rp      = r_s[lo:hi]
		sep_arr = np.array([x_s[lo:hi], y_s[lo:hi], z_s[lo:hi]])
		linsum  = np.sum(mw * sep_arr / rp,    axis=1)
		quadsum = np.sum(mw * sep_arr / rp**3, axis=1)
		out[j]  = pi * rho_bead * r * (linsum - quadsum * (r**2 - rb**2))
	return out


def compute_kernel_table(pos_list, r_vals, rb, xx, yy, zz, m, rho_bead):
	"""Compute outersum at every (position, r) pair, parallelized over positions.

	pos_list : array-like, shape (N, 3) — bead positions [x, y, z]
	r_vals   : 1D array of shell radii to evaluate
	rb       : bead radius
	xx, yy, zz : 1D coordinate arrays from build_3d_array
	m        : 3D mass array (rho * cell_volume)
	rho_bead : bead material density

	Returns array of shape (N, len(r_vals), 3) — outersum x/y/z at each (pos, r).
	"""
	pos_list = np.asarray(pos_list)
	r_vals   = np.asarray(r_vals)
	rows = Parallel(n_jobs=ncore, prefer='threads')(
		delayed(_kernel_at_pos)(pos, xx, yy, zz, m, r_vals, rb, rho_bead)
		for pos in pos_list
	)
	return np.array(rows)


def main():
	### Attractor grid — mirrors save_force_curve_periodic_parallel settings
	dxyz = 1.0e-6
	dr = 0.5e-6
	density.attractor_params['include_bridge'] = False
	density.attractor_params['width_goldfinger'] = 25.0e-6
	density.attractor_params['width_siliconfinger'] = 25.0e-6
	density.attractor_params['height'] = 10.0e-6
	density.attractor_params['total_width'] = (
		density.attractor_params['n_goldfinger'] * density.attractor_params['width_goldfinger']
		+ (density.attractor_params['n_goldfinger'] - 1) * density.attractor_params['width_siliconfinger']
		+ 2.0 * density.attractor_params['width_outersilicon']
	)
	density.attractor_params['black_height'] = 3.0e-6
	density.attractor_params['include_black'] = True
	density.attractor_params['just_black'] = False
	density.attractor_params['total_height'] = (
		density.attractor_params['height']
		+ 2 * density.attractor_params['black_height'] * density.attractor_params['include_black']
	)

	x_range = (-200e-6 + dxyz/2, 0.0)
	y_range = (-density.attractor_params['total_width']/2 + dxyz/2,
	            density.attractor_params['total_width']/2)
	z_range = (-density.attractor_params['total_height']/2 + dxyz/2,
	            density.attractor_params['total_height']/2)
	xx, yy, zz, rho = density.build_3d_array(
		x_range=x_range, dx=dxyz,
		y_range=y_range, dy=dxyz,
		z_range=z_range, dz=dxyz,
		verbose=True,
	)
	m = rho * dxyz**3

	### Bead and position parameters — mirrors save_force_curve_periodic_parallel
	rbeads   = np.array([4.99e-6])
	seps     = np.array([5.0e-6])   # single sep for timing test
	heights  = [0.0]
	rho_bead = 1850.0

	travel  = 500.0e-6
	Npoints = 1000
	bead_dx = travel / Npoints
	yposvec = np.linspace(-travel + bead_dx, travel - bead_dx, 2*Npoints - 1)

	r_vals = np.arange(dr, 250e-6 + dr, dr)

	results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../kernel_results')
	os.makedirs(results_path, exist_ok=True)

	param_list = list(itertools.product(rbeads, seps, heights))
	t_start = time.time()
	for rbead, sep, height in tqdm(param_list):
		filename = f'rbead_{rbead:.3e}_sep_{sep:.3e}_height_{height:.3e}.p'
		full_path = os.path.join(results_path, filename)

		pos_list = np.column_stack([
			np.full(len(yposvec), sep + rbead),
			yposvec,
			np.full(len(yposvec), height),
		])
		table = compute_kernel_table(pos_list, r_vals, rbead, xx, yy, zz, m, rho_bead)

		payload = {
			'table':            table,       # shape (N_ypos, N_r, 3)
			'r_vals':           r_vals,
			'yposvec':          yposvec,
			'rbead':            rbead,
			'sep':              sep,
			'height':           height,
			'rho_bead':         rho_bead,
			'attractor_params': density.attractor_params,
		}
		with open(full_path, 'wb') as f:
			pickle.dump(payload, f)

	elapsed = time.time() - t_start
	print(f'Saved {len(param_list)} files to {results_path}')
	print(f'Total time: {elapsed:.2f} s  ({elapsed/len(param_list):.2f} s/combo)')


if __name__ == '__main__':
	main()

