#!/usr/bin/python

import numpy as np
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

ncore = 10
verbose = False

### Parameter list to simulate
# rbeads = np.array([2.35e-6])  # Bangs
# rbeads = np.array([3.78e-6])#, 2.32e-6])  # German
rbeads = np.array([4.99e-6, 3.78e-6])
seps = np.arange(1.0e-6, 20.5e-6, 1.0e-6)
# seps = np.arange(2.0e-6, 10.5e-6, 1.0e-6)
# heights = np.arange(-25.0e-6, 25.5e-6, 1.0e-6)
heights = [0]

### Attractor properties in case they need to be adjusted
density.attractor_params['include_bridge'] = False
density.attractor_params['width_goldfinger'] = 25.0e-6
density.attractor_params['width_siliconfinger'] = 25.0e-6
# density.attractor_params['height'] = 8.0e-6
density.attractor_params['total_width'] = \
                density.attractor_params['n_goldfinger'] \
                        * density.attractor_params['width_goldfinger'] \
              + (density.attractor_params['n_goldfinger'] - 1) \
                        * density.attractor_params['width_siliconfinger'] \
              + 2.0 * density.attractor_params['width_outersilicon']

density.attractor_params['height'] = 10.0e-6
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
include_edge = False

### End values for the ranges are very important, see the function
### docstring (and consider how to define the CENTERS of cubic unit
### cells, such that the unit cells themselves actually span the 
### physical space that you want).
dxyz = (1)*1e-6
#dx=2
#dy=2
#dz=1
x_range = (-200*1e-6+dxyz/2, 0e-6)
y_range = ((-density.attractor_params['total_width']/2+dxyz/2), density.attractor_params['total_width']/2)
z_range = (-density.attractor_params['total_height']/2+dxyz/2, density.attractor_params['total_height']/2)
xx, yy, zz, rho = \
    density.build_3d_array(x_range=x_range, dx=dxyz, \
                           y_range=y_range, dy=dxyz, \
                           z_range=z_range, dz=dxyz, \
                           verbose=verbose, manualadjust=False)

# x_range = (-200*1e-6+dx/2, 0e-6)
# y_range = ((-density.attractor_params['total_width']/2+dy/2), density.attractor_params['total_width']/2)
# z_range = (-density.attractor_params['total_height']/2+dz/2, density.attractor_params['total_height']/2)
# xx, yy, zz, rho = \
#     density.build_3d_array(x_range=x_range, dx=dx, \
#                            y_range=y_range, dy=dy, \
#                            z_range=z_range, dz=dz, \
#                            verbose=verbose)

if verbose:
    print("Density Loaded.")
    sys.stdout.flush()

### Some other constants and simulation parameters
G = 6.67e-11       # m^3 / (kg s^2)
rhobead = 1850.0
# rhobead = 1550.0

### Define values of the Yukawa lambda parameter to simulate
#lambdas = np.logspace(-6.7, -3, 150)
lambdas = np.logspace(-6.7, -3, 15)
#lambdas = np.array([1e-7,1e-6,1e-5])
#lambdas = [2e-6]
lambdas = lambdas[::-1]

### Y-points over which to compute the result
travel = 500.0e-6
cent = 0.0e-6
Npoints = 1000

bead_dx = travel / Npoints
beadposvec = np.linspace(cent - 0.5*travel + bead_dx, \
                         cent + 0.5*travel - bead_dx, Npoints-1)
beadposvec2 = np.linspace(-1.0*travel + bead_dx, \
                          1.0*travel - bead_dx, 2*Npoints-1)




#################################################################
#######  User shouldn't need to edit things below this   ########
#######  assuming you want to take advantage of the      ########
#######  periodicity in the attractor                    ########
#################################################################

n_goldfinger = density.attractor_params['n_goldfinger']
width_goldfinger = density.attractor_params['width_goldfinger']
width_siliconfinger = density.attractor_params['width_siliconfinger']

include_bridge = density.attractor_params['include_bridge']
silicon_bridge = density.attractor_params['silicon_bridge']

finger_length = density.attractor_params['finger_length']
attractor_height = density.attractor_params['height']

black_height = density.attractor_params['black_height']
include_black = density.attractor_params['include_black']

full_period = width_goldfinger + width_siliconfinger

### Define indices for the central gold finger and half of each of 
### the neighboring silicon fingers to take advantage of periodicity
xinds2 = np.abs(xx) <= finger_length + \
                        include_bridge*silicon_bridge
yinds2 = np.abs(yy) <= 0.5 * full_period
zinds2 = np.abs(zz) <= 0.5 * attractor_height + include_black * black_height

### Define the indices outside of the repeated unit cell structure
### to be used if include_edge=True
yinds3 = np.abs(yy) >= 0.5 * n_goldfinger * full_period

### Cut the coordinate vectors and density arrays for the repeating 
### structure and the outer silicon edge
xx2 = xx[xinds2]
yy2 = yy[yinds2]
zz2 = zz[zinds2]
rho2 = rho[xinds2,:,:][:,yinds2,:][:,:,zinds2]

yy3 = yy[yinds3]
rho3 = rho[xinds2,:,:][:,yinds3,:][:,:,zinds2]

dx = np.abs(xx[1] - xx[0])
dy = np.abs(yy[1] - yy[0])
dz = np.abs(zz[1] - zz[0])

### Assuming rectangular volume elements, convert the density grid
### to a grid of point masses
cell_volume = dx * dy * dz
m = rho * cell_volume
m2 = rho2 * cell_volume
m3 = rho3 * cell_volume

### Establish a path to save the data, and create the directory if it
### isn't already there
#results_path = os.path.abspath('../raw_results/')
#results_path = os.path.expanduser('~/raw_results_/')
#results_path = f"/home/kmkohn/2024-2025/Thesis_analysis/FEA/Data/rawdata_yukawa_voxeltest/{int(dxyz*1e6)}umspacing"
results_path = f"/home/kmkohn/2024-2025/Thesis_analysis/FEA/Data/rawdata_yukawa_platinumblack/1umspacing"

test_filename = os.path.join(results_path, 'test.p')
bu.make_all_pardirs(test_filename)

### Assuming n_goldfinger is an odd integer, this just sets up some indices
### of the fingers for use in the periodic sampling part
finger_inds = np.linspace(-1.0 * int( 0.5*n_goldfinger ), \
                           1.0 * int( 0.5*n_goldfinger ), \
                           n_goldfinger)


### Function to determine which finger you're in front of, and then
### compute the equivalent coordinate assuming you're in front of the 
### central finger. Part of the perdicity
def find_ind(ypos):
    if np.abs(ypos) <= 0.5*full_period:
        ind = 0
    else:
        extrapos = np.abs(ypos) - 0.5*full_period
        ind = np.sign(ypos) * (int(extrapos / full_period) + 1)

    newypos = ypos - ind * full_period

    return ind, newypos



def simulation(params):
    '''Simulation function taking one argument and returning one object,
       for use with joblib parallelization.'''

    ### Parse the parameters
    rbead, sep, height = params

    ### Build a filename from the parameters
    filename = 'rbead_' + str(rbead)
    filename += '_sep_' + str(sep)
    filename += '_height_' + str(height)
    filename += '.p'
    full_filename = os.path.join(results_path, filename)

    ### Instantiate a dictionary that will be populated with results
    results_dic = {}
    results_dic['order'] = 'Rbead, Sep, Height, Yuklambda'
    results_dic[rbead] = {}
    results_dic[rbead][sep] = {}
    results_dic[rbead][sep][height] = {}

    ### Some timing stuff
    all_start = time.time()
    calc_times = []

    ### A thing that needs to be in every term (POSSIBLE SIGN AMBIGUITY)
    Gterm = 2. * rbead**3

    ### Loop over the long array of bead positions and compute the fseporce from
    ### only the centrsepal finger. This can be sampled and added up to build the
    ### force curve from the entire attractor
    Gforcecurves = [[], [], []]
    # Precompute x and z separations (depend only on sep and height)
    xsep = (sep + rbead) - xx2
    zsep = height - zz2
    # reshape for broadcasting into (nx, ny, nz)
    xsep = xsep[:, None, None]
    zsep = zsep[None, None, :]

    for ind, ypos in enumerate(beadposvec2):
        # y separation vector (will broadcast into middle axis)
        ysep = (ypos - yy2)[None, :, None]

        # compute full separation and prefactor without creating meshgrid
        full_sep = np.sqrt(xsep**2 + ysep**2 + zsep**2)

        ### Refer to a soon-to-exist document expanding on Alex R's
        prefac = -1.0 * ((2. * G * m2 * rhobead * np.pi) / (3. * full_sep**2))

        Gforcecurves[0].append( np.sum(prefac * Gterm * xsep / full_sep) )
        Gforcecurves[1].append( np.sum(prefac * Gterm * ysep / full_sep) )
        Gforcecurves[2].append( np.sum(prefac * Gterm * zsep / full_sep) )

    Gforcecurves = np.array(Gforcecurves)

    ### Build interpolating functions from the long position vector
    ### and the force due to a single period of the fingers
    GX = interp.interp1d(beadposvec2, Gforcecurves[0], kind='cubic')
    GY = interp.interp1d(beadposvec2, Gforcecurves[1], kind='cubic')
    GZ = interp.interp1d(beadposvec2, Gforcecurves[2], kind='cubic')


    ### Loop over the actual array of desired bead positions, and compute the
    ### force from the full attractor at that position
    newGs = np.zeros((3, len(beadposvec)))
    for ind, ypos in enumerate(beadposvec):
        start = time.time()

        ### Compute the contribution from the points external to the 
        ### periodicity, if desired
        if include_edge:
            ysep3 = (ypos - yy3)[None, :, None]

            full_sep3 = np.sqrt(xsep**2 + ysep3**2 + zsep**2)
            prefac = -1.0 * ((2. * G * m3 * rhobead * np.pi) / (3. * full_sep3**2))

            newGs[0][ind] += np.sum(prefac * Gterm * xsep / full_sep3)
            newGs[1][ind] += np.sum(prefac * Gterm * ysep3 / full_sep3)
            newGs[2][ind] += np.sum(prefac * Gterm * zsep / full_sep3)

        ### Find the finger in which we're in front of, and compute an 
        ### equivalent position as if we're in front of the center finger
        finger_ind, newypos = find_ind(ypos)

        ### Sample the interpolating functions we built before, with one sample
        ### for each finger, properly displaced
        newGs[0][ind] += np.sum(GX(newypos + (finger_inds+finger_ind) * full_period))
        newGs[1][ind] += np.sum(GY(newypos + (finger_inds+finger_ind) * full_period))
        newGs[2][ind] += np.sum(GZ(newypos + (finger_inds+finger_ind) * full_period))  
        stop = time.time()
        calc_times.append(stop - start)

    if verbose:
        print('Computed normal grav.')
        print('Processing Yukawa modifications...')
        sys.stdout.flush()


    ### Loop over the desired values of the Yukawa lambda parameter, simulating
    ### the force for each one
    nlambda = len(lambdas)
    for yukind, yuklambda in enumerate(lambdas):
        if verbose:
            bu.progress_bar(yukind, nlambda)

        ### Refer to the non-existent LaTeX document in ../documents/ to explain this.
        ### It's a term necessary for every position
        func = np.exp(-2. * rbead / yuklambda) * (1. + rbead / yuklambda) + rbead / yuklambda - 1.

        ### Loop over the long array of values computing the force from a single finger
        yukforcecurves = [[], [], []]
        for ind, ypos in enumerate(beadposvec2):
            s = full_sep - rbead

            prefac = -1.0 * ((2. * G * m2 * rhobead * np.pi) / (3. * (s + rbead)**2))
            yukterm = 3 * yuklambda**2 * (s + rbead + yuklambda) * func * np.exp( - s / yuklambda )

            ### Build up the force curve at this point in the bead's position
            yukforcecurves[0].append( np.sum(prefac * yukterm * xsep / (s + rbead)) ) 
            yukforcecurves[1].append( np.sum(prefac * yukterm * ysep / (s + rbead)) )
            yukforcecurves[2].append( np.sum(prefac * yukterm * zsep / (s + rbead)) )

        yukforcecurves = np.array(yukforcecurves)

        ### Construct interpolating functions for the yukawa modified force term 
        ### from only the central finger
        yukX = interp.interp1d(beadposvec2, yukforcecurves[0], kind='cubic')
        yukY = interp.interp1d(beadposvec2, yukforcecurves[1], kind='cubic')
        yukZ = interp.interp1d(beadposvec2, yukforcecurves[2], kind='cubic')


        ### Loop over the actual array of desired bead positions, and compute the
        ### force from the full attractor at that position
        newyuks = np.zeros((3, len(beadposvec)))
        for ind, ypos in enumerate(beadposvec):
            start = time.time()

            ### Compute the contribution from the points external to the 
            ### periodicity, if desired
            if include_edge:
                s3 = full_sep3 - rbead

                prefac = -1.0 * ((2. * G * m3 * rhobead * np.pi) / (3. * (rbead + s3)**2))
                yukterm = 3 * yuklambda**2 * (rbead + s3 + yuklambda) * func * np.exp( - s3 / yuklambda )

                newyuks[0][ind] += np.sum(prefac * yukterm * xsep / full_sep3)
                newyuks[1][ind] += np.sum(prefac * yukterm * ysep3 / full_sep3)
                newyuks[2][ind] += np.sum(prefac * yukterm * zsep / full_sep3)

            ### Find the finger in which we're in front of, and compute an 
            ### equivalent position as if we're in front of the center finger
            finger_ind, newypos = find_ind(ypos)

            ### Sample the interpolating functions we built before, with one sample 
            ### for each finger, properly displaced
            newyuks[0][ind] += np.sum(yukX(newypos + (finger_inds+finger_ind) * full_period))
            newyuks[1][ind] += np.sum(yukY(newypos + (finger_inds+finger_ind) * full_period))
            newyuks[2][ind] += np.sum(yukZ(newypos + (finger_inds+finger_ind) * full_period))
            stop = time.time()
            calc_times.append(stop - start)

        results_dic[rbead][sep][height][yuklambda] = \
                    (newGs[0], newGs[1], newGs[2], newyuks[0], newyuks[1], newyuks[2])

    all_stop = time.time()

    if verbose:
        print("100% Done!")
        print( 'Mean: {:0.3g} ms, Std.: {:0.3g} ms    per bead-position'\
                .format(np.mean(calc_times)*1e3, np.std(calc_times)*1e3) )
        print()
        print('Total Computation Time: {:0.1f}'.format(all_stop-all_start))

        # input()


    ### Save the results to a unique filename
    results_dic['posvec'] = beadposvec
    results_dic['rhobead'] = rhobead
    results_dic['attractor_params'] = density.attractor_params
    try:
        pickle.dump(results_dic, open(full_filename, 'wb') )
    except:
        print("Save didn't work! : ", full_filename)

    ### Return the file name to avoid building up too much shit when computing
    ### thousands of different parameters with a joblib implementation
    return full_filename

### Do the sim, yo
param_list = list(itertools.product(rbeads, seps, heights))
results = Parallel(n_jobs=ncore)(delayed(simulation)(param) for param in tqdm(param_list))

