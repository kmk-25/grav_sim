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

ncore = 20
verbose = True

### Parameter list to simulate
# rbeads = np.array([2.35e-6])  # Bangs
# rbeads = np.array([3.78e-6])#, 2.32e-6])  # German
rbeads = np.array([7.5e-6])
seps = np.arange(1.0e-6, 25.5e-6, 5.0e-6)
# seps = np.arange(2.0e-6, 10.5e-6, 1.0e-6)
heights = np.arange(-25.0e-6, 25.5e-6, 5.0e-6)

### Attractor properties in case they need to be adjusted
density.attractor_params['include_bridge'] = True
density.attractor_params['width_goldfinger'] = 25.0e-6
density.attractor_params['width_siliconfinger'] = 25.0e-6
density.attractor_params['height'] = 8.0e-6
# density.attractor_params['height'] = 10.0e-6

density.attractor_params['black_height'] = 3.0e-6
density.attractor_params['include_black'] = True
density.attractor_params['just_black'] = True

### Whether or not to include the outer silicon edge at the limits
### of y (I think it amounts to a 12um wide strip of silicon) so it
### shouldn't change too much. It does increase computation time by 
### a factor of a few, but is more complete to include
include_edge = True

### End values for the ranges are very important, see the function
### docstring (and consider how to define the CENTERS of cubic unit
### cells, such that the unit cells themselves actually span the 
### physical space that you want).
dxyz = 1.0e-6
x_range = (-199.5e-6, 0e-6)
y_range = (-249.5e-6, 250e-6)
z_range = (-6.5e-6, 7e-6)
# z_range = (-7.5e-6, 8e-6)
xx, yy, zz, rho = \
    density.build_3d_array(x_range=x_range, dx=dxyz, \
                           y_range=y_range, dy=dxyz, \
                           z_range=z_range, dz=dxyz, \
                           verbose=verbose)

if verbose:
    print("Density Loaded.")
    sys.stdout.flush()

### Some other constants and simulation parameters
G = 6.67e-11       # m^3 / (kg s^2)
rhobead = 1850.0
# rhobead = 1550.0

### Define values of the power law r0 parameter to simulate
r0s = np.logspace(-6.7, -3, 3)
r0s = r0s[::-1]

### Y-points over which to compute the result
travel = 500.0e-6
cent = 0.0e-6
Npoints = 200

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
height = density.attractor_params['height']

black_height = density.attractor_params['black_height']
include_black = density.attractor_params['include_black']

full_period = width_goldfinger + width_siliconfinger

### Define indices for the central gold finger and half of each of 
### the neighboring silicon fingers to take advantage of periodicity
xinds2 = np.abs(xx) <= finger_length + \
                        include_bridge*silicon_bridge
yinds2 = np.abs(yy) <= 0.5 * full_period
zinds2 = np.abs(zz) <= 0.5 * height + include_black * black_height

### Define the indices outside of the repeated unit cell structure
### to be used if include_edge=True
yinds3 = np.abs(yy) >= 0.5 * n_goldfinger * full_period

### Cut the coordinate vectors and density arrays for the repeating 
### structure and the outer silicon edge
xx2 = xx[xinds2]
yy2 = yy[yinds2]
zz2 = zz[zinds2]
rho2 = rho[xinds2,:,:][:,yinds2,:]

yy3 = yy[yinds3]
rho3 = rho[xinds2,:,:][:,yinds3,:]

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
results_path = '~/raw_results/'
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
    results_dic['order'] = 'Rbead, Sep, Height, r0'
    results_dic[rbead] = {}
    results_dic[rbead][sep] = {}
    results_dic[rbead][sep][height] = {}

    ### Some timing stuff
    all_start = time.time()
    calc_times = []

    ### A thing that needs to be in every term (POSSIBLE SIGN AMBIGUITY)
    Gterm = 2. * rbead**3

    ### Loop over the long array of bead positions and compute the force from
    ### only the central finger. This can be sampled and added up to build the
    ### force curve from the entire attractor
    Gforcecurves = [[], [], []]
    Gforcecurves_dim1 = [[], [], []]
    Gforcecurves_dim2 = [[], [], []]
    for ind, ypos in enumerate(beadposvec2):
        beadpos = [sep+rbead, ypos, height]

        ### These are used to compute projections and thus need to maintain sign.
        ### Use the xx2, yy2, and zz2 arrays which are a subselections of the full
        ### attractor covering only a single period of the fingers
        xsep, ysep, zsep = np.meshgrid(beadpos[0] - xx2, \
                                       beadpos[1] - yy2, \
                                       beadpos[2] - zz2, indexing='ij')

        ### Compute the separation between each point mass and the center 
        ### of the microsphere
        full_sep = np.sqrt(xsep**2 + ysep**2 + zsep**2)

        ### Refer to calculated values from the mathematica notebook (to do: add link to elog post)
        gravfac = 4*rbead**3/3
        dim1fac = (2*rbead*full_sep+np.log((-rbead+full_sep)/(rbead+full_sep))*(rbead**2+np.square(full_sep)))
        dim2fac = 4*rbead**3/(rbead**2-np.square(full_sep))
        prefac = -1.0 * (G * m2 * rhobead * np.pi)/np.square(full_sep)

        ### Append the computed values for the force from a single finger

        ### Standard Gravity
        Gforcecurves[0].append( np.sum(prefac * gravfac * xsep / full_sep) )
        Gforcecurves[1].append( np.sum(prefac * gravfac * ysep / full_sep) )
        Gforcecurves[2].append( np.sum(prefac * gravfac * zsep / full_sep) )

        ### 1/r^2 term
        Gforcecurves_dim1[0].append( np.sum(prefac * dim1fac * xsep / full_sep) )
        Gforcecurves_dim1[1].append( np.sum(prefac * dim1fac * ysep / full_sep) )
        Gforcecurves_dim1[2].append( np.sum(prefac * dim1fac * zsep / full_sep) )

        ### 1/r^3 term
        Gforcecurves_dim2[0].append( np.sum(prefac * dim2fac * xsep / full_sep) )
        Gforcecurves_dim2[1].append( np.sum(prefac * dim2fac * ysep / full_sep) )
        Gforcecurves_dim2[2].append( np.sum(prefac * dim2fac * zsep / full_sep) )

    Gforcecurves = np.array(Gforcecurves)
    Gforcecurves_dim1 = np.array(Gforcecurves)
    Gforcecurves_dim2 = np.array(Gforcecurves)

    ### Build interpolating functions from the long position vector
    ### and the force due to a single period of the fingers
    GX = interp.interp1d(beadposvec2, Gforcecurves[0], kind='cubic')
    GY = interp.interp1d(beadposvec2, Gforcecurves[1], kind='cubic')
    GZ = interp.interp1d(beadposvec2, Gforcecurves[2], kind='cubic')

    GX_dim1 = interp.interp1d(beadposvec2, Gforcecurves[0], kind='cubic')
    GY_dim1 = interp.interp1d(beadposvec2, Gforcecurves[1], kind='cubic')
    GZ_dim1 = interp.interp1d(beadposvec2, Gforcecurves[2], kind='cubic')

    GX_dim2 = interp.interp1d(beadposvec2, Gforcecurves[0], kind='cubic')
    GY_dim2 = interp.interp1d(beadposvec2, Gforcecurves[1], kind='cubic')
    GZ_dim2 = interp.interp1d(beadposvec2, Gforcecurves[2], kind='cubic')


    ### Loop over the actual array of desired bead positions, and compute the
    ### force from the full attractor at that position
    newGs = np.zeros((3, len(beadposvec)))
    newGs_dim1 = np.zeros((3, len(beadposvec)))
    newGs_dim2 = np.zeros((3, len(beadposvec)))
    for ind, ypos in enumerate(beadposvec):
        start = time.time()

        ### Compute the contribution from the points external to the 
        ### periodicity, if desired
        if include_edge:

            ### sep parameter is assumed to be face to face
            beadpos = [sep+rbead, ypos, height]

            ### These are used to compute projections and thus need to maintain sign
            xsep, ysep, zsep = np.meshgrid(beadpos[0] - xx2, \
                                           beadpos[1] - yy3, \
                                           beadpos[2] - zz2, indexing='ij')
            full_sep = np.sqrt(xsep**2 + ysep**2 + zsep**2)

            ### Refer to calculated values from the mathematica notebook (to do: add link to elog post)
            gravfac = 4*rbead**3/3
            dim1fac = (2*rbead*full_sep+np.log((-rbead+full_sep)/(rbead+full_sep))*(rbead**2+np.square(full_sep)))
            dim2fac = 4*rbead**3/(rbead**2-np.square(full_sep))
            prefac = -1.0 * (G * m2 * rhobead * np.pi)/np.square(full_sep)

            newGs[0][ind] += np.sum(prefac * Gterm * xsep / full_sep) 
            newGs[1][ind] += np.sum(prefac * Gterm * ysep / full_sep) 
            newGs[2][ind] += np.sum(prefac * Gterm * zsep / full_sep)

            newGs_dim1[0][ind] += np.sum(prefac * dim1fac * xsep / full_sep) 
            newGs_dim1[1][ind] += np.sum(prefac * dim1fac * ysep / full_sep) 
            newGs_dim1[2][ind] += np.sum(prefac * dim1fac * zsep / full_sep)

            newGs_dim2[0][ind] += np.sum(prefac * dim2fac * xsep / full_sep) 
            newGs_dim2[1][ind] += np.sum(prefac * dim2fac * ysep / full_sep) 
            newGs_dim2[2][ind] += np.sum(prefac * dim2fac * zsep / full_sep)

        ### Find the finger in which we're in front of, and compute an 
        ### equivalent position as if we're in front of the center finger
        finger_ind, newypos = find_ind(ypos)

        ### Sample the interpolating functions we built before, with one sample
        ### for each finger, properly displaced
        newGs[0][ind] += np.sum(GX(newypos + (finger_inds+finger_ind) * full_period))
        newGs[1][ind] += np.sum(GY(newypos + (finger_inds+finger_ind) * full_period))
        newGs[2][ind] += np.sum(GZ(newypos + (finger_inds+finger_ind) * full_period))  

        newGs_dim1[0][ind] += np.sum(GX_dim1(newypos + (finger_inds+finger_ind) * full_period))
        newGs_dim1[1][ind] += np.sum(GY_dim1(newypos + (finger_inds+finger_ind) * full_period))
        newGs_dim1[2][ind] += np.sum(GZ_dim1(newypos + (finger_inds+finger_ind) * full_period))  

        newGs_dim2[0][ind] += np.sum(GX_dim2(newypos + (finger_inds+finger_ind) * full_period))
        newGs_dim2[1][ind] += np.sum(GY_dim2(newypos + (finger_inds+finger_ind) * full_period))
        newGs_dim2[2][ind] += np.sum(GZ_dim2(newypos + (finger_inds+finger_ind) * full_period)) 


        stop = time.time()
        calc_times.append(stop - start)

    ### Each additional dimensional term is proportional to radius parameter r0^(N-1)
    ### so multiply at the end to save computation time.
    for r0 in r0s:
        results_dic[rbead][sep][height][r0] = \
                        (newGs[0], newGs[1], newGs[2], \
                        newGs_dim1[0]*r0, newGs_dim1[1]*r0, newGs_dim1[2]*r0, \
                        newGs_dim2[0]*r0**2, newGs_dim2[1]*r0**2, newGs_dim2[2]*r0**2)

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