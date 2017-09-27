# # A simple test of SymPIM-rz which can be used for profiling the Python bottlenecks
# 
# September 26, 2017
# Nathan Cook

import numpy as np

from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.sympim_rz.integrators import integrator
from rssympim.constants import constants

#from scipy.special import j0, j1, jn_zeros
#import time
#import itertools

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib as mpl

###############################
#Define some useful parameters
################################

# species data
charge = constants.electron_charge
mass = constants.electron_mass
speed_of_light = constants.c

# plasma properties
n0 = 1.e18 # cm^-3
omega_p = np.sqrt(4.*np.pi*n0*charge*charge/mass)
k_p = omega_p/speed_of_light

# compute the simulation domain volume
l_r = 4./(k_p/(2.*np.pi)) # cm
l_z = 2./(k_p/(2.*np.pi)) # cm
volume = np.pi*l_r*l_r*l_z

# Domain parameters
n_electrons = np.round(n0*volume)

# Simulation parameters
n_macro_ptcls = 1000
macro_weight = n_electrons/n_macro_ptcls
n_r_modes = 10
n_z_modes = 10

########################################
#Define initial dictionary of parameters
########################################

_PD = {
    'np_mode': 10, #number of particles per mode
    'n_r': n_r_modes,  # number of radial modes
    'n_z': n_z_modes,  # number of longitudinal modes
    'charge': charge,  # 1 esu
    'mass': mass,  # mass in m_e
    'n_e': n_electrons, #electron density
    'n_macro': n_macro_ptcls, #total number of macro particles
    'weight': n_electrons/n_macro_ptcls, #20,  # macroparticle weighting
    'R': l_r, #4./(k_p/(2.*np.pi)), #cm 4.,  # maximum R value
    'PR': 0.5,  # maximum PR value
    'Z': l_z, #2./(k_p/(2.*np.pi)), #cm #10.,  # maximum z value
    'V': np.pi*l_r*l_r*l_z, #volume
    'num_steps': 100,  # number of steps to perform
    'n_r_max': 64,# maximum number of radial modes
    'n_z_max': 64
    }
    
    
max_fields = field_data.field_data(_PD['Z'], _PD['R'], _PD['n_r_max'], _PD['n_z_max'])
dt_max = .1*2.*np.pi/np.amax(max_fields.omega) #set timestep as 1/10 of period for largest mode
_PD['dt'] = dt_max

########################################    
#User defined functions
########################################

def create_init_conds(particles, fields, PD):
    '''Instantiate particle data'''
    fields.omega_coords = particles.mc[0] * np.ones((fields.n_modes_z, fields.n_modes_r, 2))
    fields.dc_coords = np.zeros((fields.n_modes_z, fields.n_modes_r, 2))

    #better to use linspace than arange
    particles.r = np.linspace(0.1*PD['R'],0.9*PD['R'],particles.np) #np.arange(0.1*PD['R'], 0.9*PD['R'], 0.8*PD['R']/particles.np)
    particles.z = np.linspace(0.1*PD['Z'],0.9*PD['Z'],particles.np) #np.arange(0.1*PD['Z'], 0.9*PD['Z'], 0.8*PD['Z']/particles.np)
    particles.pr = -particles.mc * np.arange(0.1, .5, .4 / particles.np)
    particles.ell = particles.weight*constants.electron_mass*constants.c*particles.r
    particles.pz = particles.mc * np.arange(0., 10., 10. / particles.np)



def vary_rz_modes(mode_pair, PD, ns=1e2):
    '''
    Simulate ns number of steps of size ss using npart particles and a number of modes
    specified by mode_pair.

    Arguments:
        mode_pair (tuple): [n_r,n_z]
        PD (dict): dictionary of other fixed parameters
        ns (Optional[int]) : number of steps to run. Defaults to 100.

    Returns:
        timing (float): number of seconds/macroparticle/mode

    '''
    num_steps = ns
    PD['n_r'] = mode_pair[0]
    PD['n_z'] = mode_pair[1]
    num_modes = mode_pair[0] + mode_pair[1]
    PD['n_macro'] = PD['np_mode']*num_modes
    num_particles = PD['n_macro']
    PD['weight'] = PD['n_e']/PD['n_macro']
    
    
    # create fields, particles, integrator
    particles = particle_data.particle_data(PD['n_macro'], PD['charge'], PD['mass'], PD['weight'])
    fields = field_data.field_data(PD['Z'], PD['R'], PD['n_z'], PD['n_r'])
    create_init_conds(particles,fields, PD)
    
    
    my_integrator = integrator.integrator(PD['dt'], fields)

    step = 0

    #print "Running {} total modes and {} particles".format(num_modes, num_particles)
    
    #t0 = time.time()
    while step < num_steps:
        my_integrator.second_order_step(particles,fields)
        
        step = step + 1
        

    return
    

###################
# Main run sequence
###################

mode_pair = np.asarray([64,64])
vary_rz_modes(mode_pair, _PD, ns=1)
