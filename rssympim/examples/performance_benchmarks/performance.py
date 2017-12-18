# # Performance Testing for SymPIM-rz - script generated from IPython notebook
# 
# This will follow the BeamLoad approach, varying the number of modes in the longitudinal and radial direction as well as the number of macro-particles, to produce a few plots showing varying across a range of parameters.
# 
# Nathan Cook
# April 13, 2017
#

import numpy as np
from rssympim.constants import constants as consts
from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.sympim_rz.integrators import integrator
from scipy.special import j0, j1, jn_zeros
import time
import itertools

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl


# Define dictionary with parameters

_PD = {
    'num_p': 10, #total # of particles
    'np_mode': 10, #number of particles per mode
    'n_r': 2,  # number of radial modes
    'n_z': 4,  # number of longitudinal modes
    'charge': 1,  # 1 esu
    'mass': 1.,  # mass in m_e
    'weight': 20,  # macroparticle weighting
    'R': 4.,  # maximum R value
    'PR': 0.5,  # maximum PR value
    'Z': 10.,  # maximum z value
    'num_steps': 100,  # number of steps to perform
    'n_r_max': 5,# maximum number of radial modes
    'n_z_max': 5
    }


# ## Functions for constructing test
# 
# Use the ones constructe for running tests.

def make_particles(PD):
    # instantiate particle data
    ptcl_data = particle_data.particle_data(PD['num_p'], PD['charge'], PD['mass'], PD['weight'])

    # Add r data - evenly distribute out to r = R
    ptcl_data.r = np.arange(0., PD['R'], PD['R'] / PD['num_p']) + PD['R'] / PD['num_p']
    # Add pr and ell data - evenly distribute pr out to PR
    ptcl_data.pr = ptcl_data.mc * np.arange(0., PD['PR'], PD['PR'] / PD['num_p'])
    ptcl_data.ell = ptcl_data.r * ptcl_data.pr * .1

    # Add z data - evenly distribute out to z = Z/10.
    ptcl_data.z = np.linspace(0, _PD['Z'] / 10., _PD['num_p']) #ptcl_data.z = np.zeros(PD['num_p'])
    ptcl_data.pz = ptcl_data.mc * np.arange(0., PD['Z'], PD['Z'] / PD['num_p'])

    return ptcl_data


def make_fields(PD):
    # Instantiate field data
    fld_data = field_data.field_data(PD['Z'], PD['R'], PD['n_r'], PD['n_z'])

    # Give mode Ps and Qs normalized values of '1'
    fld_data.mode_coords = np.ones((PD['n_r'], PD['n_z'], 2))

    return fld_data


def run_integrator():
    '''Tests the integrator by running a single step and checking coordinates'''

    # create fields, particles, integrator
    particles = make_particles(_PD)
    fields = make_fields(_PD)

    dt = .1 / np.amax(fields.omega)
    my_integrator = integrator.integrator(dt, fields.omega)

    # take a single step
    my_integrator.single_step(particles, fields)


# ## Construct complete mode list and use to determine fixed step size
# 
# We'll construct the mode list for the maximum number of modes, corresponding to the smallest required step size, and use that step size for all of our runs for consistency.

r_mode_range = np.asarray([2,4,8,16,32,64,128,256])
z_mode_range = np.asarray([2,4,8,16,32,64,128,256])

_PD['n_r_max'] = max(r_mode_range)
_PD['n_z_max'] = max(z_mode_range)


#define the mode pairings
rz_modes = np.asarray(list(itertools.permutations(r_mode_range, 2)))

max_fields = field_data.field_data(_PD['Z'], _PD['R'], _PD['n_r_max'], _PD['n_z_max'])
dt_max = .1 / np.amax(max_fields.omega) #set timestep as 1/10 of period for largest mode
_PD['dt'] = dt_max


# #### Construct script for varying modes - for same # of particles (not varying 3 at once - just going for r and z mode variation here)

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
    PD['num_p'] = PD['np_mode']*num_modes
    num_particles = PD['num_p']

    # create fields, particles, integrator
    particles = make_particles(PD)
    fields = make_fields(PD)
    my_integrator = integrator.integrator(PD['dt'], fields)

    step = 0

    print "Running {} total modes and {} particles".format(num_modes, num_particles)
    
    t0 = time.time()
    while step < num_steps:
        my_integrator.single_step_fields(fields)
        my_integrator.single_step_ptcl(particles,fields)
        
        step = step + 1
        
    t1 = time.time()

    #print "Performed {:e} steps with {} particles and {} modes in {}s".format(num_steps, num_particles, num_modes, time1-time0)
    
    s_per_step = (t1-t0)/num_steps
    s_combined = s_per_step/(num_modes*num_particles)

    
    return s_combined



# #### Run for a moderate parameter set


times = []
rvs = []
zvs = []
bothvs = []
for rv in r_mode_range:
    for zv in z_mode_range:
        mode_pair = np.asarray([rv,zv]) #mode pair should be [n_r,n_z]
        speed = vary_rz_modes(mode_pair,_PD, 25)
        rvs.append(rv)
        zvs.append(zv)
        times.append(speed)


#create x/y matrices for the psuedocolor plot - reshape values in the same way
num_r = len(r_mode_range)
num_z = len(z_mode_range)

p_vals1 = np.asarray(rvs).reshape(num_r,num_z).transpose() #x coordinate is radial modes
m_vals1 = np.asarray(zvs).reshape(num_r,num_z).transpose() #y coordinate is longitudinal modes
plot_times1 = np.asarray(times).reshape(num_r,num_z).transpose() #color coordinate
log_plot_times1 = np.log10(plot_times1)

r_mode_labels = list(r_mode_range)
z_mode_labels = list(z_mode_range) #["{:.0f}".format(val) for val in np.power(2,a2)]

#save values to file
np.savetxt('256_modes_logplottimes.txt',log_plot_times1)
np.savetxt('256_modes_rvals.txt',p_vals1)
np.savetxt('256_modes_zvals.txt',m_vals1)

#pad arrays
padded_lptimes = np.lib.pad(log_plot_times1, (0,1), 'edge')

padded_p_vals = np.lib.pad(p_vals1, (0,1), 'constant', constant_values=(r_mode_range[-1]*2))
padded_p_vals[8,:] = padded_p_vals[7,:]
#print padded_p_vals

padded_m_vals = np.lib.pad(m_vals1, (0,1), 'edge')
padded_m_vals[8,:] = padded_m_vals[7,:]*2
#print padded_m_vals


################
#Create the plot
################

c_min = np.min(log_plot_times1)
c_max = np.max(log_plot_times1)

with mpl.style.context('rs_paper'):

    mpl.rcParams['font.serif'] = 'Palantino'
    mpl.rcParams['axes.titlesize'] = 14.4
    
    fig = plt.figure()
    ax = fig.gca()
    
    #Construct color axis early
    ax1 = fig.add_axes([0.92, 0.125, 0.05, 0.754]) #fig.add_axes([0.92, 0.125, 0.05, 0.775])
    
    #ax.pcolor(np.log2(p_vals1), np.log2(m_vals1), log_plot_times1, cmap='bone_r', vmin=c_min, vmax=c_max)
    ax.pcolor(np.log2(padded_p_vals), np.log2(padded_m_vals), padded_lptimes, cmap='bone_r', vmin=c_min, vmax=c_max)
    ax.set_title(r'SymPIM performance - 10 particles-per-mode (serial, 2.5GHz laptop)', x=0.64)
    ax.set_xlabel(r'Number of radial modes')
    ax.set_ylabel(r'Number of longitudinal modes')
    
    #Manually set ticks for axes
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    
    new_xticks = xticks+0.5 #offset by 0.5 in exponent
    new_yticks = yticks+0.5 #offset by 0.5 in exponent
    
    ax.set_xticks(new_xticks)
    ax.set_yticks(new_yticks)
    ax.set_yticklabels(z_mode_labels)
    ax.set_xticklabels(r_mode_labels)
    ax.minorticks_off()
    
    #Reset limits after adjusting ticks
    ax.set_xlim([np.log2(2),np.log2(512)])
    ax.set_ylim([np.log2(2),np.log2(512)])
    
    #Manually construct colorbar axis    
    norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap='bone_r',
                                norm=norm,
                                orientation='vertical',
                                spacing = 'proportional')
    cb1.set_label(r'time/(particle-mode-step) [$\mu$s]')
    
    #determine ticks
    color_ticks = ax1.get_yticks()
    c_min = np.min(log_plot_times1) #np.log10(5e-7) #np.min(log_plot_times1)
    c_max = np.max(log_plot_times1) #np.log10(250e-6) #np.max(log_plot_times1)
    m_cbar = ( c_max-c_min) #positive slope
    exp_vals = color_ticks*m_cbar + c_min #these are the values of the corresponding exponents
    reg_vals = np.power(10.,exp_vals) #these are the actual timing values
    formatted_vals = ["{:.1f}".format(val*1e6) for val in reg_vals]
    
    ax1.set_yticklabels(formatted_vals)
    
    fig.savefig('NEW_256modes_PPM_SymPIM_chart.pdf',bbox_inches='tight')

