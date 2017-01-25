# Example of a 1-dimensional PWFA simulation with SymPIM-1d

from rssympim.sympim_1d.integrators import integrator
from rssympim.sympim_1d.data import field_data, particle_data
from rssympim.constants import constants
import numpy as np
import random

from matplotlib import pyplot as plt

##
## Physical parameters of the problem
##

has_beam = True
has_plasma = True

# Plasma parameters
n_plasma = 3.e18 #ptcls/cm^3
omega_p = np.sqrt(4.*np.pi*n_plasma*constants.electron_charge**2/
                  constants.electron_mass)
k_p = omega_p/constants.c
L_p = 2.*np.pi/(k_p)
print 'plasma length =', L_p

# Beam parameters
n_beam = 6.2e18 #ptcls/cm^3
beam_duration = .3*L_p # Duration of the bunch
n_macro_beam = 1e4
beam_gamma = 50.
beam_weight = beam_duration*n_beam/n_macro_beam

# Simulation domain

# Simulation length
n_periods = 10
t_final = 2.*np.pi*n_periods/omega_p

# Make the length three plasma wavelengths
plasma_lengths = 5
L_domain = plasma_lengths*L_p
ptcls_per_plasma_length = 100
n_macro = ptcls_per_plasma_length*plasma_lengths
weight = L_domain*n_plasma/n_macro

# Resolve the plasma wavelength
modes_per_length = 10
n_modes = modes_per_length*plasma_lengths

# Create the field and particle data
my_fields = field_data.field_data(L_domain, n_modes)
freqs = my_fields.omega
dt = 2.*np.pi*.1/np.max(freqs)

if has_plasma:

    my_plasma = particle_data.particle_data(n_macro,constants.electron_charge,
                                            constants.electron_mass, weight)

    ptcl_count = 0

    while ptcl_count < n_macro:
        # Start with a cold plasma
        my_plasma.z[ptcl_count] = L_domain * random.random()
        ptcl_count += 1

if has_beam:

    my_beam = particle_data.particle_data(n_macro_beam, constants.electron_charge,
                                          constants.electron_mass, beam_weight)

    ptcl_count = 0

    while ptcl_count < n_macro_beam:
        my_beam.z[ptcl_count] = random.gauss(beam_duration, beam_duration)
        my_beam.pz[ptcl_count] = beam_gamma*my_beam.mc
        ptcl_count += 1

# Field diagnostic

z_grid = np.arange(0.,L_domain,L_domain/100)

time = 0.
step = 0

# set up the integrator
my_integrator = integrator.integrator(dt, freqs)

my_integrator.half_field_forward(my_fields)

import time

my_time = time.time()

n_steps = 100

while step <= n_steps:

    # Update both species individually
    #if has_plasma:
    my_integrator.particle_update(my_plasma,my_fields)
    #if has_beam:
    my_integrator.particle_update(my_beam, my_fields)
    my_integrator.field_update(my_fields)
    my_integrator.finalize_fields(my_fields)

    step += 1
    if step%10 == 0:
        Az = my_fields.compute_Ez(z_grid)*constants.electron_charge/(
            constants.electron_mass*constants.c**2)
        plt.plot(z_grid/L_p, Az)

        title = 'Az_time_'+str(step)+'.png'
        plt.xlabel('z/L_p')
        plt.ylabel('eA/mc^2')
        plt.savefig(title)
        plt.clf()

        #plt.scatter(my_plasma.z, my_plasma.pz/my_plasma.mc,s=1)
        #plt.xlim(np.min(my_plasma.z),np.max(my_plasma.z))
        #plt.ylim(np.min(my_plasma.pz)/my_plasma.mc,
        #         np.max(my_plasma.pz)/my_plasma.mc)
        #title = 'ptcls'+str(step)+'.png'
        #plt.savefig(title)
        #plt.clf()

my_time = time.time() - my_time

print my_time

print my_time/((n_macro_beam+n_macro)*n_modes*n_steps)

print n_macro+n_macro_beam
print n_modes