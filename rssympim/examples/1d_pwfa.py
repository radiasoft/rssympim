# Example of a 1-dimensional PWFA simulation with SymPIM-1d

from rssympim.sympim_1d.integrators import integrator
from rssympim.sympim_1d.data import field_data, particle_data
from rssympim.constants import constants
import numpy as np

##
## Physical parameters of the problem
##

# Plasma parameters
n_plasma = 1.e17 #ptcls/cm^3
omega_p = np.sqrt(4.*np.pi*n_plasma*constants.electron_charge**2/
                  constants.electron_mass)
k_p = omega_p/constants.c

# Beam parameters
n_beam = 1e9 #total ptcls
beam_length = 1.e-9 #nanoseconds
n_macro_beam = 1e3
beam_weight = int(n_beam/n_macro_beam)

# Simulation domain

# Simulation length
n_periods = 100
t_final = n_periods/omega_p

# Make the length three plasma wavelengths
plasma_lengths = 3
L_domain = 3/k_p
ptcls_per_plasma_length = 1000
n_macro = ptcls_per_plasma_length*plasma_lengths
weight = L_domain*n_plasma/n_macro

# Resolve the plasma wavelength
modes_per_length = 10
n_modes = modes_per_length*plasma_lengths

# Create the field and particle data
my_fields = field_data.field_data(L_domain, n_modes)
freqs = my_fields.omega
dt = .1/np.max(freqs)

my_plasma = particle_data.particle_data(n_macro,constants.electron_charge,
                                        constants.electron_mass, weight)

my_beam = particle_data.particle_data(n_macro_beam, constants.electron_charge,
                                      constants.electron_mass, beam_weight)

time = 0.

# set up the integrator
my_integrator = integrator.integrator(dt, freqs)

my_integrator.half_field_forward(my_fields)

while time <= t_final:

    # Update both species individually
    my_integrator.particle_update(my_plasma, my_fields)
    my_integrator.particle_update(my_beam, my_fields)
    my_integrator.field_update(my_fields)
    my_integrator.finalize_fields(my_fields)

    time += dt

    print 'taking step for t =', time,' secs'