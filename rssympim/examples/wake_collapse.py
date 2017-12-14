#####
#
# This script picks up where the wake_formation script left off,
# and begins simulating the wake collapse. The initial conditions
# are already specified by the final output of the wake_formation
# script. The only input parameters are dump periodicity and how
# long to run the simulations.
#
#####

from rssympim.sympim_rz.data import particle_data, field_data

from rssympim.sympim_rz.integrators import integrator

from rssympim.sympim_rz.boundaries import radial_thermal, \
    radial_reflecting, longitudinal_absorb

from rssympim.constants import constants as consts

import numpy as np

from rssympim.sympim_rz.io import field_io, particle_io

#####
#
# Set up the simulation parameters
#
#####

#
# Plasma parameters, make sure these match wake_formation input file
#

n0 = 1.e17 # cm^-3
k_p = np.sqrt(4*np.pi*n0 *
                  consts.electron_charge*consts.electron_charge /
                  (consts.electron_mass*consts.c))

# set the number of plasma oscillations for the simulation
n_plasma_oscillations = 100

# set the dump periodicity, i.e. how many steps between dumps
dump_periodicity = 8

# set the names of the files specifying the initial
# conditions for particles and fields
particle_file = 'wake_ptcls_0.hdf5'
fields_file = 'wake_flds_0.hdf5'

#######################
#
# Do not modify below this line
#
#######################

# Import the particle and field initial conditions
ptcl_io = particle_io.particle_io('electrons')
fld_io  = field_io.field_io('fields')

#
# Create the particle and field data classes
#

# particle data first
r, pr, z, pz, pl, weight, charge, mass = \
    ptcl_io.read_ptcls(particle_file)
n_macro = np.shape(r)[0]

electrons = particle_data.particle_data(n_macro, charge, mass, weight)
electrons.r = r
electrons.z = z

electrons.pr = pr
electrons.pz = pz

electrons.ell = pl

electrons.weight = weight

# field data second
n_modes_z, n_modes_r, L, R, P_omega, Q_omega, P_dc, Q_dc = \
    fld_io.read_field(fields_file)

fields = field_data.field_data(L, R, n_modes_z, n_modes_r)

field_data.omega_coords[:,:,0] = P_omega[:,:]
field_data.omega_coords[:,:,1] = Q_omega[:,:]

field_data.dc_coords[:,:,0] = P_dc[:,:]
field_data.dc_coords[:,:,1] = Q_dc[:,:]

#
# Create the integrator
#

# eight steps per fastest frequency
dt = (1./8.) * (2.*np.pi/np.max(fields.omega))

my_integrator = integrator.integrator(dt, fields)

#
# Set up the main loop
#

t_final = n_modes_z*2.*np.pi/k_p
t = 0.

step = 0

my_integrator.half_field_forward(field_data)

while t < t_final:

    # Integrate a single step, first particles, then fields
    my_integrator.single_step_ptcl(particle_data, field_data)
    my_integrator.single_step_fields(field_data)

    if step%dump_periodicity == 0:
        fld_io.dump_field(field_data, step)
        ptcl_io.dump_ptcls(particle_data, step)

    t += dt
    step += 1