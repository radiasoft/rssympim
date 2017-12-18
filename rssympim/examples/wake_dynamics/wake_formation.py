##########
#
# This input file for SymPIM_rz is for simulating a beam-driven
# wake collapsing under self-consistent & kinetic influences.
#
# The initial conditions are a cold plasma with regularly space
# macroparticles. The beam is simulated using FACET parameters
# and an assumed parabolic shape. Because we have not yet sorted
# out getting charge on the grid without spurious fields, the
# beam is emulated with an analytic kick to the particles,
# assuming the beam only generates A_z.
#
# This requires some special rigging on the update sequence,
# to slip the correct kick in at the correct time for
# second-order symplectic accuracy. Hence some of the weird
# functions that seem redundant to the rest of the code base.
#
# This input file works to set up the initial conditions of
# the wake. After it is run, the actual wake collapse simulation
# is controlled with a related input file.
#
# Author: Stephen Webb
#
##########

from rssympim.sympim_rz.data import particle_data, field_data

from beam_integrator import beam_integrator

from rssympim.sympim_rz.boundaries import radial_thermal, \
    radial_reflecting, longitudinal_absorb

from rssympim.constants import constants as consts

import numpy as np

from rssympim.sympim_rz.io import field_io, particle_io

from mpi4py import MPI as mpi

###
#
# Specify the bunch parameters. Assumes a parabolic distribution
#
# n_b = n_0 X (1 - r^2/r0^2) X (1 - (z - c t)^2/z0^2)
#
# The user specifies r.m.s. quantities and the total number of
# electrons in the beam. The remaining input file will calculate
# the parameters for the parabolic distribution. The beam is assumed
# to start just off the edge of the simulation domain, and will be
# included until it leaves the domain.
#
# The normalization of this distribution is such that:
#
# N_beam = n_0 * (2*pi/3)*r0^2 z0
#
# These parameters are related to r.m.s. beam parameters by
# sigma_r^2 = r0^2/3
# and
# sigma_z^2 = z0^2/5
#
###

# The initial beam parameters are taken from the FACET-II CDR.

#
# Drive beam parameters
#

N_beam = 1.e9   # number of electrons
sigma_r = 1.e-3 # cm
sigma_z = 1.e-3 # cm

#
# Plasma parameters
#

n0 = 1.e17 # cm^-3
k_p = np.sqrt(4*np.pi*n0 *
                  consts.electron_charge*consts.electron_charge /
                  (consts.electron_mass*consts.c))

plasma_temperature = 1000.*consts.k_boltzmann

# We are simulating electrons
charge = consts.electron_charge
mass = consts.electron_mass

#
# Simulation parameters
#

# Run time considerations

beta_beam = 1. # beam v_z/speed of light
domain_r = 4 # 2.*np.pi/k_p
domain_l = 5 # 2.*np.pi/k_p
steps_per_plasma_period = 30

r_modes_per_kp = 10
z_modes_per_kp = 10
num_macro_per_mode = 10

#--------------------------------------------------------------------
#
# Do not modify below this line
#
#--------------------------------------------------------------------

# MPI stuff

comm = mpi.COMM_WORLD
size = comm.size
rank = comm.rank

# Beam parameters

r_beam = np.sqrt(3)*sigma_r
z_beam = np.sqrt(5)*sigma_z
n_beam = N_beam*1.5*r_beam*r_beam*z_beam/np.pi

# Domain parameters

length = domain_l*2.*np.pi/k_p
radius = domain_r*2.*np.pi/k_p

# Step size

dtau = (2.*np.pi/k_p)/steps_per_plasma_period

#
# Create the particle data
#

n_modes_z = z_modes_per_kp*domain_l
n_modes_r = r_modes_per_kp*domain_r
n_macro_ptcls = num_macro_per_mode*n_modes_z*n_modes_r

n_electrons = np.pi*radius*radius*length*n0
macro_weight = n_electrons/n_macro_ptcls

# particle per core
n_ptcls_per_core = n_macro_ptcls/size

# This leaves a remainder, which we distribute evenly
n_left = n_macro_ptcls%size
if rank < n_left:
    n_ptcls_per_core += 1

fld_data = field_data.field_data(length, radius,
                                 n_modes_z, n_modes_r)
ptcl_data = particle_data.particle_data(n_ptcls_per_core,
                                        charge, mass, macro_weight)

# Initial conditions

# uniform particle distribution in x, y, z
x = 2.*radius*np.random.rand(n_ptcls_per_core)-radius
y = 2.*radius*np.random.rand(n_ptcls_per_core)-radius
r = np.sqrt(x*x + y*y)
z = length*np.random.rand(n_ptcls_per_core)

# Use colon notation to highlight bugs in counting
ptcl_data.r[:] = r[:]
ptcl_data.z[:] = z[:]

macro_volume = np.sum(ptcl_data.r)
# Properly adjust the weights for constant density
scale_factor = n_ptcls_per_core*ptcl_data.r/macro_volume

ptcl_data.qOc *= scale_factor
ptcl_data.q   *= scale_factor
ptcl_data.m   *= scale_factor
ptcl_data.mc  *= scale_factor

# Generate a non-relativistic thermal distribution
v_x = np.random.normal(0., plasma_temperature/(2*consts.electron_mass), n_ptcls_per_core)
v_y = np.random.normal(0., plasma_temperature/(2*consts.electron_mass), n_ptcls_per_core)
v_z = np.random.normal(0., plasma_temperature/(2*consts.electron_mass), n_ptcls_per_core)

# pretend the r-axis is aligned to the x-axis for simplicity
ell = x*v_y
v_r = v_x

ptcl_data.ell = ell * ptcl_data.m
ptcl_data.pz  = v_z * ptcl_data.m
ptcl_data.pr  = v_r * ptcl_data.m

#
# Stimulate the beam exciting the wake, then dump
#

sim_len = 2.*z_beam + length
nsteps = int(sim_len/dtau)

if rank == 0:
    print 'Simulation parameters:'
    print '|---------------------'
    print ' domain length', length, 'cm'
    print ' domain radius', radius, 'cm'
    print ' k_p          ', k_p, 'cm^-1'
    print ' dtau         ', dtau, 'cm'
    print ' nsteps       ', nsteps
    print ' macroptcls   ', n_macro_ptcls
    print ' ptcls/core   ', n_ptcls_per_core

step_num = 0

# Start the beam off the simulation domain
beam_pos = -2.*z_beam + step_num * dtau

modified_ptcl_update_sequence = \
    beam_integrator(r_beam, z_beam, n_beam, dtau, fld_data)

while beam_pos < length:

    # add field maps

    modified_ptcl_update_sequence.update(ptcl_data, fld_data, beam_pos)

    # Beam is moving at the speed of light
    beam_pos += beta_beam*dtau

    if rank == 0:
        if step_num%100 == 0:
            print 'completing step', step_num

    step_num += 1

# Dump the particles and fields to set up an initial condition for the next simulation

field_dumper = field_io.field_io('wake_flds', fld_data)
ptcl_dumper = particle_io.particle_io('wake_ptcls', ptcl_data)

field_dumper.dump_field(fld_data, 0)
ptcl_dumper.dump_ptcls(ptcl_data, 0)