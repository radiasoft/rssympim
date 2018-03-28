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

from rssympim.sympim_rz.boundaries import radial_thermal, longitudinal_thermal

from rssympim.constants import constants as consts

import numpy as np

from rssympim.sympim_rz.io import field_io, particle_io

from mpi4py import MPI as mpi

import time

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

## FACET-II parameters
N_beam  = 2.5e6  # number of electrons
sigma_r = 10.e-4   # cm
sigma_z = 10.e-4  # cm

#
# Plasma parameters
#

n0 = 2.e17 # cm^-3
k_p = np.sqrt(4*np.pi*n0 *
                  consts.electron_charge*consts.electron_charge /
                  consts.electron_mass)/consts.c


plasma_temperature = 1500. # Kelvins

# We are simulating electrons
charge = -consts.electron_charge
mass = consts.electron_mass

#
# Simulation parameters
#

# Run time considerations

beta_beam = 1.  # beam v_z/speed of light, needs to be 1 or pretty
                # close to it for the approximations on the
                # beam space charge fields.
domain_r = 3 # 2.*np.pi/k_p
domain_l = 4 # 2.*np.pi/k_p

r_modes_per_kp = 8
z_modes_per_kp = 8
num_macro_per_mode = 8

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
z_beam = np.sqrt(5)*sigma_z

# Normalization to the volume integral of the beam
#n_beam = N_beam/(8.*r_beam*r_beam*z_beam*np.pi/3.)

# Domain parameters

length = domain_l*2.*np.pi/k_p
radius = domain_r*2.*np.pi/k_p

# Step size, resolve the drive bunch

dtau = (sigma_z)/100

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

#Construct the indices for the sectioned arrays
inc = 0
end = np.zeros(size)
ind = np.zeros(size)

for i in range(size):
    if i < n_left:
        inc += 1
    end[i] += n_ptcls_per_core *(i+1)
    end[i] += inc
    if i+1 < size:
        ind[i+1] += n_ptcls_per_core *(i+1)
        ind[i+1] += inc


fld_data = field_data.field_data(length, radius,
                                 n_modes_z, n_modes_r)
ptcl_data = particle_data.particle_data(n_ptcls_per_core,
                                        charge, mass, macro_weight, n_total=n_macro_ptcls,
                                        end = end, ind = ind)

# Initial conditions
np.random.seed(0)
# uniform particle distribution in x, y, z
x = radius*np.random.rand(n_ptcls_per_core)
y = radius*np.random.rand(n_ptcls_per_core)
r = np.sqrt(x*x + y*y)
z = length*np.random.rand(n_ptcls_per_core)

print "Mean r value is {}".format(np.mean(r))

# Use colon notation to highlight bugs in counting
ptcl_data.r[:] = x[:]
ptcl_data.z[:] = z[:]

# Properly adjust the weights for constant density

# compute the number of particles contained in a "uniform" macroparticle distribution
ptcl_wgt = n0*2.*np.pi*fld_data.ptcl_width_z*fld_data.ptcl_width_r*ptcl_data.r

# scale up to get the right total number of particles
# add additional scaling for the # of particles for that core
#print "Ratio of local to total particles: {}".format(1.0*ptcl_data.np/ptcl_data.n_total)

ptcl_wgt *= (n0*np.pi*radius*radius*length/np.sum(ptcl_wgt))*(1.0*ptcl_data.np/ptcl_data.n_total)

ptcl_data.set_ptcl_weights(ptcl_wgt)

# Generate a non-relativistic thermal distribution
sigma_v = np.sqrt(consts.k_boltzmann*plasma_temperature/ptcl_data.mass)
v_x = np.random.normal(sigma_v, sigma_v, n_ptcls_per_core)
v_y = np.random.normal(0., sigma_v, n_ptcls_per_core)
v_z = np.random.normal(0., sigma_v, n_ptcls_per_core)

# pretend the r-axis is aligned to the x-axis for simplicity
ell = x*v_y
v_r = v_x

ptcl_data.ell = ell * ptcl_data.mass * ptcl_data.weight
ptcl_data.pz  = v_z * ptcl_data.mass * ptcl_data.weight
ptcl_data.pr  = v_r * ptcl_data.mass * ptcl_data.weight

#
# Simulate the beam exciting the wake, then dump
#

sim_len = (4.*z_beam + length)/2

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
    print ' n modes, r   ', n_modes_r
    print ' n modes, z   ', n_modes_z
    print ' sigma_z      ', sigma_z, 'cm'
    print ' sigma_r      ', sigma_r, 'cm'

step_num = 0

print "Rank {} processor says np is {}".format(rank,ptcl_data.np)

# Start the beam off the simulation domain
beam_pos = -2.*z_beam

modified_ptcl_update_sequence = \
    beam_integrator(sigma_r, z_beam, N_beam, dtau, fld_data)
t0 = time.time()

radial_boundary = radial_thermal.radial_thermal(plasma_temperature)
longitudinal_boundary = longitudinal_thermal.longitudinal_thermal(plasma_temperature)


# Instantiate the I/O objects
diag_period = 10
field_dumper = field_io.field_io('wake_flds', diag_period)
ptcl_dumper = particle_io.particle_io('wake_ptcls', diag_period, parallel_hdf5=True)

# Dump the particles and fields to set up an initial condition
field_dumper.dump_field(fld_data, 0)
ptcl_dumper.dump_ptcls(ptcl_data, 0)

while step_num < nsteps:

    radial_boundary.apply_boundary(ptcl_data, fld_data)
    longitudinal_boundary.apply_boundary(ptcl_data, fld_data)

    beam_pos += beta_beam * dtau / 2

    modified_ptcl_update_sequence.update(ptcl_data, fld_data, beam_pos)

    modified_ptcl_update_sequence.compute_az(ptcl_data, beam_pos)

    # Beam is moving at the speed of light
    beam_pos += beta_beam * dtau / 2

    step_num += 1

    if step_num % diag_period == 0:
        field_dumper.dump_field(fld_data, step_num)
        ptcl_dumper.dump_ptcls(ptcl_data, step_num)

    if step_num % 10 == 0:
        if rank == 0:
            print 'completing step', step_num, 'in', time.time() - t0, 'sec'