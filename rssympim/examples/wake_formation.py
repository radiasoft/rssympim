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

from rssympim.sympim_rz.integrators import integrator

from rssympim.sympim_rz.boundaries import radial_thermal, \
    radial_reflecting, longitudinal_absorb

# typically don't import the maps, but for this example
# we need to

from rssympim.sympim_rz.maps import ptcl_maps, field_maps, similarity_maps

from rssympim.constants import constants as consts

import numpy as np

from rssympim.sympim_rz.io import field_io, particle_io

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

# We are simulating electrons
charge = consts.electron_charge
mass = consts.electron_mass

#
# Simulation parameters
#

# Run time considerations

simulation_time = 100 # 2.*np.pi/k_p, in units of plasma oscillations
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

fld_data = field_data.field_data(length, radius,
                                 n_modes_z, n_modes_r)
ptcl_data = particle_data.particle_data(n_macro_ptcls, charge, mass, macro_weight)

# Initial conditions
for ptcl_idx in range(0, n_macro_ptcls):
    # distribute the particles on a grid, gonna have to use linear stride here

macro_volume = np.sum(ptcl_data.r)
# Properly adjust the weights for constant density
for ptcl_idx in range(0, n_macro_ptcls):
    ptcl_data.qOc[ptcl_idx] *= n_macro_ptcls*ptcl_data.r[ptcl_idx]/macro_volume
    ptcl_data.q[ptcl_idx]   *= n_macro_ptcls*ptcl_data.r[ptcl_idx]/macro_volume
    ptcl_data.m[ptcl_idx]   *= n_macro_ptcls*ptcl_data.r[ptcl_idx]/macro_volume
    ptcl_data.mc[ptcl_idx]  *= n_macro_ptcls*ptcl_data.r[ptcl_idx]/macro_volume

# the temporary integrator needs maps

sim_maps = similarity_maps.similarity_maps()
ptcl_maps = ptcl_maps.ptcl_maps(dtau)
field_maps = field_maps.field_maps(fld_data, dtau)

###
#
# The fields for the parabolic charge distribution are
#
# A_z = - n_0 e (r^2/2 - r^4/r0^2/4)    r < r0
# A_z = - N_b e ln(r/r0)                r > r0

kick_parameter = 4.*np.pi*consts.electron_charge*n_beam

def phi_kick(ptcl_data, beam_pos, dtau):
    """
    Applies the space charge kick due to the beam fields
    """

    kick_pr = np.zeros(np.shape(ptcl_data.pr)[0])
    kick_pz = np.zeros(np.shape(ptcl_data.pz)[0])

    # find particles inside the beam length
    ptcl_z_within_beam = np.where(np.abs(ptcl_data.z - beam_pos) < z_beam)

    # particles inside r0 get one kick, particles outside get another
    ptcl_r_within_beam = np.where(ptcl_data.r < r_beam)
    ptcl_r_without_beam = np.where(ptcl_data.r > r_beam)

    ptcls_in_beam = np.intersect1d(ptcl_z_within_beam, ptcl_r_within_beam)
    ptcls_out_beam = np.intersect1d(ptcl_z_within_beam, ptcl_r_without_beam)

    kick_pr[ptcls_in_beam] = -kick_parameter*\
                             (.5 - .125*ptcl_data.r[ptcls_in_beam]**2/r_beam**2)*ptcl_data.r[ptcls_in_beam]*\
                             (1-(ptcl_data.z[ptcls_in_beam]-beam_pos)**2/z_beam**2)
    kick_pr[ptcls_out_beam] = -kick_parameter*\
                              (.5-.125)*r_beam**2/ptcl_data.r[ptcls_out_beam]*\
                              (1-(ptcl_data.z[ptcls_in_beam]-beam_pos)**2/z_beam**2)

    kick_pz[ptcls_in_beam] = -kick_parameter*\
                             (.5 - .125*ptcl_data.r[ptcls_in_beam]**2/r_beam**2)*ptcl_data.r[ptcls_in_beam]**2*\
                             (-(ptcl_data.z[ptcls_in_beam]-beam_pos)/z_beam**2)
    kick_pz[ptcls_out_beam] = -kick_parameter*\
                              (.5-.125)*r_beam**2*np.log(ptcl_data.r[ptcls_out_beam]/r_beam)* \
                              (-(ptcl_data.z[ptcls_in_beam] - beam_pos) / z_beam ** 2)

    kick_pz *= ptcl_data.qOc*dtau
    kick_pr *= ptcl_data.qOc*dtau

    ptcl_data.pz += kick_pz
    ptcl_data.pr += kick_pr

def S_z_external(ptcl_data, beam_pos):
    """
    Applies the similarity transformation due to the beam fields
    """

    kick_pr = np.zeros(np.shape(ptcl_data.pr)[0])
    kick_pz = np.zeros(np.shape(ptcl_data.pz)[0])

    # find particles inside the beam length
    ptcl_z_within_beam = np.where(np.abs(ptcl_data.z - beam_pos) < z_beam)

    # particles inside r0 get one kick, particles outside get another
    ptcl_r_within_beam = np.where(ptcl_data.r < r_beam)
    ptcl_r_without_beam = np.where(ptcl_data.r > r_beam)

    ptcls_in_beam = np.intersect1d(ptcl_z_within_beam, ptcl_r_within_beam)
    ptcls_out_beam = np.intersect1d(ptcl_z_within_beam, ptcl_r_without_beam)

    kick_pr[ptcls_in_beam] = -kick_parameter*\
                             (.5 - .125*ptcl_data.r[ptcls_in_beam]**2/r_beam**2)*ptcl_data.r[ptcls_in_beam]*\
                             (1-(ptcl_data.z[ptcls_in_beam]-beam_pos)**2/z_beam**2)
    kick_pr[ptcls_out_beam] = -kick_parameter*\
                              (.5-.125)*r_beam**2/ptcl_data.r[ptcls_out_beam]*\
                              (1-(ptcl_data.z[ptcls_in_beam]-beam_pos)**2/z_beam**2)

    kick_pz[ptcls_in_beam] = -kick_parameter*\
                             (.5 - .125*ptcl_data.r[ptcls_in_beam]**2/r_beam**2)*ptcl_data.r[ptcls_in_beam]**2* \
                             (1. - (ptcl_data.z[ptcls_in_beam] - beam_pos) ** 2 / z_beam ** 2)
    kick_pz[ptcls_out_beam] = -kick_parameter*\
                              (.5-.125)*r_beam**2*np.log(ptcl_data.r[ptcls_out_beam]/r_beam)* \
                              (1.-(ptcl_data.z[ptcls_in_beam] - beam_pos)**2 / z_beam ** 2)

    kick_pz *= ptcl_data.qOc
    kick_pr *= ptcl_data.qOc

    ptcl_data.pz += kick_pz
    ptcl_data.pr += kick_pr

def S_z_inverse_external(ptcl_data, beam_pos):
    """
    Applies the inverse similarity transformation due to the beam fields
    """

    kick_pr = np.zeros(np.shape(ptcl_data.pr)[0])
    kick_pz = np.zeros(np.shape(ptcl_data.pz)[0])

    # find particles inside the beam length
    ptcl_z_within_beam = np.where(np.abs(ptcl_data.z - beam_pos) < z_beam)

    # particles inside r0 get one kick, particles outside get another
    ptcl_r_within_beam = np.where(ptcl_data.r < r_beam)
    ptcl_r_without_beam = np.where(ptcl_data.r > r_beam)

    ptcls_in_beam = np.intersect1d(ptcl_z_within_beam, ptcl_r_within_beam)
    ptcls_out_beam = np.intersect1d(ptcl_z_within_beam, ptcl_r_without_beam)

    kick_pr[ptcls_in_beam] = -kick_parameter * \
                             (.5 - .125 * ptcl_data.r[ptcls_in_beam] ** 2 / r_beam ** 2) * ptcl_data.r[ptcls_in_beam] * \
                             (1 - (ptcl_data.z[ptcls_in_beam] - beam_pos) ** 2 / z_beam ** 2)
    kick_pr[ptcls_out_beam] = -kick_parameter * \
                              (.5 - .125) * r_beam ** 2 / ptcl_data.r[ptcls_out_beam] * \
                              (1 - (ptcl_data.z[ptcls_in_beam] - beam_pos) ** 2 / z_beam ** 2)

    kick_pz[ptcls_in_beam] = -kick_parameter * \
                             (.5 - .125 * ptcl_data.r[ptcls_in_beam] ** 2 / r_beam ** 2) * ptcl_data.r[
                                                                                               ptcls_in_beam] ** 2 * \
                             (-(ptcl_data.z[ptcls_in_beam] - beam_pos) / z_beam ** 2)
    kick_pz[ptcls_out_beam] = -kick_parameter * \
                              (.5 - .125) * r_beam ** 2 * np.log(ptcl_data.r[ptcls_out_beam] / r_beam) * \
                              (-(ptcl_data.z[ptcls_in_beam] - beam_pos) / z_beam ** 2)

    kick_pz *= ptcl_data.qOc
    kick_pr *= ptcl_data.qOc

    ptcl_data.pz -= kick_pz
    ptcl_data.pr -= kick_pr

def modified_ptcl_update_sequence(ptcl_data, fld_data, beam_pos, dt):

    phi_kick(ptcl_data, beam_pos, 0.5*dt)

    # always compute the new gamma_mc after the field map update
    ptcl_data.compute_gamma_mc(field_data)

    # Update sequence goes
    # M_ell S_r D_r S_r^-1 S_z D_z S_z^-1 S_r D_r S_r^-1 M_ell
    ptcl_maps.half_angular_momentum(ptcl_data)

    sim_maps.S_r(fld_data, ptcl_data)
    ptcl_maps.half_drift_r(ptcl_data, fld_data)
    sim_maps.S_r_inverse(fld_data, ptcl_data)

    S_z_external(ptcl_data, beam_pos)
    sim_maps.S_z(fld_data, ptcl_data)
    ptcl_maps.drift_z(ptcl_data)
    sim_maps.S_z_inverse(fld_data, ptcl_data)
    S_z_inverse_external(ptcl_data, beam_pos)

    sim_maps.S_r(fld_data, ptcl_data)
    ptcl_maps.half_drift_r(ptcl_data, fld_data)
    sim_maps.S_r_inverse(fld_data, ptcl_data)

    ptcl_maps.half_angular_momentum(ptcl_data)

    phi_kick(ptcl_data, beam_pos, 0.5*dt)

    # Add the delta-P to each mode
    field_data.finalize_fields()



#
# Stimulate the beam exciting the wake, then dump
#

sim_len = 2.*z0 + length
nsteps = int(sim_len/dtau)

step_num = 0

# Start the beam off the simulation domain
beam_pos = -2.*z_beam + step_num * dtau

while beam_pos < length:

    # add field maps

    modified_ptcl_update_sequence(ptcl_data, fld_data, beam_pos, 0.5*dtau)

    # Beam is moving at the speed of light
    beam_pos += dtau

    step_num += 1

# Dump the particles and fields to set up an initial condition for the next simulation

field_dumper = field_io.field_io('wake_flds', fld_data)
ptcl_dumper = particle_io.particle_io('wake_ptcls', ptcl_data)

field_dumper.dump_field(fld_data, 0)
ptcl_dumper.dump_ptcls(ptcl_data, 0)