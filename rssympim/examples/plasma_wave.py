

from rssympim.sympim_rz.integrators import integrator

from rssympim.constants import constants as consts

import numpy as np

from rssympim.sympim_rz.io import field_io, particle_io

from rssympim.sympim_rz.analysis import field_analysis

from mpi4py import MPI as mpi

#
# Plasma parameters
#

n0 = 1.e17 # cm^-3
k_p = np.sqrt(4*np.pi*n0 *
                  consts.electron_charge*consts.electron_charge /
                  consts.electron_mass)/consts.c


plasma_temperature = 1. # Kelvins

# We are simulating electrons
charge = -consts.electron_charge
mass = consts.electron_mass

#
# Simulation parameters
#

# Run time considerations

domain_r = 4 # 2.*np.pi/k_p
domain_l = 5 # 2.*np.pi/k_p
steps_per_plasma_period = 15

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


while step < n_steps:

    my_integrator.half_field_forward(fld_data)
    my_integrator.single_step_ptcl(ptcl_data, fld_data)
    my_integrator.half_field_forward(fld_data)