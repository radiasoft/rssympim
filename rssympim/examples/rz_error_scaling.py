"""
This file is for testing the second-order nature of the algorithm.

Author: Stephen Webb
"""

from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.sympim_rz.integrators import integrator
from rssympim.constants import constants
import numpy as np
from matplotlib import pyplot as plt

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
n_electrons = n0*volume

# Simulation parameters
n_macro_ptcls = 9000
macro_weight = n_electrons/n_macro_ptcls
n_r_modes = 30
n_z_modes = 30

# Create simulation objects
ptcl_data = particle_data.particle_data(n_macro_ptcls, charge, mass, macro_weight)
fld_data = field_data.field_data(l_r, l_z, n_r_modes, n_z_modes)

def create_init_conds(_ptcl_data, _field_data):

    _field_data.mode_coords = np.ones((n_r_modes, n_z_modes, 2))

    _ptcl_data.r = np.arange(0.1*l_r, 0.9*l_r, 0.8*l_r/n_macro_ptcls)
    _ptcl_data.z = np.arange(0.1*l_z, 0.9*l_z, 0.8*l_z/n_macro_ptcls)
    for idx in range(0, n_macro_ptcls):
        _ptcl_data.r[idx] *= idx * l_r / n_macro_ptcls + .01*l_r
        _ptcl_data.z[idx] *= idx * l_z/n_macro_ptcls
    _ptcl_data.pr = -ptcl_data.mc * np.arange(0., .5, .5 / n_macro_ptcls)
    _ptcl_data.ell = ptcl_data.r * ptcl_data.pr
    _ptcl_data.pz = ptcl_data.mc * np.arange(0., 1000., 1000. / n_macro_ptcls)


create_init_conds(ptcl_data, fld_data)

particle_energies = ptcl_data.compute_ptcl_energy(fld_data)
field_energies = fld_data.compute_energy()
tot_energy = np.sum(particle_energies) + np.sum(field_energies)

print 'initial field energies =',np.sum(field_energies)
print 'initial particle energies =', np.sum(particle_energies)

E = []
t = []

E0 = tot_energy
n_steps = 18
step = 0

dt0 = 32./np.amax(fld_data.omega)

while step < n_steps:

    # Generate the initial conditions
    create_init_conds(ptcl_data, fld_data)

    # Span dt over decades
    dt = dt0/(2**step)

    # Create the new integrator
    my_integrator = integrator.integrator(dt, fld_data.omega)

    # Integrate a single step
    my_integrator.single_step(ptcl_data, fld_data)

    particle_energies = ptcl_data.compute_ptcl_energy(fld_data)
    field_energies = fld_data.compute_energy()
    tot_energy = np.sum(particle_energies) + np.sum(field_energies)

    print ' field energies =', np.sum(field_energies)
    print ' particle energies =', np.sum(particle_energies)

    step += 1

    E.append(np.abs(tot_energy-E0)/np.abs(E0))
    t.append(1/dt)

t = np.array(t)
E = np.array(E)

print E

plt.loglog(t, E)
plt.loglog(t, 10**5/(t*t*t))
plt.xlabel(r'$1/d\tau~ [cm^{-1}]$')
plt.ylabel(r'$|\Delta E/E_0|$')
plt.tight_layout()
plt.show()

