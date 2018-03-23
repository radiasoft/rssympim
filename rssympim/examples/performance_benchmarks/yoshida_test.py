"""
This file is for testing the second-order nature of the algorithm.

Author: Stephen Webb
"""

from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.sympim_rz.integrators import integrator

from rssympim.sympim_rz.integrators.integrator_yoshida \
    import integrator_y4, integrator_y6, integrator_yn

from rssympim.constants import constants
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib import pyplot as plt

import time

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
n_macro_ptcls = 100
macro_weight = n_electrons/n_macro_ptcls
n_r_modes = 10
n_z_modes = 10

# Create simulation objects
ptcl_data = particle_data.particle_data(n_macro_ptcls, charge, mass, macro_weight)
fld_data = field_data.field_data(l_r, l_z, n_r_modes, n_z_modes)

def create_init_conds(_ptcl_data, _field_data):

    _field_data.omega_coords = 1.e-6*(mass*speed_of_light/charge) * np.ones((n_z_modes, n_r_modes, 2))
    _field_data.dc_coords = 1.e-6*(mass*speed_of_light/charge) * np.ones((n_z_modes, n_r_modes, 2))

    _ptcl_data.set_ptcl_weights(np.ones(n_macro_ptcls)*macro_weight)
    _ptcl_data.r = np.arange(0.1*l_r, 0.9*l_r, 0.8*l_r/n_macro_ptcls)
    _ptcl_data.z = np.arange(0.1*l_z, 0.9*l_z, 0.8*l_z/n_macro_ptcls)
    _ptcl_data.pr = -_ptcl_data.mc * np.linspace(0.1, 5., n_macro_ptcls)
    _ptcl_data.ell = _ptcl_data.weight*constants.electron_mass*constants.c*_ptcl_data.r
    _ptcl_data.pz = _ptcl_data.mc * np.linspace(0., 10., n_macro_ptcls)

create_init_conds(ptcl_data, fld_data)

particle_energies = ptcl_data.compute_ptcl_hamiltonian(fld_data)
field_energies = fld_data.compute_energy()
tot_energy = np.sum(particle_energies) + np.sum(field_energies)

E2 = []
E4 = []
E6 = []
E8 = []
t = []

E0 = tot_energy
n_steps = 100
step = 0

dt0 = np.pi/np.amax(fld_data.omega)
t_setup = 0.
t_overhead = 0.
t0 = time.time()

while step < n_steps:

    # Generate the initial conditions
    create_init_conds(ptcl_data, fld_data)

    # Span dt over decades
    dt = dt0/((1.1)**step)

    # Create the new integrator w/ Yoshida coefficients
    t_oi = time.time()

    integrator_2nd = integrator.integrator(dt, fld_data)

    integrator_4th = integrator_y4(dt, fld_data)
    integrator_6th = integrator_y6(dt, fld_data)
    integrator_8th = integrator_yn(dt, fld_data, 8)

    t_of = time.time()
    t_overhead += t_of-t_oi

    # Integrate a single step w/ 2nd order
    integrator_2nd.update(ptcl_data, fld_data)

    particle_energies = ptcl_data.compute_ptcl_hamiltonian(fld_data)
    field_energies = fld_data.compute_energy()
    tot_energy = np.sum(particle_energies) + np.sum(field_energies)
    E2.append(np.abs(tot_energy-E0)/np.abs(E0))

    # Integrate a single step w/ 4th order using new function
    create_init_conds(ptcl_data, fld_data)
    integrator_4th.update(ptcl_data, fld_data)

    particle_energies = ptcl_data.compute_ptcl_hamiltonian(fld_data)
    field_energies = fld_data.compute_energy()
    tot_energy = np.sum(particle_energies) + np.sum(field_energies)
    E4.append(np.abs(tot_energy-E0)/np.abs(E0))

    # Integrate a single step w/ 6th order using new function
    create_init_conds(ptcl_data, fld_data)
    integrator_6th.update(ptcl_data, fld_data)

    particle_energies = ptcl_data.compute_ptcl_hamiltonian(fld_data)
    field_energies = fld_data.compute_energy()
    tot_energy = np.sum(particle_energies) + np.sum(field_energies)
    E6.append(np.abs(tot_energy-E0)/np.abs(E0))

    # Integrate a single step w/ 6th order using new function

    create_init_conds(ptcl_data, fld_data)
    integrator_8th.update(ptcl_data, fld_data)

    particle_energies = ptcl_data.compute_ptcl_hamiltonian(fld_data)
    field_energies = fld_data.compute_energy()
    tot_energy = np.sum(particle_energies) + np.sum(field_energies)
    E8.append(np.abs(tot_energy-E0)/np.abs(E0))

    print step

    step += 1

    t.append(dt)

tf = time.time()

print 'run time =', tf-t0 - t_overhead, 'secs'

t = np.amax(fld_data.omega)*np.array(t)/(2.*np.pi)
E4 = np.array(E4)
E6 = np.array(E6)
E8 = np.array(E8)

plt.loglog(t, E2, label=r'$2^{nd}$ order')
plt.loglog(t, 2.*(t**3)/10**2, label=r'$d\tau^{3}$', alpha=0.5, linestyle='-.')
plt.loglog(t, E4, label=r'$4^{th}$ order')
plt.loglog(t, 4.*(t**5)/10**1, label=r'$d\tau^{5}$', alpha=0.5, linestyle='-.')
plt.loglog(t, E6, label=r'$6^{th}$ order')
plt.loglog(t, (t**7)/10**(-1), label=r'$d\tau^{7}$', alpha=0.5, linestyle='-.')
plt.loglog(t, E8, label=r'$8^{th}$ order')
plt.loglog(t, (t**9)/10**(-1), label=r'$d\tau^{9}$', alpha=0.5, linestyle='-.')
plt.xlabel(r'$(c \Delta t) \times \frac{k_{max.}}{2 \pi}$')
plt.ylabel(r'$\left | \frac{\Delta {H}}{{H}_0} \right |$')
plt.ylim( 10.**-16,E2[0])
plt.legend()
plt.tight_layout()
plt.show()

