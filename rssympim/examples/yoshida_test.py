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
n_macro_ptcls = 1000
macro_weight = n_electrons/n_macro_ptcls
n_r_modes = 10
n_z_modes = 10

# Create simulation objects
ptcl_data = particle_data.particle_data(n_macro_ptcls, charge, mass, macro_weight)
fld_data = field_data.field_data(l_r, l_z, n_r_modes, n_z_modes)

def create_init_conds(_ptcl_data, _field_data):

    _field_data.omega_coords = _ptcl_data.mc[0] * np.ones((n_z_modes, n_r_modes, 2))
    _field_data.dc_coords = np.zeros((n_z_modes, n_r_modes, 2))

    _ptcl_data.r = np.arange(0.1*l_r, 0.9*l_r, 0.8*l_r/n_macro_ptcls)
    _ptcl_data.z = np.arange(0.1*l_z, 0.9*l_z, 0.8*l_z/n_macro_ptcls)
    _ptcl_data.pr = -_ptcl_data.mc * np.arange(0.1, .5, .4 / n_macro_ptcls)
    _ptcl_data.ell = _ptcl_data.weight*constants.electron_mass*constants.c*_ptcl_data.r
    _ptcl_data.pz = _ptcl_data.mc * np.arange(0., 10., 10. / n_macro_ptcls)


create_init_conds(ptcl_data, fld_data)

particle_energies = ptcl_data.compute_ptcl_hamiltonian(fld_data)
field_energies = fld_data.compute_energy()
tot_energy = np.sum(particle_energies) + np.sum(field_energies)

E = []
t = []

E0 = tot_energy
n_steps = 60
step = 0

dt0 = 2.*np.pi/np.amax(fld_data.omega)

while step < n_steps:

    # Generate the initial conditions
    create_init_conds(ptcl_data, fld_data)

    # Span dt over decades
    dt = dt0/((1.1)**step)

    # Create the new integrator w/ Yoshida coefficients
    x0 = -(2.**(1./3.)/(2.-2.**(1./3.)))
    x1 = (1./(2.-2.**(1./3.)))

    forward_integrator = integrator.integrator(x1*dt, fld_data)
    backward_integrator = integrator.integrator(x0*dt, fld_data)

    # Integrate a single step w/ 4th order
    forward_integrator.half_field_forward(fld_data)
    forward_integrator.single_step_ptcl(ptcl_data, fld_data)
    forward_integrator.half_field_forward(fld_data)

    backward_integrator.half_field_forward(fld_data)
    backward_integrator.single_step_ptcl(ptcl_data, fld_data)
    backward_integrator.half_field_forward(fld_data)

    forward_integrator.half_field_forward(fld_data)
    forward_integrator.single_step_ptcl(ptcl_data, fld_data)
    forward_integrator.half_field_forward(fld_data)

    particle_energies = ptcl_data.compute_ptcl_hamiltonian(fld_data)
    field_energies = fld_data.compute_energy()
    tot_energy = np.sum(particle_energies) + np.sum(field_energies)

    step += 1

    E.append(np.abs(tot_energy-E0)/np.abs(E0))
    t.append(dt)

t = np.amax(fld_data.omega)*np.array(t)/(2.*np.pi)
E = np.array(E)

plt.loglog(t, E, label='error')
plt.loglog(t, 2.*(t**5)/10**6, label=r'$t^{5}$', alpha=0.5, linestyle='-.')
#plt.loglog(t, (t**6)/10**5, label=r'$t^{6}$', alpha=0.5, linestyle='--')
plt.xlabel(r'$(c \Delta t) \times \frac{k_{max.}}{2 \pi}$')
plt.ylabel(r'$\left | \frac{\Delta {H}}{{H}_0} \right |$')
plt.legend()
plt.tight_layout()
plt.show()

