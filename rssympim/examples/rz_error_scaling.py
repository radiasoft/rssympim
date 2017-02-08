"""
This file is for testing the second-order nature of the algorithm.

Author: Stephen Webb
"""

from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.sympim_rz.integrators import integrator
import numpy as np
from matplotlib import pyplot as plt


n_ptcls = 100

ptcl_data = particle_data.particle_data(n_ptcls, 1., 1.3, 33.5)
fld_data = field_data.field_data(5., 2., 5, 5)

ptcl_data.r = np.ones(n_ptcls)
for idx in range(0, n_ptcls):
    ptcl_data.r[idx] *= idx * (4.) / n_ptcls + .01
ptcl_data.pr = ptcl_data.mc * np.arange(0., .5, .5 / n_ptcls)
ptcl_data.ell = ptcl_data.r * ptcl_data.pr * .01
ptcl_data.z = np.zeros(n_ptcls)
ptcl_data.pz = ptcl_data.mc * np.arange(0., 10., 10. / n_ptcls)

particle_energies = ptcl_data.compute_ptcl_energy(fld_data)
field_energies = fld_data.compute_energy()
tot_energy = np.sum(particle_energies) + np.sum(field_energies)

E = []
t = []

E0 = tot_energy
n_steps = 5
step = 1

dt0 = 10./np.amax(fld_data.omega)
while step < n_steps:

    # Generate the initial conditions
    ptcl_data = particle_data.particle_data(n_ptcls, 1., 1.3, 33.5)
    fld_data = field_data.field_data(5., 2., 5, 5)

    ptcl_data.r = np.ones(n_ptcls)
    for idx in range(0, n_ptcls):
        ptcl_data.r[idx] *= idx * (4.) / n_ptcls + .01
    ptcl_data.pr = ptcl_data.mc * np.arange(0., 0.5, .5 / n_ptcls)
    ptcl_data.ell = ptcl_data.r * ptcl_data.pr * .01
    ptcl_data.z = np.zeros(n_ptcls)
    ptcl_data.pz = ptcl_data.mc * np.arange(0., 10., 10. / n_ptcls)

    # Span dt over decades
    dt = dt0/(10**step)

    # Create the new integrator
    my_integrator = integrator.integrator(dt, fld_data.omega)

    # Integrate ten steps
    for idx in range(0,10):

        my_integrator.single_step(ptcl_data, fld_data)

    particle_energies = ptcl_data.compute_ptcl_energy(fld_data)
    field_energies = fld_data.compute_energy()
    tot_energy = np.sum(particle_energies) + np.sum(field_energies)

    step += 1
    if step%10==0:
        print 'simulation completed step', str(step)

    E.append(np.abs(tot_energy-E0)/np.abs(E0))
    t.append(1/dt)

plt.loglog(t, E)
plt.xlabel(r'$1/d\tau [cm^{-1}]$')
plt.ylabel(r'$|\Delta E|$')
plt.tight_layout()
plt.show()

