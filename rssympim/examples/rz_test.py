from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.sympim_rz.integrators import integrator
import numpy as np

from matplotlib import pyplot as plt

import time


n_ptcls = 1000
n_r = 10
n_z = 10

ptcl_data = particle_data.particle_data(n_ptcls, 1., 1.3, 33.5)
fld_data  = field_data.field_data(5., 2., n_r, n_z)

fld_data.mode_coords = np.ones((n_r, n_z, 2))

ptcl_data.r  = np.ones(n_ptcls)
for idx in range(0, n_ptcls):
    ptcl_data.r[idx] *= idx*(4.)/n_ptcls + .01
ptcl_data.pr = ptcl_data.mc*np.arange(0.,.5,.5/n_ptcls)
ptcl_data.ell = ptcl_data.r*ptcl_data.pr*.1

ptcl_data.z = np.zeros(n_ptcls)
ptcl_data.pz = ptcl_data.mc*np.arange(0.,10.,10./n_ptcls)

dt = .1/np.amax(fld_data.omega)

my_integrator = integrator.integrator(dt, fld_data)

particle_energies = ptcl_data.compute_ptcl_energy(fld_data)
field_energies = fld_data.compute_energy()
tot_energy = np.sum(particle_energies) + np.sum(field_energies)

E0 = tot_energy

E = []
t = []

E.append(tot_energy)
t.append(0.)

n_steps = 100
step = 0

t0 = time.time()
while step < n_steps:

    my_integrator.half_field_forward(fld_data)
    my_integrator.single_step_ptcl(ptcl_data, fld_data)
    my_integrator.half_field_forward(fld_data)

    # Note, this implementation will only measure first order accurate
    # energy since it is not doing the correct half-field-updates
    particle_energies = ptcl_data.compute_ptcl_energy(fld_data)
    field_energies = fld_data.compute_energy()

    step += 1
    print 'simulation completed step', str(step)

    tot_energy = np.sum(particle_energies) + np.sum(field_energies)

    E.append(abs(tot_energy-E0)/E0)
    t.append(step * dt)

tf = time.time()

print 'run time =', str(tf-t0)

plt.plot(t, E)
plt.xlabel(r'$c t [cm]$')
plt.ylabel(r'total $E$')
plt.tight_layout()
plt.show()

