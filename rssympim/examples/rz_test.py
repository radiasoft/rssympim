from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.sympim_rz.integrators import integrator
import numpy as np

from matplotlib import pyplot as plt

import time

ptcl_data = particle_data.particle_data(1000, 1., 1.3, 33.5)
fld_data  = field_data.field_data(5., 2., 25, 10)

ptcl_data.r  = np.ones(1000)
for idx in range(0, 1000):
    ptcl_data.r[idx]*= idx*(4.)/1000 + .01
ptcl_data.pr = ptcl_data.mc*np.arange(0.,.5,.5/1000)
#ptcl_data.ell = ptcl_data.r*ptcl_data.pr

ptcl_data.z = np.zeros(1000)
ptcl_data.pz = ptcl_data.mc*np.arange(0.,10.,10./1000)

dt = 1./np.amax(fld_data.omega)

my_integrator = integrator.integrator(dt, fld_data.omega)

particle_energies = ptcl_data.compute_ptcl_energy(fld_data)
field_energies = fld_data.compute_energy()
tot_energy = np.sum(particle_energies) + np.sum(field_energies)

E = []
t = []

E.append(tot_energy)
t.append(0.)

n_steps = 10000
step = 0

my_integrator.half_field_back(fld_data)

t0 = time.time()
while step < n_steps:

    my_integrator.single_step(ptcl_data, fld_data)

    # Note, this implementation will only measure first order accurate
    # energy since it is not doing the correct half-field-updates
    particle_energies = ptcl_data.compute_ptcl_energy(fld_data)
    field_energies = fld_data.compute_energy()

    step += 1
    if step%10==0:
        print 'simulation completed step', str(step)

    tot_energy = np.sum(particle_energies) + np.sum(field_energies)

    E.append(tot_energy)
    t.append(step * dt)

tf = time.time()

print 'run time =', str(tf-t0)

plt.plot(t, E)
plt.xlabel(r'$c t [cm]$')
plt.ylabel(r'total $E$')
plt.tight_layout()
plt.show()

