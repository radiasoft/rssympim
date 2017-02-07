from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.sympim_rz.integrators import integrator
import numpy as np

from matplotlib import pyplot as plt

ptcl_data = particle_data.particle_data(10, 1., 1.3, 33.5)
fld_data  = field_data.field_data(5., 2., 5, 2)

ptcl_data.r  = np.arange(0.01,1.5,1.501/10)
ptcl_data.pr = ptcl_data.mc*np.arange(0.,.5,.5/10)

ptcl_data.z = np.zeros(10)
ptcl_data.pz = ptcl_data.mc*np.arange(0.,10.,10./10)

dt = 10.

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

while step < n_steps:

    my_integrator.single_step(ptcl_data, fld_data)

    particle_energies = ptcl_data.compute_ptcl_energy(fld_data)
    field_energies = fld_data.compute_energy()

    step += 1
    if step%10==0:
        print 'simulation completed step', str(step)

    tot_energy = np.sum(particle_energies) + np.sum(field_energies)

    E.append(tot_energy)
    t.append(step * dt)

plt.plot(t, E)
plt.show()

