from rssympim.sympim_rz.data import field_data, particle_data
from rssympim.sympim_rz.integrators import integrator

import numpy as np

import timeit


n_ptcls = 40000
n_r = 40
n_z = 100
my_fields = field_data.field_data(1.,1.,n_r,n_z)
my_ptcls  = particle_data.particle_data(n_ptcls,1.,1.,1.)

my_ptcls.r = np.ones(n_ptcls)
for idx in range(0, n_ptcls):
    my_ptcls.r[idx] *= idx * (4.) / n_ptcls + .01
my_ptcls.pr = -my_ptcls.mc * np.arange(0., .5, .5 / n_ptcls)
my_ptcls.ell = my_ptcls.r * my_ptcls.pr
my_ptcls.z = np.zeros(n_ptcls)
my_ptcls.pz = my_ptcls.mc * np.arange(0., 10., 10. / n_ptcls)

my_integrator = integrator.integrator(.01,my_fields.omega)

my_time = timeit.timeit('my_integrator.single_step(my_ptcls, my_fields)')

print my_time
