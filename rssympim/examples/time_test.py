from rssympim.sympim_rz.data import field_data, particle_data
from rssympim.sympim_rz.integrators import integrator

import timeit

my_fields = field_data.field_data(1.,1.,10,10)
my_ptcls  = particle_data.particle_data(10000,1.,1.,1.)

my_integrator = integrator.integrator(.01,my_fields.omega)

my_time = timeit.timeit(my_integrator.single_step(my_ptcls, my_fields))

print my_time
