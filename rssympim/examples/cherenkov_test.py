"""
This file is for testing for numerical Cherenkov. A coasting beam with
periodic particle boundaries moves to the right with momentum equal to
20X rest momentum. Few particles per mode, many modes, should see something
if there's anything to see.

Author: Stephen Webb
"""

from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.sympim_rz.integrators import integrator
from rssympim.constants import constants
from rssympim.sympim_rz.boundaries import radial_thermal, radial_reflecting
from rssympim.sympim_rz.analysis import field_analysis
from rssympim.sympim_rz.io import field_io
import numpy as np
from matplotlib import pyplot as plt

# species data
charge = constants.electron_charge
mass = constants.electron_mass
speed_of_light = constants.c

# plasma properties
n0 = 1.e17 # cm^-3
omega_p = np.sqrt(4.*np.pi*n0*charge*charge/mass)
k_p = omega_p/speed_of_light

# compute the simulation domain volume
plasma_lengths = 3
plasma_widths  = 1
modes_per_r = 8
modes_per_z = 8
ppm = 4

l_r = plasma_widths*2.*np.pi/k_p # cm
l_z = plasma_lengths*2.*np.pi/k_p # cm
volume = np.pi*l_r*l_r*l_z

# Domain parameters
n_electrons = n0*volume

# Simulation parameters
n_r_modes = plasma_widths*modes_per_r
n_z_modes = plasma_lengths*modes_per_z

n_macro_ptcls = ppm*n_r_modes*n_z_modes
macro_weight = n_electrons/n_macro_ptcls

# run for ten plasma periods
run_time = 2.*np.pi/k_p


# Create simulation objects
ptcl_data = particle_data.particle_data(n_macro_ptcls, charge, mass, macro_weight)
fld_data = field_data.field_data(l_r, l_z, n_r_modes, n_z_modes)

# Ten steps per fastest frequency
dt = 0.1*2.*np.pi/np.amax(fld_data.omega)

my_integrator = integrator.integrator(dt, fld_data.omega)

# Initial conditions
temp = .01*mass*speed_of_light*macro_weight

for ptcl_idx in range(0, n_macro_ptcls):
    # Uniform distribution in space
    ptcl_data.r[ptcl_idx] = np.random.random()*l_r
    ptcl_data.z[ptcl_idx] = np.random.random()*l_z/3.
    ptcl_data.pr[ptcl_idx] = np.random.normal(0.,temp)#*macro_weight*mass*speed_of_light
    ptcl_data.pz[ptcl_idx] = 10.*macro_weight*mass*speed_of_light #+ np.random.normal(0.,temp)
    ptcl_data.ell[ptcl_idx] = ptcl_data.r[ptcl_idx]*ptcl_data.pr[ptcl_idx]

# Create a thermal boundary
radial_boundary = radial_thermal.radial_thermal(temp)
#radial_boundary = radial_reflecting.radial_reflecting()

# Energy diagnostics
particle_energies_0 = np.sum(ptcl_data.compute_ptcl_energy(fld_data))
field_energies_0 = np.sum(fld_data.compute_energy())
tot_energy_0 = particle_energies_0 + field_energies_0

E = []
fld_E = []
ptcl_E = []
t = []

time = 0.
step = 0

print 'n_steps =', run_time/dt

while time < run_time:

    field_energies = fld_data.compute_energy()

    particle_energies = np.sum(ptcl_data.compute_ptcl_energy(fld_data))
    field_energies = np.sum(fld_data.compute_energy())
    tot_energy = particle_energies + field_energies

    fld_E.append(field_energies/tot_energy_0)
    ptcl_E.append(particle_energies/tot_energy_0)
    E.append(tot_energy/tot_energy_0)
    t.append(k_p * time)

    # Integrate a single step
    my_integrator.single_step(ptcl_data, fld_data)
    radial_boundary.apply_boundary(ptcl_data,fld_data)
    # periodic longitudinal boundaries, until I come up with something better
    ptcl_data.z = ptcl_data.z%l_z

    step += 1
    time += dt

    if step % 10 == 0:
        print 'completing step ', step

field_energies = fld_data.compute_energy()

particle_energies = np.sum(ptcl_data.compute_ptcl_energy(fld_data))
field_energies = np.sum(fld_data.compute_energy())
tot_energy = particle_energies + field_energies

fld_E.append(field_energies / tot_energy)
ptcl_E.append(particle_energies / tot_energy)
E.append(tot_energy / tot_energy_0)
t.append(k_p * time)

plt.plot(t, fld_E)
plt.xlabel(r'$k_p \times \tau$')
plt.ylabel(r'$E_{fld.}/E_0$')
plt.tight_layout()
plt.show()
plt.clf()

plt.plot(t, ptcl_E)
plt.xlabel(r'$k_p \times \tau$')
plt.ylabel(r'$E_{ptcl.}/E_0$')
plt.tight_layout()
plt.show()
plt.clf()

plt.plot(t, E)
plt.xlabel(r'$k_p \times \tau$')
plt.ylabel(r'$E_{tot.}/E_0$')
plt.tight_layout()
plt.show()
plt.clf()

field_dumper = field_io.field_io('test_field', fld_data)
field_dumper.dump_field(fld_data, step)

analysis = field_analysis.field_analysis()

file_name = 'test_field_'+str(step)+'.hdf5'

analysis.open_file(file_name)

analysis.plot_Er('Er_test.png')
analysis.plot_Ez('Ez_test.png')

analysis.close_file()

plt.clf()
plt.scatter(ptcl_data.z, ptcl_data.r, s=1)
plt.show()