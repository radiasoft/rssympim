"""
This file is for testing the second-order nature of the algorithm.

Author: Stephen Webb
"""

from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.sympim_rz.integrators import integrator
from rssympim.constants import constants
from rssympim.sympim_rz.boundaries import longitudinal_absorb, radial_reflecting
from rssympim.sympim_rz.analysis import field_analysis
from rssympim.sympim_rz.io import field_io
import numpy as np
from matplotlib import pyplot as plt

# species data
charge = constants.electron_charge
mass = constants.electron_mass
speed_of_light = constants.c

# plasma properties
n0 = 1.e16 # cm^-3
omega_p = np.sqrt(4.*np.pi*n0*charge*charge/mass)
k_p = omega_p/speed_of_light

# compute the simulation domain volume
plasma_lengths = 2
plasma_widths  = 2
modes_per_r = 10
modes_per_z = 10
ppm = 8

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

# Create simulation objects
ptcl_data = particle_data.particle_data(n_macro_ptcls, charge, mass, macro_weight)
fld_data = field_data.field_data(l_z, l_r, n_z_modes, n_r_modes)

# Ten steps per fastest frequency
dt = 0.1*2*np.pi/np.amax(fld_data.omega)

# run for ten plasma periods
run_time = 50*dt #4.*np.pi/k_p

# Set the diagnostics
compute_energy = False

my_integrator = integrator.integrator(dt, fld_data.omega)

# Initial conditions
for ptcl_idx in range(0, n_macro_ptcls):
    # Uniform distribution in space
    ptcl_data.r[ptcl_idx] = np.random.random()*l_r
    ptcl_data.z[ptcl_idx] = np.random.normal(-0.1*l_z, 1./k_p)
    ptcl_data.pz[ptcl_idx] = 100.*mass*speed_of_light*macro_weight

# Create a thermal boundary
radial_boundary = radial_reflecting.radial_reflecting()
longitudinal_boundary = longitudinal_absorb.longitudinal_absorb()

if compute_energy:

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

# The half step at the start of the integrator
my_integrator.half_field_forward(fld_data)

while time < run_time:

    #apply particle boundaries first
    longitudinal_boundary.apply_boundary(ptcl_data, fld_data)
    radial_boundary.apply_boundary(ptcl_data, fld_data)

    if compute_energy:
        field_energies = fld_data.compute_energy()

        particle_energies = np.sum(ptcl_data.compute_ptcl_energy(fld_data))
        field_energies = np.sum(fld_data.compute_energy())
        tot_energy = particle_energies + field_energies

        fld_E.append(field_energies/tot_energy)
        ptcl_E.append(particle_energies/tot_energy)
        E.append(tot_energy/tot_energy_0)
        t.append(k_p * time)

    # Integrate a single step, first particles, then fields
    my_integrator.single_step_ptcl(ptcl_data, fld_data)
    my_integrator.single_step_fields(fld_data)

    step += 1
    time += dt

    if step % 10 == 0:
        print 'completing step ', step

# Half step back for second order
my_integrator.half_field_back(fld_data)

if compute_energy:
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

analysis.plot_Ez('Ez_test.png')
analysis.plot_Er('Er_test.png')
analysis.plot_energy_spectrum('energy_spectrum.png')

analysis.close_file()

ptcl_exists = np.where(ptcl_data.z > 0.)

plt.clf()

plt.scatter(ptcl_data.z[ptcl_exists],
            ptcl_data.r[ptcl_exists], s=1)
plt.xlabel(r'$z$')
plt.ylabel(r'$r$')
plt.tight_layout()
plt.savefig('ptcl_pos.png')

plt.clf()

plt.scatter(ptcl_data.pz[ptcl_exists]/ptcl_data.mc[ptcl_exists],
            ptcl_data.pr[ptcl_exists]/ptcl_data.mc[ptcl_exists], s=1)
plt.xlabel(r'$p_z$')
plt.ylabel(r'$p_r$')
plt.tight_layout()
plt.savefig('ptcl_mom.png')

plt.clf()

plt.scatter(ptcl_data.r[ptcl_exists],
            ptcl_data.pr[ptcl_exists]/ptcl_data.mc[ptcl_exists], s=1)
plt.xlabel(r'$r$')
plt.ylabel(r'$p_r$')
plt.tight_layout()
plt.savefig('ptcl_r.png')
