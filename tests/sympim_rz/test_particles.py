import numpy as np
from rssympim.constants import constants as consts
from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.sympim_rz.integrators import integrator

# dictionary for fixed values
_PD = {
    'num_p': 10,
    'n_r': 2,  # number of radial modes
    'n_z': 4,  # number of longitudinal modes
    'charge': 1,  # 1 esu
    'mass': 1.,  # mass in m_e
    'weight': 20,  # macroparticle weighting
    'R': 4.,  # maximum R value
    'PR': 0.5,  # maximum PR value
    'Z': 10.,  # maximum z value
    'num_steps': 1000,  # number of steps to perform
    'expected_energy': np.asarray(
        [5.995849160000000000e+11, 8.484762139977631836e+11, 1.342066067075586182e+12, 1.898207154006640137e+12,
         2.475087695233998535e+12,
         3.061004316744099121e+12, 3.651609971451465820e+12, 4.244947934725706055e+12, 4.840013448525821289e+12,
         5.436239235329887695e+12]),
    'expected_r': np.asarray(
        [0.160000000000000031, 0.640000000000000124, 1.200000000000000178, 1.600000000000000089, 2.,
         2.399999999999999911, 2.800000000000000266, 3.200000000000000178, 3.600000000000000089, 4.]),
    'expected_pr': np.asarray(
        [0.000000000000000000e+00, -2.997924580000000000e+10, 5.995849160000000000e+10, 8.993773740000001526e+10,
         1.199169832000000000e+11,
         1.498962290000000000e+11, 1.798754748000000305e+11, 2.098547206000000305e+11, 2.398339664000000000e+11,
         2.698132122000000000e+11])
}

_EPSILON = 1e-15  # minimum threshold for double precision


def _assert(expect, actual):
    assert abs(expect - actual) <= _EPSILON, \
        'expected value {} != {} actual value'.format(expect, actual)


def _assert_array(expect, actual):
    if np.shape(expect):
        diff = abs(expect - actual)
        diff_bool = diff < _EPSILON
        false_index = np.where(diff_bool == False)
        assert diff_bool.all(), \
            'expected value(s) {} != {} actual value'.format(expect[false_index], actual[false_index])
    else:
        _assert(expect, actual)


# Shared/fixed instantiation for tests

def make_particles(PD):
    # instantiate particle data
    ptcl_data = particle_data.particle_data(PD['num_p'], PD['charge'], PD['mass'], PD['weight'])

    # Add r data - evenly distribute out to r = R
    ptcl_data.r = np.arange(0., PD['R'], PD['R'] / PD['num_p']) + PD['R'] / PD['num_p']
    # Add pr and ell data - evenly distribute pr out to PR
    ptcl_data.pr = ptcl_data.mc * np.arange(0., PD['PR'], PD['PR'] / PD['num_p'])
    ptcl_data.ell = ptcl_data.r * ptcl_data.pr * .1

    # Add z data - evenly distribute out to z = Z
    ptcl_data.z = np.zeros(PD['num_p'])
    ptcl_data.pz = ptcl_data.mc * np.arange(0., PD['Z'], PD['Z'] / PD['num_p'])

    return ptcl_data


def make_negative_particles(PD):
    # instantiate particle data
    ptcl_data = particle_data.particle_data(PD['num_p'], PD['charge'], PD['mass'], PD['weight'])

    # Add r data - evenly distribute out to r = R
    ptcl_data.r = np.arange(0., PD['R'], PD['R'] / PD['num_p']) + PD['R'] / PD['num_p']
    # Switch 2 of the particles to make them negative
    ptcl_data.r[0] *= -1. * ptcl_data.r[0]
    ptcl_data.r[1] *= -1. * ptcl_data.r[1]

    # Add pr and ell data - evenly distribute pr out to PR
    ptcl_data.pr = ptcl_data.mc * np.arange(0., PD['PR'], PD['PR'] / PD['num_p'])
    ptcl_data.ell = ptcl_data.r * ptcl_data.pr * .1

    # Add z data - evenly distribute out to z = Z
    ptcl_data.z = np.zeros(PD['num_p'])
    ptcl_data.pz = ptcl_data.mc * np.arange(0., PD['Z'], PD['Z'] / PD['num_p'])

    return ptcl_data


def make_fields(PD):
    # Instantiate field data
    fld_data = field_data.field_data(PD['Z'], PD['R'], PD['n_r'], PD['n_z'])

    # Give mode Ps and Qs normalized values of '1'
    fld_data.mode_coords = np.ones((PD['n_r'], PD['n_z'], 2))

    return fld_data


def test_compute_ptcl_energy():
    particles = make_particles(_PD)
    fields = make_fields(_PD)

    _assert_array(_PD['expected_energy'], particles.compute_ptcl_energy(fields))


def test_r_boundaries():
    ptcls2 = make_negative_particles(_PD)
    ptcls2.r_boundaries()

    _assert_array(_PD['expected_r'], ptcls2.r)
    _assert_array(_PD['expected_pr'], ptcls2.pr)


if __name__ == '__main__':

    test_compute_ptcl_energy()

    test_r_boundaries()