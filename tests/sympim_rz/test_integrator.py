#import pytest
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
    'num_steps': 10,  # number of steps to perform
    'expected_energy': np.asarray(
        [5.995849160000000000e+11, 8.484762139977631836e+11, 1.342066067075586182e+12, 1.898207154006640137e+12,
         2.475087695233998535e+12,
         3.061004316744099121e+12, 3.651609971451465820e+12, 4.244947934725706055e+12, 4.840013448525821289e+12,
         5.436239235329887695e+12]),
    'expected_mode_coords': np.asarray(
        [[[0.974698151084738118, 1.052906006048294874],
          [0.958416822710722593, 1.052473514864393556],
          [0.931292989118252423, 1.051752874152074302],
          [0.893341931855663685, 1.050744289566363232]],

         [[0.889903363627608468, 1.05065288957338332],
          [0.873647997272205656, 1.050220775365997117],
          [0.846565669613187466, 1.049500715812273111],
          [0.808671418981096735, 1.048492910231891617]]]),
    'expected_r': np.asarray(
        [0.400000000000000022, 0.801893095897789454, 1.20239368820202186,
         1.602538564731053494, 2.002595848644080689, 2.402623708510590372,
         2.802639222838441579, 3.202648711257537251, 3.602654924455111374,
         4.002659209417975816]),
    'expected_z': np.asarray(
        [-2.051571831437011782e-17, 1.489725811009919298e-01,
         2.700955087935825727e-01, 3.841042251913643901e-01,
         4.963610804089199080e-01, 6.080294389435639824e-01,
         7.194508746700777335e-01, 8.307517836906107567e-01,
         9.419871821974556969e-01, 1.053184011575806212e+00]),
    'expected_pr': np.asarray(
        [1.322339778091870551e-21, 2.997995270522281647e+10,
         5.995968403984075165e+10, 8.993916095823971558e+10,
         1.199185365992634125e+11, 1.498978649931621704e+11,
         1.798771678688645325e+11, 2.098564554539902039e+11,
         2.398357331649448547e+11, 2.698150041322982483e+11]),
    'expected_pz': np.asarray(
        [0.000000000000000000e+00, 5.995849160000000000e+11,
         1.199169832000000000e+12, 1.798754748000000000e+12,
         2.398339664000000000e+12, 2.997924580000000000e+12,
         3.597509496000000000e+12, 4.197094412000000000e+12,
         4.796679328000000000e+12, 5.396264244000000000e+12]),
    'expected_ell': np.asarray(
        [0.000000000000000000e+00, 2.398339664000000000e+09,
         7.195018992000001907e+09, 1.439003798400000381e+10,
         2.398339664000000000e+10, 3.597509496000000000e+10,
         5.036513294400001526e+10, 6.715351059200001526e+10,
         8.634022790400000000e+10, 1.079252848800000000e+11])
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

    # Add z data - evenly distribute out to z = Z/10.
    ptcl_data.z = np.linspace(0, _PD['Z'] / 10., _PD['num_p']) #ptcl_data.z = np.zeros(PD['num_p'])
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

def test_integrator():
    '''Tests the integrator by running a single step and checking coordinates'''

    # create fields, particles, integrator
    particles = make_particles(_PD)
    fields = make_fields(_PD)

    dt = .1 / np.amax(fields.omega)
    my_integrator = integrator.integrator(dt, fields.omega)

    # take a single step
    my_integrator.single_step(particles, fields)

    #assertions
    _assert_array(_PD['expected_mode_coords'], fields.mode_coords)
    _assert_array(_PD['expected_r'], particles.r)
    _assert_array(_PD['expected_pr'], particles.pr)
    _assert_array(_PD['expected_z'], particles.z)
    _assert_array(_PD['expected_pz'], particles.pz)
    _assert_array(_PD['expected_ell'], particles.ell)