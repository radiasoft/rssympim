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
    'num_steps': 1000,  # number of steps to perform
    'expected_energy': np.asarray(
        [5.995849160000000000e+11, 8.484762139977631836e+11, 1.342066067075586182e+12, 1.898207154006640137e+12,
         2.475087695233998535e+12,
         3.061004316744099121e+12, 3.651609971451465820e+12, 4.244947934725706055e+12, 4.840013448525821289e+12,
         5.436239235329887695e+12]),
    'expected_dFzdQ' : np.asarray(
        [[  5.329416563916345871e-05,   3.844692476643437939e-05, 2.620639507592710505e-05, 1.740602316366593406e-05],
        [ -3.652397544320205760e-05,  -3.100019271727580731e-05, -2.393778051862682007e-05, -1.698907307145165993e-05]]),
    'expected_dFrdQ' : np.asarray(
        [[-1.455232005651519795e-05, -4.199273580361816279e-05, -6.440244625068591042e-05, -7.604525319451381146e-05],
         [1.892811316913410978e-06, 6.426191551136591621e-06, 1.116492144841025481e-05, 1.408701408137552498e-05]]),
    'expected_dFrdz': np.asarray(
        [-1.634565383223467974e-04, -1.469902207581869590e-04,
         -1.217555489162023299e-04, -9.246517464383130090e-05,
         -6.394692807495849156e-05, -3.991124084526193120e-05,
         -2.219924965309100481e-05, -1.071404065372966999e-05,
         -3.978700000916946748e-06, -4.266681380585164543e-08]),
    'expected_dFzdr': np.asarray(
        [0.000000000000000000e+00, -1.250825012270450639e-05,
         -3.092916244273120392e-05, -4.615007437337265785e-05,
         -5.020121667429228721e-05, -3.948455354248335168e-05,
         -1.634099860196912366e-05, 1.164966440360454962e-05,
         3.439278939922958299e-05, 4.302761857764898930e-05]),
    'expected_Ar': np.asarray(
        [0.000000000000000000e+00, 5.495386932676998248e-06,
         1.413406409585032007e-05, 2.260820595018311709e-05,
         2.792837342995887877e-05, 2.853538855527951461e-05,
         2.482122201111121114e-05, 1.886864827303885017e-05,
         1.352934136888968225e-05, 1.121780054081415307e-05]),
    'expected_Az': np.asarray(
        [2.237535013307200850e-04, 1.873417125997809893e-04,
        1.346916299964593397e-04, 7.730651741949247325e-05,
        2.670407297490108578e-05, -8.660715772379087876e-06,
        -2.535831924795489497e-05, -2.529155014029687255e-05,
        -1.437949252365613905e-05, -1.836517111904149318e-07]),

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


#Analytic Computations
def compute_energy_analytic(fld_data):
    '''Compute field energy'''
    Qsqrd = fld_data.mode_coords[:, :, 1] * fld_data.mode_coords[:, :, 1]
    Psqrd = fld_data.mode_coords[:, :, 0] * fld_data.mode_coords[:, :, 0]

    return 0.5 * (Psqrd + fld_data.omega * fld_data.omega * Qsqrd)

#Test functions
def test_field_energy():

    fields = make_fields(_PD)
    field_energy_analytic = compute_energy_analytic(fields)

    _assert_array(field_energy_analytic,fields.compute_energy())

def test_compute_Ar():
    particles = make_particles(_PD)
    fields = make_fields(_PD)

    _assert_array(_PD['expected_Ar'], fields.compute_Ar(particles.r, particles.z, particles.qOc))

def test_compute_Az():
    particles = make_particles(_PD)
    fields = make_fields(_PD)

    _assert_array(_PD['expected_Az'], fields.compute_Az(particles.r, particles.z, particles.qOc))

def test_compute_dFrdz():
    particles = make_particles(_PD)
    fields = make_fields(_PD)

    _assert_array(_PD['expected_dFrdz'], fields.compute_dFrdz(particles.r, particles.z, particles.qOc))

def test_compute_dFzdr():
    particles = make_particles(_PD)
    fields = make_fields(_PD)

    _assert_array(_PD['expected_dFzdr'], fields.compute_dFzdr(particles.r, particles.z, particles.qOc))

def test_compute_dFzdQ():
    particles = make_particles(_PD)
    fields = make_fields(_PD)

    _assert_array(_PD['expected_dFzdQ'], fields.compute_dFzdQ(particles.r, particles.z, particles.qOc))

def test_compute_dFrdQ():
    particles = make_particles(_PD)
    fields = make_fields(_PD)

    _assert_array(_PD['expected_dFrdQ'], fields.compute_dFrdQ(particles.r, particles.z, particles.qOc))