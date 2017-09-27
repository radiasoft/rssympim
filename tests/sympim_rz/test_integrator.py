import numpy as np
from rssympim.constants import constants as consts
from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.sympim_rz.integrators import integrator

n_r = 2
n_z = 4
num_p = 2
n_macro = n_r*n_z*num_p

#dictionary for fixed values
_PD = {
    'num_p'     : num_p,
    'n_r'       : n_r, #number of radial modes
    'n_z'       : n_z, #number of longitudinal modes
    'n_macro'   : n_macro,
    'charge'    : 1, #1 esu
    'mass'      : 1., #mass in m_e
    'weight'    : 20., #macroparticle weighting - arbitrary
    'R'         : 4., #maximum R value
    'PR'        : 0.5, #maximum PR value
    'Z'         : 4., #maximum z value
    'num_steps' : 1000, #number of steps to perform
    'expected_omega_coords': np.asarray([[[  5.614648573878920898e+11,   6.076461066345686035e+11],
            [  5.570036761301547852e+11,   6.182798863972700195e+11],
            [  5.421533983247402954e+11,   6.285883584359228516e+11],
            [  5.249662003726893921e+11,   6.385507113745642090e+11]],
           [[  4.893873313924891968e+11,   6.071467162943411865e+11],
            [  5.257707549733371582e+11,   6.177775561389276123e+11],
            [  5.221114749271083374e+11,   6.280831770292746582e+11],
            [  5.101535555403202515e+11,   6.380427727122020264e+11]]]),
    'expected_dc_coords': np.asarray([[[  1.633792374576850311e-11,   1.134262264128571455e-13],
            [  2.383150882865704731e-11,   3.851373366333830980e-13],
            [ -1.570811740009434588e-12,  -3.988735473740438959e-14],
            [  5.661284275435848583e-13,   1.960276234039529921e-14]],
           [[ -1.666880597685915834e-11,  -1.157233801665218809e-13],
            [  3.428757087143666059e-12,   5.541161417851206039e-14],
            [  7.811178185367325228e-12,   1.983479160876082209e-13],
            [ -4.229767719437667954e-13,  -1.464599326322099932e-14]]]),
    'expected_r': np.asarray([ 0.397021089571137575,  0.610195489354171339,  0.823923982747926997,
            1.037597833751698007,  1.25116658761149635 ,  1.464663906504860558,
            1.678115894351572379,  1.891538361503670274,  2.104940827941121029,
            2.318329221778988813,  2.531707381430103876,  2.745077884905913645,
            2.958442520865333769,  3.171802566029616521,  3.385158954526976238,
            3.598512384757971194]),
    'expected_z': np.asarray([ 0.399999999950094387,  0.629848301573035285,  0.850516091963723797,
            1.066690736012238183,  1.281291342579865544,  1.495277526069096474,
            1.708986240707247362,  1.922553804952052747,  2.136042644954539327,
            2.349484321533511277,  2.562896095200769775,  2.776288021522111293,
            2.98966626922819767 ,  3.203034788478130501,  3.41639620181440673 ,
            3.629752304839935828]),
    'expected_pr': np.array([ -5.995849160058638000e+10,  -7.594742271047711182e+10,
            -9.193635380680191040e+10,  -1.079252848938829346e+11,
            -1.239142159794502716e+11,  -1.399031470679048767e+11,
            -1.558920781601186829e+11,  -1.718810092544902344e+11,
            -1.878699403487746277e+11,  -2.038588714416626892e+11,
            -2.198478025333099976e+11,  -2.358367336248030090e+11,
            -2.518256647172007751e+11,  -2.678145958108753967e+11,
            -2.838035269054664307e+11,  -2.997924580002636719e+11]),
    'expected_pz': np.asarray([ -6.473512613591765330e-01,   3.997232773243527832e+11,
             7.994465546563554688e+11,   1.199169831993104004e+12,
             1.598893109330630127e+12,   1.998616386666681396e+12,
             2.398339664000764160e+12,   2.798062941333593750e+12,
             3.197786218666216309e+12,   3.597509495999271484e+12,
             3.997232773332787598e+12,   4.396956050666440918e+12,
             4.796679327999949219e+12,   5.196402605333282227e+12,
             5.596125882666570312e+12,   5.995849159999923828e+12]),
    'expected_ell': np.asarray([  2.184738736844831995e-16,   3.349932729828742820e-16,
             4.515126722812652659e-16,   5.680320715796562990e-16,
             6.845514708780474308e-16,   8.010708701764384640e-16,
             9.175902694748295958e-16,   1.034109668773220530e-15,
             1.150629068071611662e-15,   1.267148467370002794e-15,
             1.383667866668393531e-15,   1.500187265966784663e-15,
             1.616706665265175598e-15,   1.733226064563566729e-15,
             1.849745463861957861e-15,   1.966264863160348796e-15])
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
    '''Create test particle array'''
    
    particles = particle_data.particle_data(_PD['n_macro'], _PD['charge'], _PD['mass'], _PD['weight'])

    #Add r data - better to use linspace than arange
    particles.r = np.linspace(0.1*PD['R'],0.9*PD['R'],particles.np)
    particles.pr = -particles.mc * np.linspace(0.1, .5,particles.np)
    
    
    particles.ell = particles.weight*consts.electron_mass*consts.c*particles.r
    
    #Add z data 
    particles.z = np.linspace(0.1*PD['Z'],0.9*PD['Z'],particles.np) 
    particles.pz = particles.mc * np.linspace(0., 10., particles.np)

    return particles

def make_negative_particles(PD):
    '''Create test particle array'''
    
    particles = particle_data.particle_data(_PD['n_macro'], _PD['charge'], _PD['mass'], _PD['weight'])

    #Add r data - better to use linspace than arange
    particles.r = np.linspace(0.1*PD['R'],0.9*PD['R'],particles.np)
    particles.pr = -particles.mc * np.linspace(0.1, .5, particles.np)
    
    #Flip the r coordinates for two particles
    particles.r[0] *= -1. #* particles.r[0]
    particles.r[1] *= -1. #* particles.r[1]
    
    particles.ell = particles.weight*consts.electron_mass*consts.c*particles.r
    
    #Add z data 
    particles.z = np.linspace(0.1*PD['Z'],0.9*PD['Z'],particles.np) 
    particles.pz = particles.mc * np.linspace(0., 10., particles.np)

    return particles

def make_fields(PD, particles):
    '''Create test field array'''
    
    #Instantiate field data
    fields = field_data.field_data(_PD['R'], _PD['Z'], _PD['n_r'], _PD['n_z'])

    #Give mode Ps and Qs normalized values of '1'
    fields.omega_coords = particles.mc[0] * np.ones((fields.n_modes_z, fields.n_modes_r, 2))
    fields.dc_coords = np.zeros((fields.n_modes_z, fields.n_modes_r, 2))
    
    return fields
    
    
#Now define the integrator test function

def test_integrator():
    '''Tests the integrator by running a single step and checking coordinates'''

    # create fields, particles, integrator
    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)

    dt = .1 / np.amax(fields.omega)
    my_integrator = integrator.integrator(dt, fields)

    # take a single step
    my_integrator.second_order_step(particles, fields)

    #assertions
    _assert_array(_PD['expected_dc_coords'], fields.dc_coords)
    _assert_array(_PD['expected_omega_coords'], fields.omega_coords)
    _assert_array(_PD['expected_r'], particles.r)
    _assert_array(_PD['expected_pr'], particles.pr)
    _assert_array(_PD['expected_z'], particles.z)
    _assert_array(_PD['expected_pz'], particles.pz)
    _assert_array(_PD['expected_ell'], particles.ell)
    
if __name__ == '__main__':
    
    test_integrator()
    