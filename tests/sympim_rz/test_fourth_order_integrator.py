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
            [  5.421533983247400513e+11,   6.285883584359227295e+11],
            [  5.249662003726894531e+11,   6.385507113745643311e+11]],
           [[  4.893873313924893188e+11,   6.071467162943410645e+11],
            [  5.257707549733372192e+11,   6.177775561389274902e+11],
            [  5.221114749271081543e+11,   6.280831770292744141e+11],
            [  5.101535555403202515e+11,   6.380427727122015381e+11]]]),
    'expected_dc_coords': np.asarray([[[  1.633792374577470697e-11,   1.154177294013991874e-13],
            [  2.383150882865948361e-11,   3.845873986491952721e-13],
            [ -1.570811740008717267e-12,  -4.053648773792719040e-14],
            [  5.661284275436543286e-13,   1.927034642163540361e-14]],
           [[ -1.666880597685760737e-11,  -1.153455356905030625e-13],
            [  3.428757087144338143e-12,   5.741113471193605169e-14],
            [  7.811178185367586953e-12,   1.976575441455246682e-13],
            [ -4.229767719437571019e-13,  -1.537436027921663505e-14]]]),
    'expected_r': np.asarray([ 0.397021089571138797,  0.61019548935417478 ,  0.823923982747928885,
            1.037597833751698673,  1.25116658761149635 ,  1.464663906504860558,
            1.678115894351572379,  1.891538361503670052,  2.104940827941121029,
            2.318329221778988813,  2.53170738143010432 ,  2.745077884905913201,
            2.958442520865333769,  3.171802566029616521,  3.385158954526976238,
            3.598512384757971194]),
    'expected_z': np.asarray([ 0.399999999950103269,  0.629848301573040281,  0.850516091963725351,
            1.066690736012238849,  1.281291342579865544,  1.495277526069096474,
            1.708986240707247362,  1.922553804952052747,  2.136042644954539771,
            2.349484321533511277,  2.562896095200769775,  2.776288021522111737,
            2.989666269228197226,  3.203034788478130501,  3.416396201814406286,
            3.629752304839935828]),
    'expected_pr': np.asarray([ -5.995849160058627319e+10,  -7.594742271047164917e+10,
            -9.193635380679481506e+10,  -1.079252848938776398e+11,
            -1.239142159794477844e+11,  -1.399031470679045715e+11,
            -1.558920781601188965e+11,  -1.718810092544901428e+11,
            -1.878699403487740784e+11,  -2.038588714416622314e+11,
            -2.198478025333101807e+11,  -2.358367336248037720e+11,
            -2.518256647172018433e+11,  -2.678145958108762817e+11,
            -2.838035269054668579e+11,  -2.997924580002633667e+11]),
    'expected_pz': np.asarray([ -6.471967654854324792e-01,   3.997232773243538208e+11,
             7.994465546563562012e+11,   1.199169831993104004e+12,
             1.598893109330628662e+12,   1.998616386666680420e+12,
             2.398339664000762207e+12,   2.798062941333593750e+12,
             3.197786218666215332e+12,   3.597509495999272461e+12,
             3.997232773332787598e+12,   4.396956050666441406e+12,
             4.796679327999948242e+12,   5.196402605333282227e+12,
             5.596125882666569336e+12,   5.995849159999924805e+12]),
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

def test_fourth_order_integrator():
    '''Tests the integrator by running a single step and checking coordinates'''

    # create fields, particles, integrator
    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)

    dt = .1 / np.amax(fields.omega)
    my_integrator = integrator.integrator(dt, fields)
    
    #setup fourth order integration
    my_integrator.setup_fourth_order(fields)

    # take a single step
    my_integrator.fourth_order_step(particles, fields)
    

    #assertions
    _assert_array(_PD['expected_dc_coords'], fields.dc_coords)
    _assert_array(_PD['expected_omega_coords'], fields.omega_coords)
    _assert_array(_PD['expected_r'], particles.r)
    _assert_array(_PD['expected_pr'], particles.pr)
    _assert_array(_PD['expected_z'], particles.z)
    _assert_array(_PD['expected_pz'], particles.pz)
    _assert_array(_PD['expected_ell'], particles.ell)
    
if __name__ == '__main__':
    
    test_fourth_order_integrator()
    