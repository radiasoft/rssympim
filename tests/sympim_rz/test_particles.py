import numpy as np
from rssympim.sympim_rz.data import particle_data, field_data
from rssympim.constants import constants as consts

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
    'expected_energy' : np.asarray([  1.806475552056596759e+22,   2.172303640162937183e+22,
             3.008502320019848298e+22,   4.032357031801913973e+22,
             5.132772561668123971e+22,   6.269564500178906592e+22,
             7.426045905350004677e+22,   8.594271956961059878e+22,
             9.770030556152852801e+22,   1.095089574336511271e+23,
             1.213537687964598424e+23,   1.332250953666006299e+23,
             1.451164300181204275e+23,   1.570232271941271175e+23,
             1.689422175990533272e+23,   1.808709907302464126e+23]),
    'expected_r':       np.asarray([ 0.400000000000000022,  0.613333333333333397,  0.826666666666666661,
            1.040000000000000036,  1.25333333333333341 ,  1.466666666666666785,
            1.68000000000000016 ,  1.893333333333333535,  2.106666666666666909,
            2.320000000000000284,  2.533333333333333215,  2.74666666666666659 ,
            2.959999999999999964,  3.173333333333333339,  3.386666666666666714,
            3.600000000000000089]),
    'expected_pr':      np.asarray([  5.995849160000000000e+10,   7.594742269333334351e+10,
            -9.193635378666667175e+10,  -1.079252848800000000e+11,
            -1.239142159733333282e+11,  -1.399031470666666565e+11,
            -1.558920781600000000e+11,  -1.718810092533333435e+11,
            -1.878699403466666870e+11,  -2.038588714400000000e+11,
            -2.198478025333333435e+11,  -2.358367336266666565e+11,
            -2.518256647200000305e+11,  -2.678145958133333130e+11,
            -2.838035269066666870e+11,  -2.997924580000000000e+11])}

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


def test_compute_ptcl_energy():
    particles = make_particles(_PD)
    fields = make_fields(_PD,particles)
    
    _assert_array(_PD['expected_energy'],particles.compute_ptcl_energy(fields))

def test_r_boundaries():
    
    particles = make_negative_particles(_PD)
    fields = make_fields(_PD,particles)
    particles.r_boundaries(fields)
    
    _assert_array(_PD['expected_r'],particles.r)
    _assert_array(_PD['expected_pr'],particles.pr)


if __name__ == '__main__':

    test_compute_ptcl_energy()

    test_r_boundaries()