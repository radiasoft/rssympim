import numpy as np
from rssympim.constants import constants as consts
from rssympim.sympim_rz.data import particle_data, field_data

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
    'expected_energy' : np.asarray([[  4.625212775245707624e+23,   6.138440471010669683e+23,
              8.662349802019289186e+23,   1.139004983804668471e+24],
           [  1.179727312613656979e+24,   9.219474330019222614e+23,
              1.062322275983559041e+24,   1.282804770815495367e+24]]),
    'expected_Ar'     : np.asarray([  93.657294171356824108,  183.908705043838807569,
            258.67726602329923935 ,  294.394449488386214853,
            287.12180035045224713 ,  249.793939801598384065,
            201.724670373918996802,  157.79966565603632489 ,
            123.457511366052685275,   96.650947851896020779,
             73.343134553757622029,   51.798868010239210946,
             33.177932806068049842,   19.35332417386609194 ,
             10.688075055105223043,    5.705724598592614605]),
    'expected_Az'     : np.asarray([  9.825529909572915130e+02,   6.783311253145758428e+02,
             3.855414149038100504e+02,   1.714224118115019735e+02,
             5.637657191984188643e+01,   1.795717305465579017e+01,
             1.451697746690572721e+01,   1.174995444873148109e+01,
            -2.943630212682703640e+00,  -2.290414913244425676e+01,
            -3.504626295358668386e+01,  -3.213458852787961462e+01,
            -1.707074638529660149e+01,   9.140496165928900796e-01,
             1.302390085564568878e+01,   1.533615366918409428e+01]),
    'expected_Sr_kick': (np.asarray([-745.044804574503586991, -571.245892581239786523,
            -378.279832105495017913, -198.738193436852554896,
             -53.081892967634992431,   51.77448465766065766 ,
             118.359734065094613698,  152.945519871103130072,
             161.818609049729730032,  150.635477594126825807,
             125.484134200379315871,   93.486812278270022603,
              61.855183600559477952,   36.015893155238742906,
              18.275066084530955379,    7.951443590266221051]),
             np.asarray([  93.657294171356824108,  183.908705043838807569,
             258.67726602329923935 ,  294.394449488386214853,
             287.12180035045224713 ,  249.793939801598384065,
             201.724670373918996802,  157.79966565603632489 ,
             123.457511366052685275,   96.650947851896020779,
              73.343134553757622029,   51.798868010239210946,
              33.177932806068049842,   19.35332417386609194 ,
              10.688075055105223043,    5.705724598592614605]),
             np.asarray([[  3.677962513261919949e-09,   2.710302874619112486e-11,
              -3.022304037789447444e-11,  -1.058121382756226716e-11],
            [  3.522419032428696208e-10,   4.462853683091069316e-10,
              -9.043714527390094184e-13,  -1.401871831985094418e-11]]),
             np.asarray([[ -4.804780943410134459e-09,  -1.542490419481595312e-11,
               1.097197445723728701e-11,   2.819129588151482774e-12],
            [ -9.203167123478138311e-10,  -5.079807953702254274e-10,
               6.566341675250257705e-13,   7.469952738410525580e-12]])),
    'expected_Sz_kick': (np.asarray([  9.825529909572915130e+02,   6.783311253145758428e+02,
              3.855414149038100504e+02,   1.714224118115019735e+02,
              5.637657191984188643e+01,   1.795717305465579017e+01,
              1.451697746690572721e+01,   1.174995444873148109e+01,
             -2.943630212682703640e+00,  -2.290414913244425676e+01,
             -3.504626295358668386e+01,  -3.213458852787961462e+01,
             -1.707074638529660149e+01,   9.140496165928900796e-01,
              1.302390085564568878e+01,   1.533615366918409428e+01]),
             np.asarray([-372.754861152164210125, -699.492165403010517366,
            -907.981863699842620008, -898.067900406311196093,
            -678.553511054629552746, -351.332565080355777809,
             -50.07352838449535426 ,  127.83339203155439634 ,
             159.130333676922077757,   88.141968083960634317,
             -11.446899066791315747,  -79.309376179372179649,
             -92.566500763065889146,  -64.128718908427160272,
             -23.793728294802509282,    3.03553778105784744 ]),
             np.asarray([[  3.677962513261919122e-09,   2.710302874619121210e-11,
              -3.022304037789444859e-11,  -1.058121382756228978e-11],
            [  3.522419032428697242e-10,   4.462853683091068282e-10,
              -9.043714527390311278e-13,  -1.401871831985093933e-11]]),
             np.asarray([[  2.815405823550203656e-09,   4.762260808490095780e-11,
              -8.325139410813381912e-11,  -3.971512573780237388e-11],
            [  1.348170219397992743e-10,   3.920829916840381766e-10,
              -1.245575946210740659e-12,  -2.630866221158199306e-11]]))}

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


#Analytic Computations
#def compute_energy_analytic(fld_data):
#'''Compute field energy'''
#Qsqrd = fld_data.mode_coords[:, :, 1] * fld_data.mode_coords[:, :, 1]
#Psqrd = fld_data.mode_coords[:, :, 0] * fld_data.mode_coords[:, :, 0]

    #return 0.5 * (Psqrd + fld_data.omega * fld_data.omega * Qsqrd)

#Test functions
def test_field_energy():

    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)

    _assert_array(_PD['expected_energy'],fields.compute_energy())

def test_compute_Ar():
    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)
    
    _assert_array(_PD['expected_Ar'], fields.compute_Ar(particles.r, particles.z, particles.qOc))

def test_compute_Az():
    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)

    _assert_array(_PD['expected_Az'], fields.compute_Az(particles.r, particles.z, particles.qOc))

###############
#S_r_kick tests
###############
    
def test_compute_Sr_kickz():
    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)
    
    kick_z, kick_r, kick_Q0, kick_Qomega = fields.compute_S_r_kick(particles.r, particles.z, particles.qOc)

    _assert_array(_PD['expected_Sr_kick'][0], kick_z)    

def test_compute_Sr_kickr():
    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)
    
    kick_z, kick_r, kick_Q0, kick_Qomega = fields.compute_S_r_kick(particles.r, particles.z, particles.qOc)

    _assert_array(_PD['expected_Sr_kick'][1], kick_r)    
    
def test_compute_Sr_kickQ0():
    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)
    
    kick_z, kick_r, kick_Q0, kick_Qomega = fields.compute_S_r_kick(particles.r, particles.z, particles.qOc)

    _assert_array(_PD['expected_Sr_kick'][2], kick_Q0)    

def test_compute_Sr_kickQomega():
    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)
    
    kick_z, kick_r, kick_Q0, kick_Qomega = fields.compute_S_r_kick(particles.r, particles.z, particles.qOc)

    _assert_array(_PD['expected_Sr_kick'][3], kick_Qomega)
    
###############
#S_z_kick tests
###############
def test_compute_Sz_kickz():
    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)
    
    kick_z, kick_r, kick_Q0, kick_Qomega = fields.compute_S_z_kick(particles.r, particles.z, particles.qOc)

    _assert_array(_PD['expected_Sz_kick'][0], kick_z)    

def test_compute_Sz_kickr():
    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)
    
    kick_z, kick_r, kick_Q0, kick_Qomega = fields.compute_S_z_kick(particles.r, particles.z, particles.qOc)

    _assert_array(_PD['expected_Sz_kick'][1], kick_r)    
    
def test_compute_Sz_kickQ0():
    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)
    
    kick_z, kick_r, kick_Q0, kick_Qomega = fields.compute_S_z_kick(particles.r, particles.z, particles.qOc)

    _assert_array(_PD['expected_Sz_kick'][2], kick_Q0)    

def test_compute_Sz_kickQomega():
    particles = make_particles(_PD)
    fields = make_fields(_PD, particles)
    
    kick_z, kick_r, kick_Q0, kick_Qomega = fields.compute_S_z_kick(particles.r, particles.z, particles.qOc)

    _assert_array(_PD['expected_Sz_kick'][3], kick_Qomega)
    
if __name__ == '__main__':

    test_field_energy()

    test_compute_Ar()
    test_compute_Az()
    
    test_compute_Sr_kickz()
    test_compute_Sr_kickr()
    test_compute_Sr_kickQ0()
    test_compute_Sr_kickQomega()
    
    test_compute_Sz_kickz()
    test_compute_Sz_kickr()
    test_compute_Sz_kickQ0()
    test_compute_Sz_kickQomega()
             
    