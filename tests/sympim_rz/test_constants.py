import numpy as np
from rssympim.constants import constants as consts

# dictionary for fixed values
_PD = {
    'c': 29979245800,
    'q': 4.80325e-10,
    'me': 9.10938e-28,
    'mp': 1.676219e-24
}

_EPSILON = 1e-6  # constants aren't maintained below this accuracy


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


def test_vals():

    _assert(consts.c,_PD['c'])
    _assert(consts.electron_charge, _PD['q'])
    _assert(consts.electron_mass, _PD['me'])
    _assert(consts.proton_mass, _PD['mp'])