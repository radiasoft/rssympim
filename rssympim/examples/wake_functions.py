from rssympim.sympim_rz.data import particle_data, field_data

from beam_integrator import beam_integrator

from rssympim.sympim_rz.boundaries import radial_thermal, \
    radial_reflecting, longitudinal_absorb

from rssympim.constants import constants as consts

import numpy as np

from rssympim.sympim_rz.io import field_io, particle_io

from rssympim.sympim_rz.analysis import field_analysis

from mpi4py import MPI as mpi

import time
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

r_beam = .01 # cm
q_beam = -consts.electron_charge # single ring electron

domain_r = 3 * r_beam
domain_l = 4 * r_beam
