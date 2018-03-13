"""
Standard post-processing tools for visualizing the particles from a sympim_rz simulation

Stephen Webb, Nathan Cook

"""

import h5py
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from rssympim.constants import constants as consts
import numpy as np
from numpy import einsum, cos, sin
from scipy.special import j0, j1, jn_zeros
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cPickle as pickle

#load matplotlib rc settings from file
mplrc = pickle.load( open( "rssympim_style.txt", "r" ) )
mpl.rcParams = mplrc

class ptcl_analysis(object):
    def __init__(self):
        self.file_name = False

    def open_file(self, file_name):
        self.file = h5py.File(file_name, 'r')
        self.file_name = file_name

    def file_name(self):
        return self.file_name

    def close_file(self):
        self.file.close()
        self.file_name = False

    def get_particle_data(self, file_name):

        self.open_file(file_name)

        self.mc = self.file.attrs['mc']

        self.r = self.file.get('r')
        self.z = self.file.get('z')
        self.wgt = self.file.get('weight')

        self.pr = self.file.get('r')
        self.pz = self.file.get('z')
        self.pl = self.file.get('pl')

        self.close_file()

    def get_field_quantities(self, file_name):
        '''
        The field data object contains the effective widths for the particles.
        This function imports field data such that these values can be obtained.
        '''

        self.open_file(file_name)
        #self.fieldfile = h5py.File(file_name, 'r')
        self.ptcl_width_r = self.file.attrs['dr']
        self.ptcl_width_z = self.file.attrs['dz']
        self.radius = self.file.attrs['R']
        self.length = self.file.attrs['L']
        self.close_file()

    def plot_particles(self, fig_name, scale=False):
        """
        Plot of particle distribution in r-z space

        :param fig_name:
        :return:
        """

        # pd is the particle weight divided by the volume of a cell -e.g. e-/cm^3
        pd = self.wgt / (2. * np.pi * self.ptcl_width_r * self.ptcl_width_z * self.r)

        H, zbins, rbins = np.histogram2d(self.z, self.r, weights=pd, bins=(20, 20))

        #with mpl.style.context('rs_paper'):
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title('Plasma density color plot')

        # Need to transpose the data for correct plotting
        dplt = ax.imshow(H.T, interpolation='nearest', origin='low',
                         extent=[zbins[0], zbins[-1], rbins[0], rbins[-1]])

        cbar = fig.colorbar(dplt)
        cbar.ax.set_xlabel(r"n [cm$^{-3}$]")
        cbar.ax.xaxis.set_label_position('bottom')

    def plot_r_phase(self, fig_name, scale=False):
        """
        Plot of distribution of r-pr

        :param fig_name:
        :return:
        """

        #with mpl.style.context('rs_paper')

        fig = plt.figure()
        ax = fig.gca()

        ax.scatter(self.r, self.pr / self.mc, s=2)

        ax.set_xlabel('r')
        ax.set_ylabel(r'p$_r/mc$')

        ax.set_title(R'p$_r$ distribution')
        ax.set_xlim([0, self.radius])
        ax.set_ylim([0, 0.1 * max(self.pz / self.mc)])

    def plot_z_phase(self, fig_name, scale=False):
        """
        Plot of distribution of z-pz

        :param fig_name:
        :return:
        """

        #with mpl.style.context('rs_paper')

        fig = plt.figure()
        ax = fig.gca()

        ax.scatter(self.z, self.pz / self.mc, s=2)

        ax.set_xlabel('r')
        ax.set_ylabel(r'p$_z/mc$')

        ax.set_title(R'p$_z$ distribution')
        ax.set_xlim([0, self.length])
        ax.set_ylim([0, 0.1 * max(self.pz / self.mc)])
