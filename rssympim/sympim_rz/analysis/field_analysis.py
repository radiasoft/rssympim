"""
Standard post-processing tools for visualizing the fields from a sympim_rz simulation

Author: Stephen Webb
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

#import rcParams dictionary
mplrc = pickle.load( open( "rssympim_style.txt", "r" ) )
mpl.rcParams = mplrc


class field_analysis(object):

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


    def plot_energy_spectrum(self, fig_name, scale=False):
        """
        Contour plot of the energy in the fields versus k_r and k_z

        :param fig_name:
        :return:
        """
        plt.clf()

        P_omega = np.array(self.file.get('p_omega'))
        Q_omega = np.array(self.file.get('q_omega'))

        P_dc = np.array(self.file.get('p_dc'))

        kr = self.file.get('kr')
        kz = self.file.get('kz')

        mm = self.file.get('mode_mass')

        n_modes_r = np.shape(kr)[0]
        n_modes_z = np.shape(kz)[0]

        omega = np.zeros((n_modes_z, n_modes_r))
        for idx_r in range(0, n_modes_r):
            for idx_z in range(0, n_modes_z):
                omega[idx_z, idx_r] = \
                    np.sqrt(kr[idx_r] ** 2 + kz[idx_z] ** 2)

        P_O_sqrd = P_omega*P_omega
        Q_O_sqrd = Q_omega*Q_omega

        P_D_sqrd = P_dc*P_dc

        Energy = 0.5 * (P_O_sqrd/mm + mm*(omega * Q_O_sqrd) ** 2)
        Energy += 0.5 * P_D_sqrd/mm

        if scale:
            Energy /= scale


        Energy_plot = plt.imshow(Energy.transpose(),
                                 origin='lower',
                                 cmap=plt.cm.viridis,
                                 extent=[0, kr[-1], 0, kz[-1]],
                                 aspect=np.max(kr)/np.max(kz),
                                 interpolation='gaussian')

        plt.xlabel(r'$k_z$ [cm${}^{-1}$]')
        plt.ylabel(r'$k_r$ [cm${}^{-1}$]')
        cbar = plt.colorbar(Energy_plot)
        if scale:
            cbar.ax.set_ylabel(r'$E/E_0$')
        else:
            cbar.ax.set_ylabel(r'$E$ [ergs]')

        plt.tight_layout()
        print 'Saving figure', fig_name
        plt.savefig(fig_name)


    def plot_Ez(self, fig_name, **kwargs):
        """
        Plot the longitudinal electric field in units of statV/cm.

        :param fig_name:
        :return:
        """

        plt.clf()

        R = self.file.attrs['R']
        L = self.file.attrs['L']
        zmin = 0.

        if 'rmax' in kwargs.keys():
            R = kwargs['rmax']

        if 'zmax' in kwargs.keys():
            L = kwargs['zmax']

        if 'zmin' in kwargs.keys():
            zmin = kwargs['zmin']

        EZ, RR, LL = self.compute_Ez(zmin, L, R)

        if 'scale' in kwargs.keys():
            EZ /= kwargs['scale']

        Ez_plot = plt.imshow(EZ.transpose(),
                             cmap=plt.cm.RdBu,
                             extent=[zmin, L, 0, R],
                             origin='lower', interpolation='gaussian')

        Ez_contour = plt.contour(LL, RR, EZ, colors='k')

        #fmt = ticker.LogFormatterSciNotation()
        #fmt.create_dummy_axis()

        #plt.clabel(Ez_contour, inline=True, fontsize=9, fmt=fmt)

        plt.xlabel(r'$z$ [cm]')
        plt.ylabel(r'$r$ [cm]')
        cbar = plt.colorbar(Ez_plot)
        if 'scale' in kwargs.keys():
            cbar.ax.set_ylabel(r'$E_z/E_0$')
        else:
            cbar.ax.set_ylabel(r'$E_z$ [statV/cm]')

        plt.tight_layout()
        print 'Saving figure', fig_name
        plt.savefig(fig_name)


    def plot_Er(self, fig_name, **kwargs):
        """
        Plot the longitudinal electric field in units of statV/cm.

        :param fig_name:
        :return:
        """

        R = kwargs.pop('rmax', self.file.attrs['R'])
        L = kwargs.pop('zmax', self.file.attrs['L'])
        zmin = kwargs.pop('zmin', 0)

        ER, RR, LL = self.compute_Er(zmin, L, R)

        scale = kwargs.pop('scale', 1.)
        ER /= scale

        pltextent = [zmin, L, 0, R]

        # desired aspect is ratio of horizontal axis to vertical axis
        wantaspect = (pltextent[1] - pltextent[0]) / (pltextent[3] - pltextent[2])

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()

        Er_plot = ax.imshow(ER.transpose(),
                            cmap=plt.cm.RdBu,
                            extent=pltextent,
                            origin='lower', interpolation='bicubic', aspect=wantaspect)

        lvls = np.linspace(np.min(ER), np.max(ER), 5)

        # Er_contour = ax.contourf(LL, RR, ER,lvls,cmap=plt.cm.RdBu)

        Er_contour = ax.contour(LL, RR, ER, colors='k')  # cmap=plt.cm.RdBu)

        divider = make_axes_locatable(ax)
        # negative pad moves the color bar closer
        cax = divider.append_axes("right", size="4%", pad=-0.4)

        ax.set_xlabel(r'$z$ [cm]')
        ax.set_ylabel(r'$r$ [cm]')
        cbar = plt.colorbar(Er_plot, cax=cax)
        if 'scale' in kwargs.keys():
            cbar.ax.set_ylabel(r'$E_r/E_0$')
        else:
            cbar.ax.set_ylabel(r'$E_r$ [statV/cm]')

        ax.set_title(r'E$_r$')

        # fig.tight_layout()
        print 'Saving figure', fig_name
        fig.savefig(fig_name)


    def plot_acceleration(self, fig_name,
                          charge2mass=consts.electron_charge/consts.electron_mass):

        """
        Plot the longitudinal acceleration in units of a species rest energy/cm

        :param fig_name:
        :param charge2mass: defaults to electron
        :return:
        """

        EZ = self.compute_Ez()

        R = self.file.attrs['R']
        L = self.file.attrs['L']

        Gradient = charge2mass*EZ/consts.c**2

        Grad_plot = plt.imshow(Gradient,
                               cmap=plt.cm.RdBu,
                               extent=[0, L, 0, R])

        plt.xlabel(r'$z$ [cm]')
        plt.ylabel(r'$r$ [cm]')
        cbar = plt.colorbar(Grad_plot)
        cbar.ax.set_ylabel(r'$Gradient$ [$mc^2$/cm]')

        plt.tight_layout()
        print 'Saving figure', fig_name
        plt.savefig(fig_name)


    def compute_Ez(self, zmin, L, R):
        """
        Compute the longitudinal electric field in units of statV/cm.
        :return: EZ, a meshgrid array of the electric field
        """

        if self.file_name:

            # get the k-vectors
            kr = np.array(self.file.get('kr'))
            kz = np.array(self.file.get('kz'))
            om = np.array(self.file.get('omega'))
            mm = np.array(self.file.get('mode_mass'))

            n_modes_r = np.shape(kr)[0]
            n_modes_z = np.shape(kz)[0]

            P_dc = self.file.get('p_dc')
            P_omega = self.file.get('p_omega')

            P_dc = np.array(P_dc)
            P_omega = np.array(P_omega)

            dotQz = (P_dc + P_omega)/(mm)

            omO2kz = .5*np.einsum('zr, z -> zr', om, 1/kr)

            dotQz *= omO2kz

            R_range = np.arange(0., R, R/n_modes_r)
            Z_range = np.arange(zmin, L, (L-zmin)/n_modes_z)

            RR, ZZ = np.meshgrid(R_range, Z_range)

            kr_cross_r = einsum('k, lm -> klm', kr, RR)
            kz_cross_z = einsum('i, lm -> ilm', kz, ZZ)

            # generate a mesh grid
            the_j0 = j0(kr_cross_r)
            the_cos = cos(kz_cross_z)

            EZ = einsum('ik, klm, ilm->lm', dotQz, the_j0, the_cos)

            return EZ, RR, ZZ

        else:
            print 'File must be opened first.'


    def compute_Er(self, zmin, L, R):
        """
        Compute the radial electric field in units of statV/cm.
        :return: ER, a meshgrid array of the electric field
        """

        if self.file_name:

            # get the k-vectors
            kr = np.array(self.file.get('kr'))
            kz = np.array(self.file.get('kz'))
            om = np.array(self.file.get('omega'))
            mm = np.array(self.file.get('mode_mass'))

            n_modes_r = np.shape(kr)[0]
            n_modes_z = np.shape(kz)[0]

            P_dc = self.file.get('p_dc')
            P_omega = self.file.get('p_omega')

            P_dc = np.array(P_dc)
            P_omega = np.array(P_omega)

            dotQr = (P_dc - P_omega) / (mm)

            omO2kr = .5 * np.einsum('zr, z -> zr', om, 1 / kz)

            dotQr *= omO2kr

            R_range = np.linspace(0, R, n_modes_r)  # np.arange(0., R, R/n_modes_r)
            Z_range = np.linspace(zmin, L, n_modes_z)  # np.arange(zmin, L, (L-zmin)/n_modes_z)

            RR, ZZ = np.meshgrid(R_range, Z_range)

            kr_cross_r = np.einsum('k, lm -> klm', kr, RR)
            kz_cross_z = np.einsum('i, lm -> ilm', kz, ZZ)

            # generate a mesh grid
            the_j1 = j1(kr_cross_r)
            the_sin = np.sin(kz_cross_z)

            ER = np.einsum('ik, klm, ilm->lm', dotQr, the_j1, the_sin)

            return ER, RR, ZZ

        else:
            print 'File must be opened first.'
