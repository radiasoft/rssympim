"""
Standard post-processing tools for visualizing the fields from a sympim_rz simulation

Author: Stephen Webb
"""

import h5py
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from rssympim.constants import constants as consts
import numpy as np
from numpy import einsum, cos, sin
from scipy.special import j0, j1, jn_zeros


class field_analysis:

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


    def plot_Ez(self, fig_name, scale=False):
        """
        Plot the longitudinal electric field in units of statV/cm.

        :param fig_name:
        :return:
        """

        plt.clf()

        EZ, RR, LL = self.compute_Ez()

        R = self.file.attrs['R']
        L = self.file.attrs['L']

        if scale:
            EZ /= scale

        Ez_plot = plt.imshow(EZ.transpose(),
                             cmap=plt.cm.RdBu,
                             extent=[0, L, 0, R],
                             origin='lower', interpolation='gaussian')

        Ez_contour = plt.contour(LL, RR, EZ, colors='k')

        #fmt = ticker.LogFormatterSciNotation()
        #fmt.create_dummy_axis()

        #plt.clabel(Ez_contour, inline=True, fontsize=9, fmt=fmt)

        plt.xlabel(r'$z$ [cm]')
        plt.ylabel(r'$r$ [cm]')
        cbar = plt.colorbar(Ez_plot)
        if scale:
            cbar.ax.set_ylabel(r'$E_z/E_0$')
        else:
            cbar.ax.set_ylabel(r'$E_z$ [statV/cm]')

        plt.tight_layout()
        print 'Saving figure', fig_name
        plt.savefig(fig_name)


    def plot_Er(self, fig_name, scale=False):
        """
        Plot the longitudinal electric field in units of statV/cm.

        :param fig_name:
        :return:
        """

        plt.clf()

        ER, RR, LL = self.compute_Er()

        R = self.file.attrs['R']
        L = self.file.attrs['L']

        if scale:
            ER /= scale

        Er_plot = plt.imshow(ER.transpose(),
                             cmap=plt.cm.RdBu,
                             extent=[0, L, 0, R],
                             origin='lower', interpolation='gaussian')

        Er_contour = plt.contour(LL, RR, ER, colors='k')

        #fmt = ticker.LogFormatterSciNotation()
        #fmt.create_dummy_axis()

        #plt.clabel(Er_contour, inline=True, fontsize=9, fmt=fmt)

        plt.xlabel(r'$z$ [cm]')
        plt.ylabel(r'$r$ [cm]')
        cbar = plt.colorbar(Er_plot)
        if scale:
            cbar.ax.set_ylabel(r'$E_r/E_0$')
        else:
            cbar.ax.set_ylabel(r'$E_r$ [statV/cm]')

        plt.tight_layout()
        print 'Saving figure', fig_name
        plt.savefig(fig_name)


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


    def compute_Ez(self):
        """
        Compute the longitudinal electric field in units of statV/cm.
        :return: EZ, a meshgrid array of the electric field
        """

        if self.file_name:

            # get the domain length and radius
            R = self.file.attrs['R']
            L = self.file.attrs['L']

            # get the k-vectors
            kr = np.array(self.file.get('kr'))
            kz = np.array(self.file.get('kz'))
            om = np.array(self.file.get('omega'))
            mm = np.array(self.file.get('mode_mass'))

            n_modes_r = np.shape(kr)[0]
            n_modes_z = np.shape(kz)[0]

            P_dc = self.file.get('p_dc')
            P_omega = self.file.get('p_omega')

            kzpdc = np.einsum('z, zr -> zr', kz, P_dc)
            krpom = np.einsum('r, zr -> zr', kr, P_omega)

            P_z = (kzpdc + krpom)/om

            R_range = np.arange(0., R, R/n_modes_r)
            Z_range = np.arange(0., L, L/n_modes_z)

            RR, ZZ = np.meshgrid(R_range, Z_range)

            kr_cross_r = einsum('k, lm -> klm', kr, RR)
            kz_cross_z = einsum('i, lm -> ilm', kz, ZZ)

            # generate a mesh grid
            the_j0 = j0(kr_cross_r)
            the_cos = cos(kz_cross_z)

            EZ = einsum('ik, klm, ilm, ik->lm', P_z, the_j0, the_cos, mm)

            return EZ, RR, ZZ


        else:
            print 'File must be opened first.'

    def compute_Er(self):
        """
        Compute the radial electric field in units of statV/cm.
        :return: ER, a meshgrid array of the electric field
        """

        if self.file_name:

            # get the domain length and radius
            # get the domain length and radius
            R = self.file.attrs['R']
            L = self.file.attrs['L']

            # get the k-vectors
            kr = np.array(self.file.get('kr'))
            kz = np.array(self.file.get('kz'))
            om = np.array(self.file.get('omega'))
            mm = np.array(self.file.get('mode_mass'))

            n_modes_r = np.shape(kr)[0]
            n_modes_z = np.shape(kz)[0]

            P_dc = self.file.get('p_dc')
            P_omega = self.file.get('p_omega')

            krpdc = np.einsum('r, zr -> zr', kr, P_dc)
            kzpom = np.einsum('z, zr -> zr', kz, P_omega)

            P_r = (kzpom - krpdc)/om

            R_range = np.arange(0., R, R/n_modes_r)
            Z_range = np.arange(0., L, L/n_modes_z)

            RR, ZZ = np.meshgrid(R_range, Z_range)

            kr_cross_r = einsum('i, lm -> ilm', kr, RR)
            kz_cross_z = einsum('k, lm -> klm', kz, ZZ)

            # generate a mesh grid
            the_j1 = j1(kr_cross_r)
            the_sin = sin(kz_cross_z)

            ER = einsum('ik, klm, ilm, ik->lm', P_r, the_j1, the_sin, mm)

            return ER, RR, ZZ


        else:
            print 'File must be opened first.'