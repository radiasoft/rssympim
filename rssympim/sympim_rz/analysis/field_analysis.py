"""
Standard post-processing tools for visualizing the fields from a sympim_rz simulation

Author: Stephen Webb
"""

import h5py
from matplotlib import pyplot as plt
from rssympim.constants import constants as consts
import numpy as np
from numpy import einsum, cos
from scipy.special import j0


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


    def plot_energy_spectrum(self, fig_name):
        """
        Contour plot of the energy in the fields versus k_r and k_z

        :param fig_name:
        :return:
        """

        P = self.file.get('mode_p')
        Q = self.file.get('mode_q')

        kr = self.file.get('kr')
        kz = self.file.get('kz')

        n_modes_r = np.shape(kr)[0]
        n_modes_z = np.shape(kz)[0]

        omega = np.zeros((n_modes_r, n_modes_z))
        for idx_r in range(0, n_modes_r):
            for idx_z in range(0, n_modes_z):
                omega[idx_r, idx_z] = \
                    np.sqrt(kr[idx_r] ** 2 + kz[idx_z] ** 2)

        Psqrd = P*P
        Qsqrd = Q*Q

        Energy = 0.5*(Psqrd + (omega*Qsqrd)**2)

        Energy_plot = plt.imshow(Energy, cmap=plt.cm.viridis, extent=[0, kr[-1], 0, kz[-1]])

        plt.xlabel(r'$k_z$ [cm${}^{-1}$]')
        plt.ylabel(r'$k_r$ [cm${}^{-1}$]')
        cbar = plt.colorbar(Energy_plot)
        cbar.ax.set_ylabel(r'$E$ [ergs]')

        plt.tight_layout()

        plt.savefig(fig_name)


    def plot_Ez(self, fig_name):
        """
        Plot the longitudinal electric field in units of statV/cm.

        :param fig_name:
        :return:
        """

        EZ = self.compute_Ez()

        R = self.file.attrs['R']
        L = self.file.attrs['L']

        Ez_plot = plt.imshow(EZ, cmap=plt.cm.RdBu, extent=[0, L, 0, R])

        plt.xlabel(r'$z$ [cm]')
        plt.ylabel(r'$r$ [cm]')
        cbar = plt.colorbar(Ez_plot)
        cbar.ax.set_ylabel(r'$E_z$ [statV/cm]')

        plt.tight_layout()

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

        Grad_plot = plt.imshow(Gradient, cmap=plt.cm.RdBu, extent=[0, L, 0, R])

        plt.xlabel(r'$z$ [cm]')
        plt.ylabel(r'$r$ [cm]')
        cbar = plt.colorbar(Grad_plot)
        cbar.ax.set_ylabel(r'$Gradient$ [$mc^2$/cm]')

        plt.tight_layout()

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
            kr = self.file.get('kr')
            kz = self.file.get('kz')

            n_modes_r = np.shape(kr)[0]
            n_modes_z = np.shape(kz)[0]

            mode_P = self.file.get('mode_p')

            R_range = np.arange(0., R, R/n_modes_r)
            Z_range = np.arange(0., L, L/n_modes_z)

            RR, ZZ = np.meshgrid(R_range, Z_range)

            kr_cross_r = einsum('i, lm -> ilm', kr, RR)
            kz_cross_z = einsum('k, lm -> klm', kz, ZZ)

            # generate a mesh grid
            the_j0 = j0(kr_cross_r)
            the_cos = cos(kz_cross_z)

            EZ = einsum('ik, ilm, klm->lm', mode_P, the_j0, the_cos)

            return EZ


        else:
            print 'File must be opened first.'