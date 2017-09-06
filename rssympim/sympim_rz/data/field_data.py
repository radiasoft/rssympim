import numpy as np
from numpy import sin, cos, einsum
from scipy.special import j0, j1, jn_zeros
from rssympim.constants import constants as consts
# Commented out until MPI implementation is ready
#from mpi4py import MPI

# # #
#
# A Note On Indexing Conventions
#
# # #
#
# This class relies heavily on the einsum numpy function.
#
# * the z mode is always indexed as 'i'
# * the r mode is always indexed as 'k'
# * the particle number is always indexed as 'j'
#
# This improves readability and makes debugging easier.
#
# # #

class field_data(object):

    def __init__(self, L, R, n_modes_z, n_modes_r):

        self.n_modes_r = n_modes_r
        self.n_modes_z = n_modes_z

        self.domain_L = L
        self.domain_R = R

        self.kr = jn_zeros(0, self.n_modes_r)/R
        self.oneOkr = 1./self.kr
        self.kz = np.pi * np.arange(1, self.n_modes_z + 1) / L

        # Needed for the normalization
        zero_zeros = jn_zeros(0, self.n_modes_r)

        self.omega_coords = np.zeros((self.n_modes_z,self.n_modes_r, 2))
        self.dc_coords = np.zeros((self.n_modes_z, self.n_modes_r, 2))
        self.mode_mass = np.ones((self.n_modes_z, self.n_modes_r))

        self.omega = np.zeros((self.n_modes_z,self.n_modes_r))
        for idx_r in range(0,self.n_modes_r):
            for idx_z in range(0,self.n_modes_z):
                self.omega[idx_z,idx_r]= \
                    np.sqrt(self.kr[idx_r]**2 +self.kz[idx_z]**2)
                # Integral of cos^2(k_z z)*J_z(k_r r)^2 over the domain volume
                self.mode_mass[idx_z, idx_r] = .5*R*R*L*(j1(zero_zeros[idx_r]))**2/4.


        self.delta_P_dc = np.zeros((self.n_modes_z,self.n_modes_r))
        self.delta_P_omega = np.zeros((self.n_modes_z,self.n_modes_r))

        # Particles are tent functions with widths the narrowest of the
        # k-vectors for each direction. Default for now is to have the
        # particle widths be half the shortest wavelength, which should
        # resolve the wave physics reasonably well.
        self.ptcl_width_z = .25*2.*np.pi/max(self.kz)
        self.ptcl_width_r = .25*2.*np.pi/max(self.kr)

        self.shape_function_z = np.exp(-0.5*(self.kz*self.ptcl_width_z)**2)


    def convolved_j0(self, _x, delta_x):
        """
        Use Romberg integration to approximate the convolution integral
        with j0 to fourth order in the particle size
        :param _x:
        :return:
        """

        return (j0(_x-0.5*delta_x) +
                4.*j0(_x) +
                j0(_x+0.5*delta_x))/6.


    def convolved_j1(self, _x, delta_x):
        """
        Use Romberg integration to approximate the convolution integral
        with j1 to fourth order in the particle size
        :param _x:
        :return:
        """

        return (j1(_x-0.5*delta_x) +
                4.*j1(_x) +
                j1(_x+0.5*delta_x))/6.


    def int_convolved_j1(self, _x, delta_x):

        return -1.*(j0(_x-0.5*delta_x) +
                4.*j0(_x) +
                j0(_x+0.5*delta_x))/6.


    def compute_Ar(self, r, z, qOc):
        """
        Evaluate Ar for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Ar, a numpy array
        """

        kz_cross_z = einsum('i, j -> ij', self.kz, z)
        kr_cross_r = einsum('k, j -> kj', self.kr, r)
        delta_r = np.ones(np.shape(r)[0])*self.ptcl_width_r
        delta_u = einsum('k, j -> kj', self.kr, delta_r)
        convolved_j1 = self.convolved_j1(kr_cross_r, delta_u)
        convolved_sin = einsum('ij, i -> ij', sin(kz_cross_z), self.shape_function_z)

        modeQr = (np.einsum('k, ik -> ik', -self.kr, self.dc_coords[:,:,1]) +\
                    np.einsum('i, ik -> ik', self.kz, self.omega_coords[:,:,1]))/self.omega

        Ar = einsum('ik, kj, ij -> j', modeQr, convolved_j1, convolved_sin)*qOc

        return Ar


    def compute_dFrdz(self, r, z, qOc):

        kr_cross_r = einsum('k, j -> kj', self.kr, r)
        kz_cross_z = einsum('i, j -> ij', self.kz, z)
        delta_r = np.ones(np.shape(r)[0])*self.ptcl_width_r
        delta_u = einsum('k, j -> kj', self.kr, delta_r)
        int_convolved_j1 = einsum('kj, k -> kj', self.int_convolved_j1(kr_cross_r, delta_u), self.oneOkr)
        d_convolved_sin_dz = einsum('ij, i -> ij', cos(kz_cross_z), self.kz*self.shape_function_z)

        modeQr = (np.einsum('k, ik -> ik', -self.kr, self.dc_coords[:,:,1]) +\
                    np.einsum('i, ik -> ik', self.kz, self.omega_coords[:,:,1]))/self.omega

        dFrdz = einsum('ik, kj, ij -> j', modeQr, int_convolved_j1, d_convolved_sin_dz)*qOc

        return dFrdz


    def compute_dFrdQ(self, r, z, qOc):
        """
        Evaluate Fr for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Fr, a numpy array
        """

        # Unlike the above functions, this sums over the particles not the modes
        kr_cross_r = einsum('k, j -> kj', self.kr, r)
        kz_cross_z = einsum('i, j -> ij', self.kz, z)
        delta_r = np.ones(np.shape(r)[0])*self.ptcl_width_r
        delta_u = einsum('k, j -> kj', self.kr, delta_r)
        convolved_j1 = einsum('kj, k -> kj', self.int_convolved_j1(kr_cross_r, delta_u), self.oneOkr)
        convolved_sin = einsum('ij, i -> ij', sin(kz_cross_z), self.shape_function_z)

        dFrdQ = einsum('kj, ij, j -> ik', convolved_j1, convolved_sin, qOc)

        dFrdQ0     = dFrdQ*einsum('k, ik -> ik', -self.kr, 1./self.omega)
        dFrdQomega = dFrdQ*einsum('i, ik -> ik', self.kz, 1./self.omega)

        return dFrdQ0, dFrdQomega


    def compute_Az(self, r, z, qOc):
        """
        Evaluate Az for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Az, a numpy array
        """

        kr_cross_r = einsum('k, j -> kj', self.kr, r)
        kz_cross_z = einsum('i, j -> ij', self.kz, z)
        delta_r = np.ones(np.shape(r)[0])*self.ptcl_width_r
        delta_u = einsum('k, j -> kj', self.kr, delta_r)
        convolved_j0 = self.convolved_j0(kr_cross_r, delta_u)
        convolved_cos = einsum('ij, i -> ij', cos(kz_cross_z), self.shape_function_z)

        modeQz = (np.einsum('k, ik -> ik', self.kz, self.dc_coords[:,:,1]) +\
                    np.einsum('i, ik -> ik', self.kr, self.omega_coords[:,:,1]))/self.omega

        Az = einsum('ik, kj, ij -> j', modeQz, convolved_j0, convolved_cos)*qOc

        return Az


    def compute_dFzdr(self, r, z, qOc):

        kr_cross_r = einsum('k, j -> kj', self.kr, r)
        kz_cross_z = einsum('i, j -> ij', self.kz, z)
        delta_r = np.ones(np.shape(r)[0])*self.ptcl_width_r
        delta_u = einsum('k, j -> kj', self.kr, delta_r)
        d_convolved_j0_dr = einsum('kj, k -> kj',-self.convolved_j1(kr_cross_r, delta_u), self.kr)
        int_convolved_cos_dz = einsum('ij, i -> ij', sin(kz_cross_z), self.shape_function_z/self.kz)

        modeQz = (np.einsum('k, ik -> ik', self.kz, self.dc_coords[:,:,1]) +\
                    np.einsum('i, ik -> ik', self.kr, self.omega_coords[:,:,1]))/self.omega

        dFzdr = einsum('ik, kj, ij -> j', modeQz, d_convolved_j0_dr, int_convolved_cos_dz)*qOc

        return dFzdr


    def compute_dFzdQ(self, r, z, qOc):
        """
        Evaluate Fz for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Fz, a numpy array
        """

        # Unlike the above functions, this sums over the particles not the modes
        kr_cross_r = einsum('k, j -> kj', self.kr, r)
        kz_cross_z = einsum('i, j -> ij', self.kz, z)
        delta_r = np.ones(np.shape(r)[0])*self.ptcl_width_r
        delta_u = einsum('k, j -> kj', self.kr, delta_r)
        convolved_j0 = self.convolved_j0(kr_cross_r, delta_u)
        int_convolved_cos_dz = einsum('ij, i -> ij', sin(kz_cross_z), self.shape_function_z/self.kz)

        dFzdQ = einsum('kj, ij, j -> ik', convolved_j0, int_convolved_cos_dz, qOc)

        dFrdQ0     = dFzdQ*einsum('k, ik -> ik', self.kz, 1./self.omega)
        dFrdQomega = dFzdQ*einsum('i, ik -> ik', self.kr, 1./self.omega)

        return dFrdQ0, dFrdQomega


    def finalize_fields(self):
        """
        MPI communication on the fields at the end of the update sequence
        :return:
        """
        # Commented out until the MPI implementation is ready
        #self.comm.allreduce(self.delta_P, op=MPI.SUM, root=0)
        self.dc_coords[:,:,0]    += self.delta_P_dc[:,:]
        self.omega_coords[:,:,0] += self.delta_P_omega[:,:]

        self.delta_P_dc    = np.zeros((self.n_modes_z,self.n_modes_r))
        self.delta_P_omega = np.zeros((self.n_modes_z,self.n_modes_r))


    def compute_energy(self):
        """
        Computes the energy stored in each mode
        :return: numpy array with the field energy of each mode
        """

        Qsqrd = self.omega_coords[:,:,1]*self.omega_coords[:,:,1]
        Psqrd = self.omega_coords[:,:,0]*self.omega_coords[:,:,0]

        Dsqrd = self.dc_coords[:,:,0]*self.dc_coords[:,:,0]

        energy = 0.5*((Psqrd+Dsqrd)/self.mode_mass +
                      self.mode_mass*self.omega*self.omega*Qsqrd)

        return energy