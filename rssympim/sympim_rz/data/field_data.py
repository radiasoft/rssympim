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
# * the r mode is always indexed as 'i'
# * the z mode is always indexed as 'k'
# * the particle number is always indexed as 'j'
#
# This improves readability and makes debugging easier.
#

class field_data(object):

    def __init__(self, L, R, n_modes_r, n_modes_z):

        self.n_modes_r = n_modes_r
        self.n_modes_z = n_modes_z

        self.kr = jn_zeros(0, self.n_modes_r)/R
        self.oneOkr = 1./self.kr
        self.kz = np.pi*np.arange(1,self.n_modes_z+1)/L

        # Needed for the normalization
        zero_zeros = jn_zeros(0, self.n_modes_r)

        self.mode_coords = np.zeros((self.n_modes_r,self.n_modes_z, 2))
        self.mode_mass = np.ones((self.n_modes_r, self.n_modes_z))
        self.radial_coeff = np.ones((self.n_modes_r, self.n_modes_z))

        self.omega = np.zeros((self.n_modes_r,self.n_modes_z))
        for idx_r in range(0,self.n_modes_r):
            for idx_z in range(0,self.n_modes_z):
                self.omega[idx_r,idx_z]= \
                    np.sqrt(self.kr[idx_r]**2 +self.kz[idx_z]**2)
                self.radial_coeff[idx_r, idx_z] = self.kz[idx_z]/self.kr[idx_r]
                self.mode_mass[idx_r, idx_z] = \
                    np.sqrt(consts.c/
                            (.25*R*R*L*(j1(zero_zeros[idx_r])**2)*(1 + self.kz[idx_z]**2/self.kr[idx_r]**2)))


        self.delta_P = np.zeros((self.n_modes_r,self.n_modes_z))

        # Particles are tent functions with widths the narrowest of the
        # k-vectors for each direction. Default for now is to have the
        # particle widths be half the shortest wavelength, which should
        # resolve the wave physics reasonably well.
        ptcl_width_z = 1./max(self.kz)
        self.ptcl_width_r = 1./max(self.kr)

        self.shape_function_z = 2.*(1.-cos(self.kz*ptcl_width_z))/\
                                (self.kz*self.kz*ptcl_width_z*ptcl_width_z)

        # The shape function for r cannot be analytically evaluated nicely

        #some numerical constants to save computation time & space
        self.root2 = np.sqrt(2.)
        self.root2opi = np.sqrt(2./np.pi)
        self.quarterpi = 0.25*np.pi


    def convolved_j0(self, _x):
        """
        Use Romberg integration to approximate the convolution integral
        with j0 to fourth order in the particle size
        :param _x:
        :return:
        """

        return (j0(_x-.5*self.ptcl_width_r) +
                4.*j0(_x) +
                j0(_x+.5*self.ptcl_width_r))/6.


    def convolved_j1(self, _x):
        """
        Use Romberg integration to approximate the convolution integral
        with j1 to fourth order in the particle size
        :param _x:
        :return:
        """

        return (j1(_x-.5*self.ptcl_width_r) +
                4.*j1(_x) +
                j1(_x+.5*self.ptcl_width_r))/6.


    def int_convolved_j1(self, _x):

        return -1.*(j0(_x-.5*self.ptcl_width_r) +
                4.*j0(_x) +
                j0(_x+.5*self.ptcl_width_r))/6.


    def compute_Ar(self, r, z, qOc):
        """
        Evaluate Ar for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Ar, a numpy array
        """

        kr_cross_r = einsum('i,j->ij', self.kr, r)
        kz_cross_z = einsum('k,j->kj', self.kz, z)
        convolved_j1 = einsum('ij, i->ij', self.convolved_j1(kr_cross_r), self.oneOkr)
        convolved_sin = einsum('kj, k->kj', sin(kz_cross_z), self.shape_function_z)

        modeQ = self.mode_coords[:,:,1]*self.mode_mass*self.radial_coeff

        Ar = einsum('ik, ij, kj->j', modeQ, convolved_j1, convolved_sin)*qOc

        return Ar


    def compute_Az(self, r, z, qOc):
        """
        Evaluate Az for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Az, a numpy array
        """

        kr_cross_r = einsum('i,j->ij', self.kr, r)
        kz_cross_z = einsum('k,j->kj', self.kz, z)
        convolved_j0 = einsum('ij, i->ij', self.convolved_j0(kr_cross_r), self.oneOkr)
        convolved_cos = einsum('kj, k->kj', cos(kz_cross_z), self.shape_function_z)

        modeQ = self.mode_coords[:,:,1]*self.mode_mass

        Az = einsum('ik, ij, kj->j', modeQ, convolved_j0, convolved_cos)*qOc

        return Az


    def compute_dFrdz(self, r, z, qOc):

        kr_cross_r = einsum('i,j->ij', self.kr, r)
        kz_cross_z = einsum('k,j->kj', self.kz, z)
        convolved_j1 = einsum('ij, i->ij', self.convolved_j1(kr_cross_r), self.oneOkr)
        convolved_sin = einsum('kj, k->kj', cos(kz_cross_z), self.kz*self.shape_function_z)

        modeQ = self.mode_coords[:,:,1]*self.mode_mass*self.radial_coeff

        dFrdz = einsum('ik, ij, kj->j', modeQ, convolved_j1, convolved_sin)*qOc

        return dFrdz


    def compute_dFzdr(self, r, z, qOc):

        kr_cross_r = einsum('i,j->ij', self.kr, r)
        kz_cross_z = einsum('k,j->kj', self.kz, z)
        convolved_j0 = einsum('ij, i->ij', self.convolved_j0(kr_cross_r), self.oneOkr)
        convolved_cos= einsum('kj, k->kj', cos(kz_cross_z), self.shape_function_z)

        modeQ = self.mode_coords[:,:,1]*self.mode_mass

        dFzdr = einsum('ik, ij, kj->j', modeQ, convolved_j0, convolved_cos)*qOc

        return dFzdr


    def compute_dFzdQ(self, r, z, qOc):
        """
        Evaluate Fz for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Fz, a numpy array
        """

        # Unlike the above functions, this sums over the particles not the modes
        kr_cross_r = einsum('i,j->ij', self.kr, r)
        kz_cross_z = einsum('k,j->kj', self.kz, z)
        convolved_j0 = einsum('ij, i->ij', self.convolved_j0(kr_cross_r), self.oneOkr)
        convolved_cos= einsum('kj, k->kj', cos(kz_cross_z), self.shape_function_z)

        dFzdQ = einsum('ij, kj, ik, j -> ik', convolved_j0, convolved_cos, self.mode_mass, qOc)

        return dFzdQ


    def compute_dFrdQ(self, r, z, qOc):
        """
        Evaluate Fr for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Fr, a numpy array
        """

        # Unlike the above functions, this sums over the particles not the modes
        kr_cross_r = einsum('i,j->ij', self.kr, r)
        kz_cross_z = einsum('k,j->kj', self.kz, z)
        convolved_j1 = einsum('ij, i->ij', self.convolved_j1(kr_cross_r), self.oneOkr)
        convolved_sin = einsum('kj, k->kj', sin(kz_cross_z), self.shape_function_z)

        dFrdQ = einsum('ij, kj, ik, j -> ik', convolved_j1, convolved_sin,
                       self.mode_mass*self.radial_coeff, qOc)

        return dFrdQ


    def finalize_fields(self):
        """
        MPI communication on the fields at the end of the update sequence
        :return:
        """
        # Commented out until the MPI implementation is ready
        #self.comm.allreduce(self.delta_P, op=MPI.SUM, root=0)
        self.mode_coords[:,:,0] += self.delta_P[:,:]
        self.delta_P = np.zeros((self.n_modes_r,self.n_modes_z))


    def compute_energy(self):
        """
        Computes the energy stored in each mode
        :return: numpy array with the field energy of each mode
        """

        Qsqrd = self.mode_coords[:,:,1]*self.mode_coords[:,:,1]
        Psqrd = self.mode_coords[:,:,0]*self.mode_coords[:,:,0]

        return 0.5*(Psqrd + self.omega*self.omega*Qsqrd)