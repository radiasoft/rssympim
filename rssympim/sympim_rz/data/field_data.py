import numpy as np
from numpy import sin, cos, einsum
from scipy.special import j0, j1, jn_zeros
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
# * the z mode is always indexed as 'z'
# * the r mode is always indexed as 'r'
# * the particle number is always indexed as 'p'
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

        # These are the parameters required for the moving window,
        # which when applied moves the fields along by the shortest wavelength resolved.
        # For a spectral code, this is a projection from one cylindrical basis to another
        # translated by mw_d -- convolution integrals and the like.
        mw_d = np.pi/np.max(self.kz)
        eikL = np.exp(1.j*self.kz*self.domain_L)
        eikd = np.exp(1.j*self.kz*mw_d)

        # Build a matrix for this stuff
        matrix_eikd_sum = np.einsum('i,j -> ij', eikd, eikd)
        matrix_eikL_sum = np.einsum('i,j -> ij', eikL, eikL)

        matrix_eikd_diff = np.einsum('i,j -> ij', eikd, 1. / eikd)
        matrix_eikL_diff = np.einsum('i,j -> ij', eikL, 1. / eikL)

        Ki, Kj = np.meshgrid(self.kz, self.kz)

        oneOplus = 1. / (1.j * (Ki + Kj))
        # This will have pythonic 'inf' along the diagonal, and will be handled later
        oneOminus = 1. / (1.j * (Ki - Kj))

        # Let's create the two individual terms that appear in the moving window
        sum_matrix  = np.einsum('ij, ij->ij', (matrix_eikL_sum-matrix_eikd_sum), oneOplus)
        diff_matrix = np.einsum('ij, ij->ij', (matrix_eikL_diff-matrix_eikd_diff), oneOminus)

        # diff_matrix has pythonic 'nan' along the diagonal since 0.*float('inf') = nan, but
        # L'Hopital's Rule says that these elements are actually zero
        diff_matrix[np.isnan(diff_matrix)] = 0.

        # Both arrays have a phase multiplier

        sum_matrix = np.einsum('ij, j -> ij', sum_matrix, 1./eikd)
        diff_matrix = np.einsum('ij, j -> ij', diff_matrix, 1./eikd)

        # Compute the final matrices that tell you how the field amplitudes change
        # when projected on a new basis shifted d to the right/
        self.r_shift_matrix = 0.5*np.real(sum_matrix + diff_matrix)
        self.z_shift_matrix = -0.5*np.real(sum_matrix - diff_matrix)


    def convolved_j0(self, _x, delta_x):
        """
        Use Romberg integration to approximate the convolution integral
        with j0 to fourth order in the particle size
        :param _x:
        :return:
        """

        return (j0(_x - 0.5 * delta_x) +
                4.*j0(_x) +
                j0(_x + 0.5 * delta_x)) / 6.


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

        return -(j0(_x-0.5*delta_x) +
                4.*j0(_x) +
                j0(_x+0.5*delta_x))/6.


    def compute_S_r_kick(self, r, z, qOc):

        """
                Evaluate Ar for a set of particles
                :param _r: radial coordinates
                :param _z: longitudinal coordinates
                :return: Ar, a numpy array
                """

        kz_cross_z = einsum('z, p -> zp', self.kz, z)
        kr_cross_r = einsum('r, p -> rp', self.kr, r)
        delta_r = np.ones(np.size(r)) * self.ptcl_width_r
        delta_u = einsum('r, p -> rp', self.kr, delta_r)

        # Calculate the convolution quantities we need
        convolved_j1 = self.convolved_j1(kr_cross_r, delta_u)
        convolved_sin = einsum('zp, z -> zp', sin(kz_cross_z), self.shape_function_z)
        int_convolved_j1 = einsum('rp, r -> rp', self.int_convolved_j1(kr_cross_r, delta_u), self.oneOkr)
        d_convolved_sin_dz = einsum('zp, z -> zp', cos(kz_cross_z), self.kz*self.shape_function_z)

        # Calculate Q_r for each mode
        modeQr = (np.einsum('r, zr -> zr', -self.kr, self.dc_coords[:, :, 1]) + \
                  np.einsum('z, zr -> zr', self.kz, self.omega_coords[:, :, 1])) / self.omega

        kick_z = einsum('zr, rp, zp -> p', modeQr, int_convolved_j1, d_convolved_sin_dz)*qOc
        kick_r = einsum('zr, rp, zp -> p', modeQr, convolved_j1, convolved_sin) * qOc
        dFrdQ = einsum('rp, zp, p -> zr', int_convolved_j1, convolved_sin, qOc)

        kick_Q0     = dFrdQ*einsum('r, zr -> zr', -self.kr, 1./self.omega)
        kick_Qomega = dFrdQ*einsum('z, zr -> zr', self.kz, 1./self.omega)

        return kick_z, kick_r, kick_Q0, kick_Qomega


    def compute_Ar(self, r, z, qOc):
        """
        Evaluate Ar for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Ar, a numpy array
        """

        kz_cross_z = einsum('z, p -> zp', self.kz, z)
        kr_cross_r = einsum('r, p -> rp', self.kr, r)
        delta_r = np.ones(np.size(r))*self.ptcl_width_r
        delta_u = einsum('r, p -> rp', self.kr, delta_r)
        convolved_j1 = self.convolved_j1(kr_cross_r, delta_u)
        convolved_sin = einsum('zp, z -> zp', sin(kz_cross_z), self.shape_function_z)

        modeQr = (np.einsum('r, zr -> zr', -self.kr, self.dc_coords[:,:,1]) +\
                    np.einsum('z, zr -> zr', self.kz, self.omega_coords[:,:,1]))/self.omega

        Ar = einsum('zr, rp, zp -> p', modeQr, convolved_j1, convolved_sin)*qOc

        return Ar


    def compute_S_z_kick(self, r, z, qOc):

        """
        Evaluate the kicks for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Ar, a numpy array
        """

        kz_cross_z = einsum('z, p -> zp', self.kz, z)
        kr_cross_r = einsum('r, p -> rp', self.kr, r)
        delta_r = np.ones(np.size(r)) * self.ptcl_width_r
        delta_u = einsum('r, p -> rp', self.kr, delta_r)

        # Calculate the convolution quantities we need
        convolved_j0 = self.convolved_j0(kr_cross_r, delta_u)
        convolved_cos = einsum('zp, z -> zp', cos(kz_cross_z), self.shape_function_z)
        d_convolved_j0_dr = einsum('rp, r -> rp', -self.convolved_j1(kr_cross_r, delta_u), self.kr)
        int_convolved_cos_dz = einsum('zp, z -> zp', sin(kz_cross_z), self.shape_function_z/self.kz)

        # Calculate Q_z for each mode
        modeQz = (np.einsum('z, zr -> zr', self.kz, self.dc_coords[:,:,1]) +\
                    np.einsum('r, zr -> zr', self.kr, self.omega_coords[:,:,1]))/self.omega

        kick_z = einsum('zr, rp, zp -> p', modeQz, convolved_j0, convolved_cos)*qOc
        kick_r = einsum('zr, rp, zp -> p', modeQz, d_convolved_j0_dr, int_convolved_cos_dz)*qOc

        dFzdQ = einsum('rp, zp, p -> zr', convolved_j0, int_convolved_cos_dz, qOc)

        kick_Q0     = dFzdQ*einsum('z, zr -> zr', self.kz, 1./self.omega)
        kick_Qomega = dFzdQ*einsum('r, zr -> zr', self.kr, 1./self.omega)


        return kick_z, kick_r, kick_Q0, kick_Qomega


    def compute_Az(self, r, z, qOc):
        """
        Evaluate Az for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Az, a numpy array
        """

        kr_cross_r = einsum('r, p -> rp', self.kr, r)
        kz_cross_z = einsum('z, p -> zp', self.kz, z)
        delta_r = np.ones(np.size(r)) * self.ptcl_width_r
        delta_u = einsum('r, p -> rp', self.kr, delta_r)
        convolved_j0 = self.convolved_j0(kr_cross_r, delta_u)
        convolved_cos = einsum('zp, z -> zp', cos(kz_cross_z), self.shape_function_z)

        modeQz = (np.einsum('z, zr -> zr', self.kz, self.dc_coords[:,:,1]) +\
                    np.einsum('r, zr -> zr', self.kr, self.omega_coords[:,:,1]))/self.omega

        Az = einsum('zr, rp, zp -> p', modeQz, convolved_j0, convolved_cos)*qOc

        return Az


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

        # radiation energy
        Qsqrd = self.omega_coords[:,:,1]*self.omega_coords[:,:,1]
        Psqrd = self.omega_coords[:,:,0]*self.omega_coords[:,:,0]

        e_rad = (Psqrd/self.mode_mass + (self.mode_mass*self.omega**2)*Qsqrd)/2

        # space charge energy
        Dsqrd = self.dc_coords[:,:,0]*self.dc_coords[:,:,0]

        e_drft = Dsqrd/(2.*self.mode_mass)

        energy = e_rad+e_drft

        return energy


    def apply_moving_window(self):
        """
        Shift the fields onto a new basis a distance d to the right
        :return:
        """

        # shift the r coordinates
        self.dc_coords = np.einsum('zjn, rj -> zrn', self.dc_coords, self.r_shift_matrix)
        self.omega_coords = np.einsum('zjn, rj -> zrn', self.omega_coords, self.r_shift_matrix)

        # shift the z coordinates
        self.dc_coords = np.einsum('jrn, zj -> zjn', self.dc_coords, self.z_shift_matrix)
        self.omega_coords = np.einsum('jrn, zj -> zjn', self.omega_coords, self.z_shift_matrix)