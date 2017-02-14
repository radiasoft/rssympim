import numpy as np
from numpy import sin, cos, einsum
from scipy.special import j0, j1, jn_zeros, fresnel
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
        self.mode_norms = np.ones((self.n_modes_r, self.n_modes_z))
        #self.mode_momenta = np.zeros((n_modes_r, n_modes_z))
        self.omega = np.zeros((self.n_modes_r,self.n_modes_z))
        for idx_r in range(0,self.n_modes_r):
            for idx_z in range(0,self.n_modes_z):
                self.omega[idx_r,idx_z]= \
                    np.sqrt(self.kr[idx_r]**2 +self.kz[idx_z]**2)
                self.mode_norms[idx_r, idx_z] = \
                    1/np.sqrt(.25*R*R*L*j1(zero_zeros[idx_r])*j1(zero_zeros[idx_r]))


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


    def my_j0(self, _x):
        """
        Evaluating the integrals of j0 is extremely expensive, so the
        in-between is a piecewise, continuous function that approximates j0
        and can have its antiderivative evaluated quickly. To preserve
        symplecticity, the integral has to come from an analytic form
        :param _x: an array of values
        :return: j0: approximate value of j0
        """

        # Use Horner's rule to minimize computation a bit

        _xsqrd = _x*_x

        return np.where(_x < 4.14048,
                        1. + _xsqrd*(
                            -1./4. + _xsqrd*(
                                1./64. + _xsqrd*(
                                    -1./2304. + _xsqrd*(
                                        1./147456. + _xsqrd*(
                                            -1./14745600 + _xsqrd/2123366400.)
                                        )
                                    )
                                )
                            ),
                        self.root2opi/np.sqrt(_x)*np.cos(_x-self.quarterpi))


    def int_my_j0(self,_x):
        """
        Provide an analytic estimate for the integral of j0 using the
        antiderivatives of my_j0.
        :param _x:
        :return: int_j0: approximate value of the integral of j0
        """

        fresnelC, fresnelS = fresnel(self.root2opi*np.sqrt(_x))
        _xsqrd = _x*_x

        # Use Horner's rule to minimize computation a bit
        # This includes an error correction factor for the asymptotic form
        return np.where(_x < 4.14048,
                 _x*(1. + _xsqrd*(
                     -1./12. + _xsqrd*(
                         1./320. + _xsqrd*(
                             -1./16128. + _xsqrd*(
                                 1./1327104. + _xsqrd*(
                                     -1./162201600. + _xsqrd/27603763200.)
                             )
                         )
                     )
                 )),
                 self.root2*(fresnelC + fresnelS)-.414)


    def convolved_j0(self, _x):
        """
        Use Romberg integration to approximate the convolution integral
        with j0 to fourth order in the particle size
        :param _x:
        :return:
        """

        return (self.my_j0(_x-.5*self.ptcl_width_r) +
                4.*self.my_j0(_x) +
                self.my_j0(_x+.5*self.ptcl_width_r))/6.


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


    def int_convolved_j0(self, _x):

        return (self.int_my_j0(_x-.5*self.ptcl_width_r) +
                4.*self.int_my_j0(_x) +
                self.int_my_j0(_x+.5*self.ptcl_width_r))/6.


    def int_convolved_j1(self, _x):

        return -1.*(j0(_x-.5*self.ptcl_width_r) +
                4.*j0(_x) +
                j0(_x+.5*self.ptcl_width_r))/6.


    def compute_Ar(self, _r, _z):
        """
        Evaluate Ar for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Ar, a numpy array
        """

        kr_cross_r = einsum('i,j->ij', self.kr, _r)
        kz_cross_z = einsum('k,j->kj', self.kz, _z)
        convolved_j1 = einsum('ij, i->ij', self.convolved_j1(kr_cross_r), self.oneOkr)
        convolved_sin = einsum('kj, k->kj', sin(kz_cross_z), self.shape_function_z)

        modeQ = self.mode_coords[:,:,1]*self.mode_norms

        Ar = einsum('ik, ij, kj->j', modeQ, convolved_j1, convolved_sin)

        return Ar


    def compute_Az(self, _r, _z):
        """
        Evaluate Az for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Az, a numpy array
        """

        kr_cross_r = einsum('i,j->ij', self.kr, _r)
        kz_cross_z = einsum('k,j->kj', self.kz, _z)
        convolved_j0 = einsum('ij, i->ij', self.convolved_j0(kr_cross_r), self.oneOkr)
        convolved_cos = einsum('kj, k->kj', cos(kz_cross_z), self.shape_function_z)

        modeQ = self.mode_coords[:,:,1]*self.mode_norms

        Az = einsum('ik, ij, kj->j', modeQ, convolved_j0, convolved_cos)

        return Az


    def compute_dFrdz(self, _r, _z):

        kr_cross_r = einsum('i,j->ij', self.kr, _r)
        kz_cross_z = einsum('k,j->kj', self.kz, _z)
        convolved_j1 = einsum('ij, i->ij', self.convolved_j1(kr_cross_r), self.oneOkr)
        convolved_sin = einsum('kj, k->kj', cos(kz_cross_z), self.shape_function_z)

        modeQ = self.mode_coords[:,:,1]*self.mode_norms

        dFrdz = einsum('ik, ij, kj->j', modeQ, convolved_j1, convolved_sin)

        return dFrdz


    def compute_dFzdr(self, _r, _z):

        kr_cross_r = einsum('i,j->ij', self.kr, _r)
        kz_cross_z = einsum('k,j->kj', self.kz, _z)
        convolved_j0 = einsum('ij, i->ij', self.convolved_j0(kr_cross_r), self.oneOkr)
        convolved_cos= einsum('kj, k->kj', cos(kz_cross_z), self.shape_function_z)

        modeQ = self.mode_coords[:,:,1]*self.mode_norms

        dFzdr = einsum('ik, ij, kj->j', modeQ, convolved_j0, convolved_cos)

        return dFzdr


    def compute_dFzdQ(self, _r, _z):
        """
        Evaluate Fz for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Fz, a numpy array
        """

        # Unlike the above functions, this sums over the particles not the modes
        kr_cross_r = einsum('i,j->ij', self.kr, _r)
        kz_cross_z = einsum('k,j->kj', self.kz, _z)
        convolved_j0 = einsum('ij, i->ij', self.convolved_j0(kr_cross_r), self.oneOkr)
        convolved_cos= einsum('kj, k->kj', cos(kz_cross_z), self.shape_function_z)

        dFzdQ = einsum('ij, kj, ik -> ik', convolved_j0, convolved_cos, self.mode_norms)

        return dFzdQ


    def compute_dFrdQ(self, _r, _z):
        """
        Evaluate Fr for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Fr, a numpy array
        """

        # Unlike the above functions, this sums over the particles not the modes
        kr_cross_r = einsum('i,j->ij', self.kr, _r)
        kz_cross_z = einsum('k,j->kj', self.kz, _z)
        convolved_j1 = einsum('ij, i->ij', self.convolved_j1(kr_cross_r), self.oneOkr)
        convolved_sin = einsum('kj, k->kj', sin(kz_cross_z), self.shape_function_z)

        dFrdQ = einsum('ij, kj, ik -> ik', convolved_j1, convolved_sin, self.mode_norms)

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