import numpy as np
from numpy import sin, cos
from scipy.special import j0, j1, jn_zeros, fresnel

class field_data(object):

    def __init__(self, L, R, n_modes_r, n_modes_z):

        self.kr = jn_zeros(0, n_modes_r)/R
        self.kz = 2.*np.pi*np.arange(1,n_modes_z+1)/L

        # Use linear strides for indexing the modes
        self.mode_coords = np.zeros((n_modes_r*n_modes_z,2))
        self.omega = np.zeros(n_modes_r*n_modes_z)
        for idx_r in range(0,n_modes_r):
            for idx_z in range(0,n_modes_z):
                self.omega[idx_r + (n_modes_r)*idx_z]= \
                    np.sqrt(self.kr[idx_r]**2 +self.kz[idx_z]**2)


        self.n_modes_r = n_modes_r
        self.n_modes_z = n_modes_z

        # Particles are tent functions with widths the narrowest of the
        # k-vectors for each direction. Default for now is to have the
        # particle widths be half the shortest wavelength, which should
        # resolve the wave physics reasonably well.
        ptcl_width_z = .5*2.*np.pi/max(self.kz)
        self.ptcl_width_r = .5*2.*np.pi/max(self.kr)

        self.shape_function_z = 2.*(1.-cos(self.kz*ptcl_width_z))/\
                                (self.kz*self.kz*ptcl_width_z)

        # The shape function for r cannot be analytically evaluated nicely

        #some numerical constants to save computation time
        self.root2 = np.sqrt(2.)
        self.root2opi = np.sqrt(2./np.pi)
        self.quarterpi = 0.25*np.pi


    def my_j0(self, _x):
        """
        Evaluating the integrals of j0 is extremely expensive, so the
        in-between is a piecewise, continuous function that approximates j0
        and can have its antiderivative evaluated quickly. To preserve
        symplecticity, the integral has to come from
        :param _x: an array of values
        :return: j0: approximate value of j0
        """
        return np.where(_x < 1.13229,
                 1.+_x*_x*(.015625*_x*_x-.25),
                 self.root2opi/np.sqrt(_x)*np.cos(_x-self.quarterpi))


    def int_my_j0(self,_x):
        """
        Provide an analytic estimate for the integral of j0 using the
        antiderivatives of my_j0.
        :param _x:
        :return: int_j0: approximate value of the integral of j0
        """

        fresnelC, fresnelS = fresnel(self.root2opi*np.sqrt(_x))

        return np.where(_x < 1.13229,
                 _x*(1.+_x*_x*(_x*_x/320. - 1./12.)),
                 self.root2*(fresnelC + fresnelS))


    def convolved_j0(self, _x):
        """
        Use Romberg integration to approximate the convolution integral
        with j0 to fourth order in the particle size
        :param _x:
        :return:
        """

        return (self.my_j0(np.abs(_x - .5*self.ptcl_width_r))+
                self.my_j0(_x+.5*self.ptcl_width_r))/6. + \
               (self.my_j0(np.abs(_x-.25*self.ptcl_width_r)))+ \
               self.my_j0(_x+.25*self.ptcl_width_r)*2./3.+\
               self.my_j0(_x)/3.


    def convolved_j1(self, _x):
        """
        Use Romberg integration to approximate the convolution integral
        with j1 to fourth order in the particle size
        :param _x:
        :return:
        """

        return (j1(np.abs(_x - .5*self.ptcl_width_r))+
                j1(_x+.5*self.ptcl_width_r))/6. + \
               (j1(np.abs(_x-.25*self.ptcl_width_r)))+\
               j1(_x+.25*self.ptcl_width_r)*2./3.+j1(_x)/3.

    def int_convolved_j0(self, _x):

        return (self.int_my_j0(np.abs(_x - .5*self.ptcl_width_r))+
                self.int_my_j0(_x+.5*self.ptcl_width_r))/6. + \
               (self.int_my_j0(np.abs(_x-.25*self.ptcl_width_r))+ \
               self.int_my_j0(_x+.25*self.ptcl_width_r))*2./3.+\
               self.int_my_j0(_x)/3.


    def int_convolved_j1(self, _x):

        return -1.*((j0(np.abs(_x - .5*self.ptcl_width_r))+
                j0(_x+.5*self.ptcl_width_r))/6. + \
               (j0(np.abs(_x-.25*self.ptcl_width_r)))+\
               j0(_x+.25*self.ptcl_width_r)*2./3.+j0(_x)/3.)


    def compute_Ar(self, _r, _z):
        """
        Evaluate Ar for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Ar, a numpy array
        """
        n_ptcls = np.shape(_r)[0]
        Ar = np.zeros(n_ptcls)

        for idx_r in range(0,self.n_modes_r):
            self.convolution = \
                self.convolved_j1(self.kr[idx_r]*_r)/self.kr[idx_r]
            print self.convolution
            for idx_z in range(0,self.n_modes_z):
                Ar += self.mode_coords[idx_r + self.n_modes_r*idx_z][1]* \
                      self.convolution*\
                      sin(self.kz[idx_z]*_z)*self.shape_function_z[idx_z]

        return Ar


    def compute_Az(self, _r, _z):
        """
        Evaluate Az for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Az, a numpy array
        """
        n_ptcls = np.shape(_r)[0]
        Az = np.zeros(n_ptcls)

        for idx_r in range(0,self.n_modes_r):
            self.convolution = \
                self.convolved_j0(self.kr[idx_r]*_r)/self.kr[idx_r]
            for idx_z in range(0,self.n_modes_z):
                Az += self.mode_coords[idx_r + self.n_modes_r*idx_z][1]* \
                      self.convolution*\
                      cos(self.kz[idx_z]*_z)*self.shape_function_z[idx_z]

        return Az


    def compute_dFrdz(self, _r, _z):

        n_ptcls = np.shape(_r)[0]
        dFrdz = np.zeros(n_ptcls)

        for idx_r in range(0,self.n_modes_r):
            self.convolution = \
                self.int_convolved_j1(self.kr[idx_r]*_r)/self.kr[idx_r]
            for idx_z in range(0,self.n_modes_z):
                dFrdz += -self.kz[idx_z]*\
                         self.mode_coords[idx_r + self.n_modes_r*idx_z][1]* \
                         self.convolution*sin(self.kz[idx_z]*_z)*\
                         self.shape_function_z[idx_z]

        return dFrdz


    def compute_dFzdr(self, _r, _z):

        n_ptcls = np.shape(_r)[0]
        dFzdr = np.zeros(n_ptcls)

        for idx_r in range(0,self.n_modes_r):
            self.convolution = \
                self.int_convolved_j0(self.kr[idx_r]*_r)/self.kr[idx_r]
            for idx_z in range(0,self.n_modes_z):
                dFzdr += self.mode_coords[idx_r + self.n_modes_r*idx_z][1]* \
                         self.convolution*\
                         cos(self.kz[idx_z]*_z)*self.shape_function_z[idx_z]

        return dFzdr


    def compute_dFzdQ(self, _r, _z):
        """
        Evaluate Fz for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Fz, a numpy array
        """
        n_ptcls = np.shape(_r)[0]
        dFzdQ = np.zeros(n_ptcls)

        for idx_r in range(0,self.n_modes_r):
            self.convolution = \
                self.int_convolved_j0(self.kr[idx_r]*_r)/self.kr[idx_r]
            for idx_z in range(0,self.n_modes_z):
                dFzdQ += self.convolution*\
                      cos(self.kz[idx_z]*_z)*self.shape_function_z[idx_z]

        return dFzdQ


    def compute_dFrdQ(self, _r, _z):
        """
        Evaluate Fr for a set of particles
        :param _r: radial coordinates
        :param _z: longitudinal coordinates
        :return: Fr, a numpy array
        """
        n_ptcls = np.shape(_r)[0]
        dFrdQ = np.zeros(n_ptcls)

        for idx_r in range(0,self.n_modes_r):
            self.convolution = \
                self.int_convolved_j1(self.kr[idx_r]*_r)/self.kr[idx_r]
            for idx_z in range(0,self.n_modes_z):
                dFrdQ += self.convolution*\
                      sin(self.kz[idx_z]*_z)*self.shape_function_z[idx_z]

        return dFrdQ