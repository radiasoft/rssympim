import numpy as np
from numpy import sin, cos, einsum
from scipy.special import j0, j1, jn_zeros
from rssympim.constants import constants as consts
# Commented out until MPI implementation is ready
from mpi4py import MPI as mpi

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
    """
    Class that stores the field data and can compute particle-field interactions
     
    Parameters
    ----------
    L: float (cm)
        Length (z) of domain
    
    R: float (cm)
        Radius of domain
    
    n_modes_z: int
        Number of longitudinal modes to be computed
    
    n_modes_r: int
        Number of radial modes to be computed
    
    """
    

    def __init__(self, L, R, n_modes_z, n_modes_r):

        self.n_modes_r = n_modes_r
        self.n_modes_z = n_modes_z

        self.domain_L = L
        self.domain_R = R

        self.kr = jn_zeros(0, self.n_modes_r)/R
        self.kz = np.pi * np.arange(1, self.n_modes_z + 1) / L

        self.oneOkr = 1./self.kr
        self.oneOkz = 1./self.kz

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
                self.mode_mass[idx_z, idx_r] = R*R*L*(j1(zero_zeros[idx_r]))**2/(4.*consts.c)

        self.omegaOtwokz = 0.5 * np.einsum('z, zr -> zr', 1. / self.kz, self.omega)
        self.omegaOtwokr = 0.5 * np.einsum('r, zr -> zr', 1. / self.kr, self.omega)

        self.oneOomega = 1./self.omega

        self.kzOomega = einsum('z, zr -> zr', self.kz, self.oneOomega)
        self.krOomega = einsum('r, zr -> zr', self.kr, self.oneOomega)


        self.delta_P_dc = np.zeros((self.n_modes_z,self.n_modes_r))
        self.delta_P_omega = np.zeros((self.n_modes_z,self.n_modes_r))

        # Particles are tent functions with widths the narrowest of the
        # k-vectors for each direction. Default for now is to have the
        # particle widths be half the shortest wavelength, which should
        # resolve the wave physics reasonably well.
        self.ptcl_width_z = .25*2.*np.pi/max(self.kz)
        self.ptcl_width_r = .25*2.*np.pi/max(self.kr)

        self.shape_function_z = np.exp(-0.5*(self.kz*self.ptcl_width_z)**2)

        # With the conducting boundaries, unphysically large fields can
        # get trapped on the conducting surfaces. To squelch this, we put
        # a tanh-function envelope on the fields to make the fields go to
        # zero quickly but smoothly on the boundaries.

        self.tanh_width = np.max(self.kz)
        self.z_mean = 0.5*self.domain_L

        # Create the mpi communicator
        self.comm = mpi.COMM_WORLD


    def convolved_j0(self, _x, delta_x):
        """
        Use Romberg integration to approximate the convolution integral
        with j0 to fourth order in the particle size
        
        Parameters
        ----------
        _x: float (cm)
            a 2darray of macroparticle phases k_x*x, of shape (n_modes,particles.np)
        
        delta_x: float(cm)
            macroparticle width - 1darray of reals
        
        Returns
        -------
        A 2darray of reals, of shape (n_modes, particles.np)
        
        """
        return (
                   j0(_x - 0.5 * delta_x) +
                    4.*j0(_x) +
                    j0(_x + 0.5 * delta_x)
               ) / 6.


    def convolved_j1(self, _x, delta_x):
        """
        Use Romberg integration to approximate the convolution integral
        with j1 to fourth order in the particle size
        
        Parameters
        ----------
        _x: float (cm)
            a 2darray of macroparticle phases k_x*x, of shape (n_modes,particles.np)
        
        delta_x: float(cm)
            macroparticle width - 1darray of reals
        
        Returns
        -------
        A 2darray of reals, of shape (n_modes, particles.np)
        
        """
        return (
                   j1(_x-0.5*delta_x) +
                    4.*j1(_x) +
                    j1(_x+0.5*delta_x)
               )/6.


    def int_convolved_j1(self, _x, delta_x):
        """
        Analytic integral of the convolved_j1 Romberg approximation
        with j1 to fourth order in the particle size
        
        Parameters
        ----------
        _x: float (cm)
            a 2darray of macroparticle phases k_x*x, of shape (n_modes,particles.np)
        
        delta_x: float(cm)
            macroparticle width - 1darray of reals
        
        Returns
        -------
        A 2darray of reals, of shape (n_modes, particles.np)
        
        """
        return -(
                    j0(_x - 0.5*delta_x) +
                    4.*j0(_x) +
                    j0(_x + 0.5*delta_x)
                )/6.


    def compute_S_r_kick(self, r, z, qOc, **kwargs):

        """
        Evaluates the radial kicks for a set of particles
        
        Parameters
        ----------
        r: float (cm)
            a 1darray of macroparticle coordinates, of shape (particles.np,)
        
        z: float(cm)
            a 1darray of macroparticle coordinates, of shape (particles.np,)
        
         qOc: float(cm)
            a 1darray of macroparticle charge:mass ratios, of shape (particles.np,)
        
        Returns
        -------
        A length-4 list of 1darrays, each of shape (particles.np,)

        """
        # Calculate the convolution quantities we need
        kr_cross_r = einsum('r, p -> rp', self.kr, r)
        # z does not change between S_r and S_r-inverse, so only need to compute once
        if kwargs['inverse'] == False:
            self.kz_cross_z = einsum('z, p -> zp', self.kz, z)
            self.convolved_sin = einsum('zp, z -> zp', sin(self.kz_cross_z), self.shape_function_z)
            self.d_convolved_sin_dz = einsum('zp, z -> zp', cos(self.kz_cross_z), self.kz*self.shape_function_z)
            # same here
            self.delta_r = np.ones(np.size(r)) * self.ptcl_width_r
            self.delta_u = einsum('r, p -> rp', self.kr, self.delta_r)

            # z doesn't change, so tanh-z doesn't change
            self.tanhz = -np.tanh(((z-self.z_mean)**2 - self.z_mean**2)*self.tanh_width**2)

        j1 = self.convolved_j1(kr_cross_r, self.delta_u)
        int_j1 = einsum('rp, r -> rp', self.int_convolved_j1(kr_cross_r, self.delta_u), self.oneOkr)

        # Calculate Q_r for each mode
        modeQr = self.omegaOtwokz * (self.dc_coords[:,:,1] - self.omega_coords[:,:,1])

        # We dress the charge instead of the fields proper
        dressed_charge = self.tanhz*qOc

        kick_z = einsum('zr, rp, zp, p -> p', modeQr, int_j1, self.d_convolved_sin_dz, dressed_charge)
        kick_r = einsum('zr, rp, zp, p -> p', modeQr, j1, self.convolved_sin, dressed_charge)
        dFrdQ = einsum('rp, zp, p -> zr', int_j1, self.convolved_sin, dressed_charge)

        kick_Q0     = dFrdQ*self.omegaOtwokz
        kick_Qomega = -dFrdQ*self.omegaOtwokz

        return kick_z, kick_r, kick_Q0, kick_Qomega


    def compute_Ar(self, r, z, qOc):
        """
        Evaluates the vector potential component Ar for a set of particles
        
        Parameters
        ----------
        r: float (cm)
            a 1darray of macroparticle coordinates, of shape (particles.np,)
        
        z: float(cm)
            a 1darray of macroparticle coordinates, of shape (particles.np,)
        
         qOc: float(cm)
            a 1darray of macroparticle charge:mass ratios, of shape (particles.np,)
        
        Returns
        -------
        A 1darray of shape (particles.np,)

        """

        kz_cross_z = einsum('z, p -> zp', self.kz, z)
        kr_cross_r = einsum('r, p -> rp', self.kr, r)
        delta_r = np.ones(np.size(r))*self.ptcl_width_r
        delta_u = einsum('r, p -> rp', self.kr, delta_r)
        j1 = self.convolved_j1(kr_cross_r, delta_u)
        convolved_sin = einsum('zp, z -> zp', sin(kz_cross_z), self.shape_function_z)

        modeQr = self.omegaOtwokz * (self.dc_coords[:,:,1] - self.omega_coords[:,:,1])

        self.tanhz = -np.tanh(((z - self.z_mean) ** 2 - self.z_mean ** 2) * self.tanh_width ** 2)
        # We dress the charge instead of the fields proper
        dressed_charge = self.tanhz*qOc


        Ar = einsum('zr, rp, zp -> p', modeQr, j1, convolved_sin)*dressed_charge

        return Ar


    def compute_S_z_kick(self, r, z, qOc, **kwargs):

        """
        Evaluates the longitudinal kicks for a set of particles
        
        Parameters
        ----------
        r: float (cm)
            a 1darray of macroparticle coordinates, of shape (particles.np,)
        
        z: float(cm)
            a 1darray of macroparticle coordinates, of shape (particles.np,)
        
         qOc: float(cm)
            a 1darray of macroparticle charge:mass ratios, of shape (particles.np,)
        
        Returns
        -------
        A length-4 list of 1darrays, each of shape (particles.np,)

        """
        # Calculate the convolution quantities we need

        kz_cross_z = einsum('z, p -> zp', self.kz, z)
        convolved_cos = einsum('zp, z -> zp', cos(kz_cross_z), self.shape_function_z)
        int_convolved_cos_dz = einsum('zp, z -> zp', sin(kz_cross_z), self.shape_function_z*self.oneOkz)

        # r does not change between S_z and S_z-inverse, so only need to compute once
        if kwargs['inverse'] == False:
            self.kr_cross_r = einsum('r, p -> rp', self.kr, r)
            self.delta_r = np.ones(np.size(r)) * self.ptcl_width_r
            self.delta_u = einsum('r, p -> rp', self.kr, self.delta_r)
            self.j0 = self.convolved_j0(self.kr_cross_r, self.delta_u)
            self.d_convolved_j0_dr = einsum('rp, r -> rp',
                                            -self.convolved_j1(self.kr_cross_r, self.delta_u), self.kr)

        # Calculate Q_z for each mode
        modeQz = self.omegaOtwokr * (self.dc_coords[:,:,1] + self.omega_coords[:,:,1])

        self.tanhz = -np.tanh(((z - self.z_mean) ** 2 - self.z_mean ** 2) * self.tanh_width ** 2)
        # We dress the charge instead of the fields proper
        dressed_charge = self.tanhz*qOc

        kick_z = einsum('zr, rp, zp -> p', modeQz, self.j0, convolved_cos)*dressed_charge
        kick_r = einsum('zr, rp, zp -> p', modeQz, self.d_convolved_j0_dr, int_convolved_cos_dz)*dressed_charge

        dFzdQ = einsum('rp, zp, p -> zr', self.j0, int_convolved_cos_dz, dressed_charge)

        kick_Q0     = dFzdQ*self.omegaOtwokr
        kick_Qomega = dFzdQ*self.omegaOtwokr

        return kick_z, kick_r, kick_Q0, kick_Qomega


    def compute_Az(self, r, z, qOc):
        """
        Evaluates the vector potential component Az for a set of particles
        
        Parameters
        ----------
        r: float (cm)
            a 1darray of macroparticle coordinates, of shape (particles.np,)
        
        z: float(cm)
            a 1darray of macroparticle coordinates, of shape (particles.np,)
        
         qOc: float(cm)
            a 1darray of macroparticle charge:mass ratios, of shape (particles.np,)
        
        Returns
        -------
        A 1darray of shape (particles.np,)

        """

        kr_cross_r = einsum('r, p -> rp', self.kr, r)
        kz_cross_z = einsum('z, p -> zp', self.kz, z)
        delta_r = np.ones(np.size(r)) * self.ptcl_width_r
        delta_u = einsum('r, p -> rp', self.kr, delta_r)
        convolved_j0 = self.convolved_j0(kr_cross_r, delta_u)
        convolved_cos = einsum('zp, z -> zp', cos(kz_cross_z), self.shape_function_z)

        modeQz = self.omegaOtwokr * (self.dc_coords[:,:,1] + self.omega_coords[:,:,1])

        self.tanhz = -np.tanh(((z - self.z_mean) ** 2 - self.z_mean ** 2) * self.tanh_width ** 2)
        # We dress the charge instead of the fields proper
        dressed_charge = self.tanhz*qOc

        Az = einsum('zr, rp, zp -> p', modeQz, convolved_j0, convolved_cos)*dressed_charge

        return Az


    def finalize_fields(self):
        """MPI communication on the fields at the end of the update sequence"""
        
        # Commented out until the MPI implementation is ready
        self.comm.allreduce(self.delta_P_dc, op=mpi.SUM)
        self.comm.allreduce(self.delta_P_omega, op=mpi.SUM)

        self.dc_coords[:,:,0]    += self.delta_P_dc[:,:]
        self.omega_coords[:,:,0] += self.delta_P_omega[:,:]

        self.delta_P_dc    = np.zeros((self.n_modes_z,self.n_modes_r))
        self.delta_P_omega = np.zeros((self.n_modes_z,self.n_modes_r))


    def compute_energy(self):
        """
        Computes the energy stored in each mode
        
        Returns
        -------
        A 2darray of floats with shape (n_modes_z, n_modes_r)
        """

        # radiation energy
        Qsqrd = self.omega_coords[:,:,1]*self.omega_coords[:,:,1]
        Psqrd = self.omega_coords[:,:,0]*self.omega_coords[:,:,0]

        e_rad = (Psqrd/self.mode_mass + (self.mode_mass*self.omega**2)*Qsqrd)*.5

        # space charge energy
        Dsqrd = self.dc_coords[:,:,0]*self.dc_coords[:,:,0]

        e_drft = Dsqrd/(2.*self.mode_mass)

        energy = e_rad+e_drft

        return energy