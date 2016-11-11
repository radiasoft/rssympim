import numpy as np
from numpy import cos
# Commented out until MPI implementation is ready
#from mpi4py import MPI

class field_data(object):

    def __init__(self, L, n_modes_z):

        self.kz = 2.*np.pi*np.arange(1,n_modes_z+1)/L

        # Use linear strides for indexing the modes
        self.mode_coords = np.zeros((n_modes_z,2))
        self.omega = np.zeros(n_modes_z)
        for idx_z in range(0,n_modes_z):
            self.omega[idx_z]= self.kz[idx_z]

        self.delta_P = np.zeros(n_modes_z)

        self.n_modes_z = n_modes_z

        # Particles are tent functions with widths the narrowest of the
        # k-vectors for each direction. Default for now is to have the
        # particle widths be half the shortest wavelength, which should
        # resolve the wave physics reasonably well.
        ptcl_width_z = .1/max(self.kz)

        self.shape_function_z = 2.*(1.-cos(self.kz*ptcl_width_z))/\
                                (self.kz*self.kz*ptcl_width_z)


    def compute_Az(self, _z):
        """
        Evaluate Az for a set of particles
        :param _z: longitudinal coordinates
        :return: Az, a numpy array
        """
        n_ptcls = np.shape(_z)[0]
        Az = np.zeros(n_ptcls)

        for idx_z in range(0,self.n_modes_z):
            Az += self.mode_coords[idx_z,1]* \
                  cos(self.kz[idx_z]*_z)*self.shape_function_z[idx_z]

        return Az


    def compute_dFzdQ(self, _z):
        """
        Evaluate Fz for a set of particles
        :param _z: longitudinal coordinates
        :return: Fz, a numpy array
        """
        n_ptcls = np.shape(_z)[0]
        dFzdQ = np.zeros(n_ptcls)

        for idx_z in range(0,self.n_modes_z):
            dFzdQ += cos(self.kz[idx_z]*_z)*self.shape_function_z[idx_z]

        return dFzdQ


    def finalize_fields(self):
        """
        MPI communication on the fields at the end of the update sequence
        :return:
        """
        # Commented out until the MPI implementation is ready
        #self.comm.allreduce(self.delta_P, op=MPI.SUM, root=0)
        self.mode_coords[:,0] += self.delta_P[:]
        self.delta_P = np.zeros(self.n_modes_z)


    def compute_energy(self):
        """
        Computes the energy stored in each mode
        :return: numpy array with the field energy of each mode
        """

        squares = self.mode_coords*self.mode_coords
        return 0.5*(squares[:,0] + self.omega*squares[:,1])