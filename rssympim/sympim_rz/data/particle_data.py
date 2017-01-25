import numpy as np

from rssympim.constants import constants as consts


class particle_data:

    def __init__(self, n_particles, charge, mass, weight):
        """
        Stores the particle data and can compute the particle ensemble
        quantities
        :param n_particles: number of macroparticles, an int
        :param charge: charge of the physical species in esu
        :param mass: mass of the physical species in esu
        :param weight: number of particles per macroparticle
        """

        self.r = np.zeros(n_particles)
        self.z = np.zeros(n_particles)
        self.pr = np.zeros(n_particles)
        self.pz = np.zeros(n_particles)
        self.ell = np.zeros(n_particles)
        self.gamma = np.zeros(n_particles)
        self.gamma_mc = np.zeros(n_particles)

        self.weight = weight

        self.q = self.weight*charge
        self.m = np.abs(self.weight*mass)

        self.qOm = self.q/self.m
        self.qOc = self.q/consts.c
        self.mc = mass*consts.c


    def compute_gamma(self, field_data):

        self.gamma = np.sqrt((self.pr - self.qOm*
                              field_data.compute_Ar(self.r,self.z))**2 +\
                             (self.pz - self.qOm*
                              field_data.compute_Az(self.r, self.z))**2 +\
                             self.ell**2/(self.m*self.r**2) +
                             (self.mc)**2
                             )/(self.mc)


    def compute_gamma_mc(self, field_data):

        self.gamma_mc = consts.c*np.sqrt(
            (self.pr - self.qOc*field_data.compute_Ar(self.r,self.z))**2 +
            (self.pz - self.qOc*field_data.compute_Az(self.r, self.z))**2 +
            self.ell**2/(self.m*self.r**2) + (self.mc)**2
            )

    def compute_ptcl_energy(self, field_data):

        return self.compute_gamma_mc(field_data)*consts.c


    def r_boundaries(self):
        """
        Cylindrical coordinates wrap back on each other when the particle
        passes through the origin. This flips the sign on p_r and makes r
        positive if a particle has drifted past the axis
        """

        is_negative = np.where(self.r < 0.)
        self.r[is_negative] *= -1.
        self.pr[is_negative] *= -1.