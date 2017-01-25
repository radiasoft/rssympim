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

        self.z = np.zeros(int(n_particles))
        self.pz = np.zeros(int(n_particles))
        self.gamma = np.zeros(int(n_particles))
        self.gamma_mc = np.zeros(int(n_particles))

        self.weight = weight

        self.q = self.weight*charge
        self.m = np.abs(self.weight*mass)

        self.qOm = self.q/self.m
        self.qOc = self.q/consts.c
        self.mc = mass*consts.c


    def compute_gamma(self, field_data):

        self.gamma = np.sqrt((self.pz - self.qOm*
                              field_data.compute_Az(self.r, self.z))**2 +\
                             self.ell**2/(self.m*self.r**2) +
                             (self.mc)**2
                             )/(self.mc)


    def compute_gamma_mc(self, field_data):

        self.gamma_mc = np.sqrt(
            (self.pz-self.qOc*field_data.compute_Az(self.z))**2+
            (self.mc)**2
            )


    def compute_ptcl_energy(self, field_data):

        return self.compute_gamma_mc(field_data)*consts.c