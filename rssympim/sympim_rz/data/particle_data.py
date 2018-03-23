import numpy as np

from rssympim.constants import constants as consts


class particle_data:
    """
    Class that stores the particle data and can compute the particle ensemble quantities
     
    Parameters
    ----------
    n_total: int
        total number of macroparticles across all processors

    n_particles: int
        number of macroparticles
    
    charge: float (esu)
        charge of the physical species
    
    mass: float (grams)
        mass of the physical species
    
    weight: float
        number of particles per macroparticle
    
    species_name: string (optional)
        name of the particle species
    """

    def __init__(self, n_particles, charge, mass, weight, species_name=False):

        self.n_total = n_particles
        self.np = n_particles

        self.r = np.zeros(n_particles)
        self.z = np.zeros(n_particles)
        self.pr = np.zeros(n_particles)
        self.pz = np.zeros(n_particles)
        self.ell = np.zeros(n_particles)
        self.gamma = np.zeros(n_particles)
        self.gamma_mc = np.zeros(n_particles)

        self.weight = weight*np.ones(n_particles)

        self.charge = charge
        self.mass   = mass

        self.species_name = species_name

        self.qOc = (self.charge/consts.c)*np.ones(n_particles)
        self.mc = (self.mass*consts.c)*np.ones(n_particles)


    def compute_gamma(self, field_data):
        """
        Compute the individual particle gammas, mostly for the constant magnetic field part of the simulation.
        
        Parameters
        ----------
        field_data: data.field_data
            field data class object
        """

        self.gamma = np.sqrt((self.pr - field_data.compute_Ar(self.r, self.z, self.qOc))**2 +\
                             (self.pz - field_data.compute_Az(self.r, self.z, self.qOc))**2 +\
                             self.ell**2/(self.r**2) + (self.mc)**2 )/(self.mc)


    def compute_gamma_mc(self, field_data):
        """
        Compute the quantity gamma*m*c for the particle array
        
        Parameters
        ----------
        field_data: data.field_data
            field data class object
        """

        self.compute_gamma(field_data)
        self.gamma_mc = self.gamma*self.mc


    def compute_ptcl_energy(self, field_data):
        """
        Compute the quantity gamma*m*c^2 for the particle array
        
        Parameters
        ----------
        field_data: data.field_data
            field data class object
        
        Returns
        -------
        A 1darray of reals, of shape (particles.np,)
        
        """

        self.compute_gamma_mc(field_data)

        # Return the actual energy in ergs, versus the particle Hamiltonian
        # Note that this is not the "conserved quantity" that would be
        # associated with the Hamiltonian.
        return self.gamma_mc*consts.c


    def compute_ptcl_hamiltonian(self, field_data):
        """
        Return the Hamiltonian (H/c) conjugate to ct being the independent variable
        
        Parameters
        ----------
        field_data: data.field_data
            field data class object
        
        Returns
        -------
        A 1darray of reals, of shape (particles.np,)
        """

        self.compute_gamma_mc(field_data)

        return self.gamma_mc


    def set_ptcl_weights(self, wgts):
        """
        Set the particle weights, and the associated qOc and mc values, consistently.

        :param wgts:
        :return:
        """

        self.weight[:] = wgts[:]
        self.qOc = self.weight*self.charge/consts.c
        self.mc = self.weight*self.mass*consts.c


    def r_boundaries(self, fld_data):
        """
        Cylindrical coordinates wrap back on each other when the particle
        passes through the origin. This flips the sign on p_r and makes r
        positive if a particle has drifted past the axis
        
        Parameters
        ----------
        field_data: data.field_data
            field data class object
        
        """

        is_negative = np.where(self.r < 0.)
        # convert to mechanical momentum
        self.pr[is_negative] = self.pr[is_negative] - \
                               fld_data.compute_Ar(self.r[is_negative],
                                                   self.z[is_negative],
                                                   self.qOc[is_negative])

        # flip across the axis
        self.r[is_negative] *= -1.
        self.pr[is_negative] *= -1.

        # return to canonical momentum
        self.pr[is_negative] = self.pr[is_negative] + \
                               fld_data.compute_Ar(self.r[is_negative],
                                                   self.z[is_negative],
                                                   self.qOc[is_negative])