"""
Longitudinal thermal boundary condition.

If a particle leaves the domain longitudinally, then this
boundary replaces that particle with a particle from a thermal distribution.

Author: Stephen Webb
"""

from rssympim.constants import constants as consts
import numpy as np

class longitudinal_thermal:

    def __init__(self, temperature):
        """
        Longitudinal thermal boundary condition. Checks the particle data to see if a particle
        has left the domain longitudinally, then replaces it with a thermal particle of equal weight to
        the lost particle.

        :param temperature: the temperature of the thermal background plasma
        """
        self.temp = temperature


    def apply_boundary(self, ptcl_data, fld_data):
        """
        Apply the radial boundary. Note that this operation may not commute
        with a longitudinal boundary, or they may overwrite or whatnot, so
        be careful and look for artifacts, especially in the corners of the
        domain.

        :param ptcl_data:
        :param fld_data:
        :return:
        """

        out_of_bounds_forward = np.where(ptcl_data.z > fld_data.domain_L)

        n_new_ptcls = np.shape(out_of_bounds_forward)[0]
        if n_new_ptcls > 0:

            # pretend all the particles are on the x-axis for simplicity
            sigma = np.sqrt(consts.k_boltzmann*self.temp/consts.electron_mass)
            vx  = np.random.normal(0.,sigma, n_new_ptcls)
            vy  = np.random.normal(0.,sigma, n_new_ptcls)
            vz  = -np.abs(np.random.normal(0.,sigma, n_new_ptcls)) # Must be moving back into domain
            ptcl_data.z[out_of_bounds_forward] = fld_data.domain_L

            Ar = fld_data.compute_Ar(ptcl_data.r[out_of_bounds_forward],
                                     ptcl_data.z[out_of_bounds_forward],
                                     ptcl_data.qOc[out_of_bounds_forward])
            Az = fld_data.compute_Az(ptcl_data.r[out_of_bounds_forward],
                                     ptcl_data.z[out_of_bounds_forward],
                                     ptcl_data.qOc[out_of_bounds_forward])

            weighted_mass = ptcl_data.mass*ptcl_data.weight[out_of_bounds_forward]

            ptcl_data.pr[out_of_bounds_forward] = weighted_mass*vx + Ar
            ptcl_data.pz[out_of_bounds_forward] = weighted_mass*vz + Az
            # set the angular momentum
            ptcl_data.ell[out_of_bounds_forward] = weighted_mass*vy*ptcl_data.r[out_of_bounds_forward]

        out_of_bounds_backward = np.where(ptcl_data.z < 0.)

        n_new_ptcls = np.shape(out_of_bounds_backward)[0]

        if n_new_ptcls > 0:

            # pretend all the particles are on the x-axis for simplicity
            sigma = np.sqrt(consts.k_boltzmann*self.temp/consts.electron_mass)
            vx  = np.random.normal(0.,sigma, n_new_ptcls)
            vy  = np.random.normal(0.,sigma, n_new_ptcls)
            vz  = -np.abs(np.random.normal(0.,sigma, n_new_ptcls)) # Must be moving back into domain
            ptcl_data.z[out_of_bounds_backward] = 0.

            Ar = fld_data.compute_Ar(ptcl_data.r[out_of_bounds_backward],
                                     ptcl_data.z[out_of_bounds_backward],
                                     ptcl_data.qOc[out_of_bounds_backward])
            Az = fld_data.compute_Az(ptcl_data.r[out_of_bounds_backward],
                                     ptcl_data.z[out_of_bounds_backward],
                                     ptcl_data.qOc[out_of_bounds_backward])

            weighted_mass = ptcl_data.mass*ptcl_data.weight[out_of_bounds_backward]

            ptcl_data.pr[out_of_bounds_backward] = weighted_mass*vx + Ar
            ptcl_data.pz[out_of_bounds_backward] = weighted_mass*vz + Az
            # set the angular momentum
            ptcl_data.ell[out_of_bounds_backward] = weighted_mass*vy*ptcl_data.r[out_of_bounds_backward]