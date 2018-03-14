"""
Radial thermal boundary condition.

If a particle leaves the domain radially (r > R_domain), then this
boundary replaces that particle with a particle from a thermal distribution.

Author: Stephen Webb
"""

from rssympim.constants import constants as consts
import numpy as np

class radial_thermal:

    def __init__(self, temperature):
        """
        Radial thermal boundary condition. Checks the particle data to see if a particle
        has left the domain radially, then replaces it with a thermal particle of equal weight to
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

        out_of_bounds = np.where(ptcl_data.r > fld_data.domain_R)

        n_new_ptcls = np.shape(out_of_bounds[0])[0]

        if n_new_ptcls > 0:

            # pretend all the particles are on the x-axis for simplicity
            sigma = np.sqrt(consts.k_boltzmann*self.temp/consts.electron_mass)
            vx  = -np.abs(np.random.normal(0.,sigma, n_new_ptcls)) # Must be moving radially inward
            vy  = np.random.normal(0.,sigma, n_new_ptcls)
            vz  = np.random.normal(0.,sigma, n_new_ptcls)
            ptcl_data.r[out_of_bounds] = fld_data.domain_R

            Ar = fld_data.compute_Ar(ptcl_data.r[out_of_bounds],
                                     ptcl_data.z[out_of_bounds],
                                     ptcl_data.qOc[out_of_bounds])
            Az = fld_data.compute_Az(ptcl_data.r[out_of_bounds],
                                     ptcl_data.z[out_of_bounds],
                                     ptcl_data.qOc[out_of_bounds])

            weighted_mass = ptcl_data.mass*ptcl_data.weight[out_of_bounds]

            ptcl_data.pr[out_of_bounds] = weighted_mass*vx + Ar
            ptcl_data.pz[out_of_bounds] = weighted_mass*vz + Az
            # set the angular momentum
            ptcl_data.ell[out_of_bounds] = weighted_mass*vy*ptcl_data.r[out_of_bounds]

