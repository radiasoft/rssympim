"""
Boundary condition for particles.

If a particle exits the domain radially (r > R_domain) then this boundary condition will reflect the particle back into the domain.

Author: Stephen Webb
"""

import numpy as np

class radial_reflecting:

    def __init__(self):
        """
        Radial reflecting boundary condition. Checks the particle data to see if a particle
        has left the domain radially, then flips the sign of its velocity
        """


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

        # compute v_r
        Ar = fld_data.compute_Ar(ptcl_data.r[out_of_bounds],
                                 ptcl_data.z[out_of_bounds],
                                 ptcl_data.qOc[out_of_bounds])
        Pr = ptcl_data.pr[out_of_bounds] + ptcl_data.qOc[out_of_bounds]*Ar

        # compute how far out the particle got and for how long
        diff_r = ptcl_data.r[out_of_bounds] - fld_data.domain_R

        # This is very incorrect for a relativistic particle.
        diff_t = diff_r*ptcl_data.m[out_of_bounds]/Pr

        # Flip the mechanical momentum
        Pr *= -1

        # return the particle to the boundary, then drift it back in
        ptcl_data.r[out_of_bounds] = fld_data.domain_R + Pr*diff_t/(ptcl_data.m[out_of_bounds])

        # set the correct momentum
        Ar = fld_data.compute_Ar(ptcl_data.r[out_of_bounds],
                                 ptcl_data.z[out_of_bounds],
                                 ptcl_data.qOc[out_of_bounds])
        ptcl_data.pr[out_of_bounds] = Pr - ptcl_data.qOc[out_of_bounds]*Ar
