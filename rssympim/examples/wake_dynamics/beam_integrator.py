#####
#
# The wake formation simulation requires a different
# update sequence with external particles and fields
# -- the beam space charge -- and therefore we have
# a modified integrator class for that. This class
# models a parabolic beam moving to the right at the
# speed of light, kicking particles with externally
# defined fields along the way. This is to be used
# with the wake_formation.py input.
#
#####

from rssympim.sympim_rz.maps import ptcl_maps, field_maps, similarity_maps

import numpy as np

from scipy import special

from rssympim.constants import constants as consts

class beam_integrator:

    def __init__(self, beam_radius, beam_length, N_beam, dt, fld_data):

        self.r_beam = beam_radius
        self.z_beam = beam_length

        self.dt = dt

        # A kick coefficient that appears frequently
        self.kick_coeff = -3*N_beam*consts.electron_charge/(4*self.z_beam)

        self.ptcl_maps = ptcl_maps.ptcl_maps(dt)
        self.fld_maps  = field_maps.field_maps(fld_data, dt)
        self.sim_maps  = similarity_maps.similarity_maps()


    def update(self, ptcl_data, fld_data, beam_pos):

        self.fld_maps.half_advance_forward(fld_data)
        self.update_ptcls(ptcl_data, fld_data, beam_pos)
        self.fld_maps.half_advance_forward(fld_data)


    def update_ptcls(self, ptcl_data, fld_data, beam_pos):
        self.phi_kick(ptcl_data, beam_pos, 0.5 * self.dt)

        # always compute the new gamma_mc after the field map update
        self.compute_gamma_mc(ptcl_data, fld_data, beam_pos)

        # Update sequence goes
        # M_ell S_r D_r S_r^-1 S_z D_z S_z^-1 S_r D_r S_r^-1 M_ell
        self.ptcl_maps.half_angular_momentum(ptcl_data)

        #self.sim_maps.S_r(fld_data, ptcl_data)
        #self.ptcl_maps.half_drift_r(ptcl_data, fld_data)
        #self.sim_maps.S_r_inverse(fld_data, ptcl_data)

        self.S_z_external(ptcl_data, beam_pos)
        self.sim_maps.S_z(fld_data, ptcl_data)
        self.ptcl_maps.half_drift_z(ptcl_data)
        self.sim_maps.S_z_inverse(fld_data, ptcl_data)
        self.S_z_inverse_external(ptcl_data, beam_pos)

        self.sim_maps.S_r(fld_data, ptcl_data)
        self.ptcl_maps.drift_r(ptcl_data)
        self.sim_maps.S_r_inverse(fld_data, ptcl_data)

        self.S_z_external(ptcl_data, beam_pos)
        self.sim_maps.S_z(fld_data, ptcl_data)
        self.ptcl_maps.half_drift_z(ptcl_data)
        self.sim_maps.S_z_inverse(fld_data, ptcl_data)
        self.S_z_inverse_external(ptcl_data, beam_pos)

        self.ptcl_maps.half_angular_momentum(ptcl_data)

        self.phi_kick(ptcl_data, beam_pos, 0.5 * self.dt)

        # Add the delta-P to each mode
        fld_data.finalize_fields()


    def phi_kick(self, ptcl_data, beam_pos, dtau):
        """
        Applies the space charge kick due to the beam fields
        """

        # find particles inside the beam length
        ptcl_z_within_beam = np.where(np.abs(ptcl_data.z - beam_pos) < self.z_beam)
        r_arg = ptcl_data.r[ptcl_z_within_beam]**2/(2.*self.r_beam**2)
        z_arg = (ptcl_data.z[ptcl_z_within_beam]-beam_pos)/(self.z_beam)

        grad_r = 2.* ((1. - np.exp(-r_arg))/ptcl_data.r[ptcl_z_within_beam]) \
                    * (1. - z_arg**2)
        grad_z = (
                        consts.euler_gamma + special.gamma(0.1) *
                        (1. - special.gammainc(0.1, r_arg))
                        + np.log(r_arg)
                    ) * (-2.*z_arg/self.z_beam**2)

        grad_r *= self.kick_coeff
        grad_z *= self.kick_coeff

        kick_pr = ptcl_data.qOc[ptcl_z_within_beam] * grad_r * dtau
        kick_pz = ptcl_data.qOc[ptcl_z_within_beam] * grad_z * dtau

        ptcl_data.pr[ptcl_z_within_beam] += kick_pr
        ptcl_data.pz[ptcl_z_within_beam] += kick_pz


    def S_z_external(self, ptcl_data, beam_pos):
        """
        Applies the similarity transformation due to the beam fields
        """
        kick_pz, kick_pr = self.compute_kick(ptcl_data, beam_pos)

        ptcl_data.pz -= kick_pz
        ptcl_data.pr -= kick_pr


    def S_z_inverse_external(self, ptcl_data, beam_pos):
        """
        Applies the inverse similarity transformation due to the beam fields
        """

        kick_pz, kick_pr = self.compute_kick(ptcl_data, beam_pos)

        ptcl_data.pz += kick_pz
        ptcl_data.pr += kick_pr


    def compute_kick(self, ptcl_data, beam_pos):

        kick_pr = np.zeros(np.shape(ptcl_data.pr))
        kick_pz = np.zeros(np.shape(ptcl_data.pz))

        ptcl_z_within_beam = np.where(np.abs(ptcl_data.z - beam_pos) < self.z_beam)

        r_arg = ptcl_data.r[ptcl_z_within_beam]**2/(2.*self.r_beam**2)
        z_arg = ptcl_data.z[ptcl_z_within_beam]

        grad_r_int_z = -2.*(
                         (1. - np.exp(-r_arg))/ptcl_data.r[ptcl_z_within_beam]
                        ) * (
                                z_arg**3/(3.*self.z_beam**2) - z_arg/self.z_beam
                            )

        grad_z_int_z = (
                         consts.euler_gamma + special.gamma(0.1) *
                         (1. - special.gammainc(0.1, r_arg))
                         + np.log(r_arg)
                        ) * (
                            1. - (z_arg/self.z_beam)**2
                           )

        grad_r_int_z *= self.kick_coeff
        grad_z_int_z *= self.kick_coeff


        kick_pr[ptcl_z_within_beam] = ptcl_data.qOc[ptcl_z_within_beam] * grad_r_int_z
        kick_pz[ptcl_z_within_beam] = ptcl_data.qOc[ptcl_z_within_beam] * grad_z_int_z

        return kick_pz, kick_pr


    def compute_az(self, ptcl_data, beam_pos):

        psi = np.zeros(np.shape(ptcl_data.pz))

        # find particles inside the beam length
        ptcl_z_within_beam = np.where(np.abs(ptcl_data.z - beam_pos) < self.z_beam)
        r_arg = ptcl_data.r[ptcl_z_within_beam]**2/(2.*self.r_beam**2)
        z_arg = (ptcl_data.z[ptcl_z_within_beam] - beam_pos)**2/(self.z_beam**2)

        psi[ptcl_z_within_beam] = (
                                consts.euler_gamma + special.gamma(0.1) *
                                (1 - special.gammainc(0.1, r_arg)) + np.log(r_arg)
                                  ) * (1 - z_arg)

        psi[ptcl_z_within_beam] *= self.kick_coeff

        return ptcl_data.qOc * psi

    def compute_gamma_mc(self, ptcl_data, fld_data, beam_pos):
        """
        Replicates the compute gamma function in the particle data,
         but including the external fields.
        :param ptcl_data:
        :param fld_data:
        """
        gammamc = np.sqrt(
                        (ptcl_data.pr -
                            fld_data.compute_Ar(ptcl_data.r, ptcl_data.z, ptcl_data.qOc))**2 +\
                        (ptcl_data.pz -
                            fld_data.compute_Az(ptcl_data.r, ptcl_data.z, ptcl_data.qOc) -
                            self.compute_az(ptcl_data, beam_pos))**2 +\
                        ptcl_data.ell**2/(ptcl_data.r**2) + (ptcl_data.mc)**2
                        )

        ptcl_data.gamma_mc = gammamc