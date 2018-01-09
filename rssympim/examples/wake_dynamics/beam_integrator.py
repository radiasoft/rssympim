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

from rssympim.constants import constants as consts

class beam_integrator:

    def __init__(self, beam_radius, beam_length, beam_N, dt, fld_data):

        self.r_beam = beam_radius
        self.z_beam = beam_length

        self.dt = dt

        self.kick_parameter = -4.*np.pi*consts.electron_charge*beam_N

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
        ptcl_data.compute_gamma_mc(fld_data)

        # Update sequence goes
        # M_ell S_r D_r S_r^-1 S_z D_z S_z^-1 S_r D_r S_r^-1 M_ell
        self.ptcl_maps.half_angular_momentum(ptcl_data)

        self.sim_maps.S_r(fld_data, ptcl_data)
        self.ptcl_maps.half_drift_r(ptcl_data, fld_data)
        self.sim_maps.S_r_inverse(fld_data, ptcl_data)

        self.S_z_external(ptcl_data, beam_pos)
        self.sim_maps.S_z(fld_data, ptcl_data)
        self.ptcl_maps.drift_z(ptcl_data)
        self.sim_maps.S_z_inverse(fld_data, ptcl_data)
        self.S_z_inverse_external(ptcl_data, beam_pos)

        self.sim_maps.S_r(fld_data, ptcl_data)
        self.ptcl_maps.half_drift_r(ptcl_data, fld_data)
        self.sim_maps.S_r_inverse(fld_data, ptcl_data)

        self.ptcl_maps.half_angular_momentum(ptcl_data)

        self.phi_kick(ptcl_data, beam_pos, 0.5 * self.dt)

        # Add the delta-P to each mode
        fld_data.finalize_fields()


    ###
    #
    # The fields for the parabolic charge distribution are
    #
    # A_z = - n_0 e (r^2/2 - r^4/r0^2/4)    r < r0
    # A_z = - N_b e ln(r/r0)                r > r0
    #
    # The scalar potential is the same
    #
    ###

    def phi_kick(self, ptcl_data, beam_pos, dtau):
        """
        Applies the space charge kick due to the beam fields
        """

        kick_pr = np.zeros(np.shape(ptcl_data.pr)[0])
        kick_pz = np.zeros(np.shape(ptcl_data.pz)[0])

        # find particles inside the beam length
        ptcl_z_within_beam = np.where(np.abs(ptcl_data.z - beam_pos) < self.z_beam)

        # particles inside r0 get one kick, particles outside get another
        ptcl_r_within_beam = np.where(ptcl_data.r < self.r_beam)
        ptcl_r_without_beam = np.where(ptcl_data.r > self.r_beam)

        ptcls_in_beam = np.intersect1d(ptcl_z_within_beam, ptcl_r_within_beam)
        ptcls_out_beam = np.intersect1d(ptcl_z_within_beam, ptcl_r_without_beam)

        kick_pr[ptcls_in_beam] = -self.kick_parameter * \
                                 (.5 - .125 * ptcl_data.r[ptcls_in_beam] ** 2 / self.r_beam ** 2) * ptcl_data.r[
                                     ptcls_in_beam] * \
                                 (1 - (ptcl_data.z[ptcls_in_beam] - beam_pos) ** 2 / self.z_beam ** 2)
        kick_pr[ptcls_out_beam] = -self.kick_parameter * \
                                  (.5 - .125) * self.r_beam ** 2 / ptcl_data.r[ptcls_out_beam] * \
                                  (1 - (ptcl_data.z[ptcls_out_beam] - beam_pos) ** 2 / self.z_beam ** 2)

        kick_pz[ptcls_in_beam] = -self.kick_parameter * \
                                 (.5 - .125 * ptcl_data.r[ptcls_in_beam] ** 2 / self.r_beam ** 2) * ptcl_data.r[
                                                                                                        ptcls_in_beam] ** 2 * \
                                 (-(ptcl_data.z[ptcls_in_beam] - beam_pos) / self.z_beam ** 2)
        kick_pz[ptcls_out_beam] = -self.kick_parameter * \
                                  (.5 - .125) * self.r_beam ** 2 * np.log(ptcl_data.r[ptcls_out_beam] / self.r_beam) * \
                                  (-(ptcl_data.z[ptcls_out_beam] - beam_pos) / self.z_beam ** 2)

        kick_pz *= ptcl_data.qOc * dtau
        kick_pr *= ptcl_data.qOc * dtau

        ptcl_data.pz += kick_pz
        ptcl_data.pr += kick_pr

    def S_z_external(self, ptcl_data, beam_pos):
        """
        Applies the similarity transformation due to the beam fields
        """

        kick_pr = np.zeros(np.shape(ptcl_data.pr)[0])
        kick_pz = np.zeros(np.shape(ptcl_data.pz)[0])

        # find particles inside the beam length
        ptcl_z_within_beam = np.where(np.abs(ptcl_data.z - beam_pos) < self.z_beam)
        ptcl_z_without_beam = np.where(np.abs(ptcl_data.z - beam_pos) > self.z_beam)

        # particles inside r0 get one kick, particles outside get another
        ptcl_r_within_beam = np.where(ptcl_data.r < self.r_beam)
        ptcl_r_without_beam = np.where(ptcl_data.r > self.r_beam)

        #define 4 sectors of particles
        ptcls_in_beam = np.intersect1d(ptcl_z_within_beam, ptcl_r_within_beam)
        ptcls_out_beam = np.intersect1d(ptcl_z_without_beam, ptcl_r_without_beam)
        ptcls_out_z = np.intersect1d(ptcl_z_without_beam, ptcl_r_within_beam) #particles that are within rbeam but not zbeam
        ptcls_out_r = np.intersect1d(ptcl_z_within_beam, ptcl_r_without_beam) #particles that are within zbeam but not rbeam

        #Pre-compute z-dependent density
        ptcl_nz_in = (1 - (ptcl_data.z[ptcl_z_within_beam] - beam_pos) ** 2 / self.z_beam ** 2)
        
        #kick_pz is just a_z = phi in Quasistatic (v=c) approximation
        #note that those outside the z-boundary get no pz kick
        #wrap in try/excepts to validate case of zero particles
        try:
            kick_pz[ptcls_in_beam] = -self.kick_parameter*ptcl_nz_in* \
                                (ptcl_data.r[ptcls_in_beam]**2/4 - ptcl_data.r[ptcls_in_beam]**4/(16*self.r_beam**2))
        except ValueError:
            pass
        try:
            kick_pz[ptcls_out_beam] = -self.kick_parameter*ptcl_nz_in* \
                                ( (np.log(self.r_beam/ptcl_data.r[ptcls_out_beam])*self.r_beam**2/4) + (3*self.r_beam**2/16))
        except ValueError:
            pass                    
        
        #kick_pr is int[ (daz/dr) dz]
        #kick_pr requires distinguishing between r and z particles
        try:
            kick_pr[ptcls_in_beam] = -self.kick_parameter *\
                                 ptcl_data.z[ptcls_in_beam] * (1 - (2*ptcl_data.z[ptcls_in_beam]**2 -\
                                 6*ptcl_data.z[ptcls_in_beam]*(ptcl_data.z[ptcls_in_beam] - beam_pos)+\
                                 3*(ptcl_data.z[ptcls_in_beam] - beam_pos)**2)/(6*self.z_beam**2))*\
                                 (ptcl_data.r[ptcls_in_beam]/2 - ptcl_data.r[ptcls_in_beam]**3/(4*self.r_beam**2))
        except ValueError:
            pass

        try:
            kick_pr[ptcls_out_z] = -self.kick_parameter * self.z_beam*\
                                 (self.z_beam*self.r_beam**2/2*ptcl_data.r[ptcls_out_z])
        except ValueError:
            pass
        
        try:
            kick_pr[ptcls_out_r] = -self.kick_parameter *\
                                 ptcl_data.z[ptcls_out_r] * (1 - (2*ptcl_data.z[ptcls_out_r]**2 -\
                                 6*ptcl_data.z[ptcls_out_r]*(ptcl_data.z[ptcls_out_r] - beam_pos)+\
                                 3*(ptcl_data.z[ptcls_out_r] - beam_pos)**2)/(6*self.z_beam**2))*\
                                 (ptcl_data.r[ptcls_out_r]/2 - ptcl_data.r[ptcls_out_r]**3/(4*self.r_beam**2))
        except ValueError:
            pass
            
        try:                     
            kick_pr[ptcls_out_beam] = -self.kick_parameter *  self.z_beam*\
                                  (ptcl_data.r[ptcls_out_beam]/2 - ptcl_data.r[ptcls_out_beam]**3/(4*self.r_beam**2))
        except ValueError:
            pass


        kick_pz *= ptcl_data.qOc
        kick_pr *= ptcl_data.qOc

        ptcl_data.pz += kick_pz
        ptcl_data.pr += kick_pr

    def S_z_inverse_external(self, ptcl_data, beam_pos):
        """
        Applies the inverse similarity transformation due to the beam fields
        """

        kick_pr = np.zeros(np.shape(ptcl_data.pr)[0])
        kick_pz = np.zeros(np.shape(ptcl_data.pz)[0])

        # find particles inside the beam length
        ptcl_z_within_beam = np.where(np.abs(ptcl_data.z - beam_pos) < self.z_beam)
        ptcl_z_without_beam = np.where(np.abs(ptcl_data.z - beam_pos) > self.z_beam)

        # particles inside r0 get one kick, particles outside get another
        ptcl_r_within_beam = np.where(ptcl_data.r < self.r_beam)
        ptcl_r_without_beam = np.where(ptcl_data.r > self.r_beam)

        #define 4 sectors of particles
        ptcls_in_beam = np.intersect1d(ptcl_z_within_beam, ptcl_r_within_beam)
        ptcls_out_beam = np.intersect1d(ptcl_z_without_beam, ptcl_r_without_beam)
        ptcls_out_z = np.intersect1d(ptcl_z_without_beam, ptcl_r_within_beam) #particles that are within rbeam but not zbeam
        ptcls_out_r = np.intersect1d(ptcl_z_within_beam, ptcl_r_without_beam) #particles that are within zbeam but not rbeam

        #Pre-compute z-dependent density
        ptcl_nz_in = (1 - (ptcl_data.z[ptcl_z_within_beam] - beam_pos) ** 2 / self.z_beam ** 2)
        
        #kick_pz is just a_z = phi in Quasistatic (v=c) approximation
        #note that those outside the z-boundary get no pz kick
        #wrap in try/excepts to validate case of zero particles
        try:
            kick_pz[ptcls_in_beam] = -self.kick_parameter*ptcl_nz_in* \
                                (ptcl_data.r[ptcls_in_beam]**2/4 - ptcl_data.r[ptcls_in_beam]**4/(16*self.r_beam**2))
        except ValueError:
            pass
        try:
            kick_pz[ptcls_out_beam] = -self.kick_parameter*ptcl_nz_in* \
                                ( (np.log(self.r_beam/ptcl_data.r[ptcls_out_beam])*self.r_beam**2/4) + (3*self.r_beam**2/16))
        except ValueError:
            pass                    
        
        #kick_pr is int[ (daz/dr) dz]
        #kick_pr requires distinguishing between r and z particles
        try:
            kick_pr[ptcls_in_beam] = -self.kick_parameter *\
                                 ptcl_data.z[ptcls_in_beam] * (1 - (2*ptcl_data.z[ptcls_in_beam]**2 -\
                                 6*ptcl_data.z[ptcls_in_beam]*(ptcl_data.z[ptcls_in_beam] - beam_pos)+\
                                 3*(ptcl_data.z[ptcls_in_beam] - beam_pos)**2)/(6*self.z_beam**2))*\
                                 (ptcl_data.r[ptcls_in_beam]/2 - ptcl_data.r[ptcls_in_beam]**3/(4*self.r_beam**2))
        except ValueError:
            pass

        try:
            kick_pr[ptcls_out_z] = -self.kick_parameter * self.z_beam*\
                                 (self.z_beam*self.r_beam**2/2*ptcl_data.r[ptcls_out_z])
        except ValueError:
            pass
        
        try:
            kick_pr[ptcls_out_r] = -self.kick_parameter *\
                                 ptcl_data.z[ptcls_out_r] * (1 - (2*ptcl_data.z[ptcls_out_r]**2 -\
                                 6*ptcl_data.z[ptcls_out_r]*(ptcl_data.z[ptcls_out_r] - beam_pos)+\
                                 3*(ptcl_data.z[ptcls_out_r] - beam_pos)**2)/(6*self.z_beam**2))*\
                                 (ptcl_data.r[ptcls_out_r]/2 - ptcl_data.r[ptcls_out_r]**3/(4*self.r_beam**2))
        except ValueError:
            pass
            
        try:                     
            kick_pr[ptcls_out_beam] = -self.kick_parameter *  self.z_beam*\
                                  (ptcl_data.r[ptcls_out_beam]/2 - ptcl_data.r[ptcls_out_beam]**3/(4*self.r_beam**2))
        except ValueError:
            pass

        
        kick_pz *= ptcl_data.qOc
        kick_pr *= ptcl_data.qOc

        ptcl_data.pz -= kick_pz
        ptcl_data.pr -= kick_pr