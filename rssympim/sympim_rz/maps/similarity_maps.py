#
# Class for handling deposition of sources to the field amplitudes and
# rotations of the particle coordinates.
#

class similarity_maps():
    """Class for handling deposition of sources to fields and rotations of the particle coordinates"""

    def S_r(self, field_data, ptcl_data):
        """
        Compute the effects of the r similarity map on the fields and particles
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        field_data: data.field_data
            field data class object
        """

        kick_z, kick_r, kick_Q0, kick_Qomega = \
            field_data.compute_S_r_kick(ptcl_data.r, ptcl_data.z, ptcl_data.qOc)

        # Update the particle momenta
        ptcl_data.pz -= kick_z
        ptcl_data.pr -= kick_r

        field_data.delta_P_dc    -= kick_Q0
        field_data.delta_P_omega -= kick_Qomega


    def S_r_inverse(self, field_data, ptcl_data):
        """
        Compute the effects of the r-inverse similarity map on the fields and particles
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        field_data: data.field_data
            field data class object
        """

        kick_z, kick_r, kick_Q0, kick_Qomega = \
            field_data.compute_S_r_kick(ptcl_data.r, ptcl_data.z, ptcl_data.qOc)

        # Update the particle momenta
        ptcl_data.pz += kick_z
        ptcl_data.pr += kick_r

        field_data.delta_P_dc    += kick_Q0
        field_data.delta_P_omega += kick_Qomega


    def S_z(self, field_data, ptcl_data):
        """
        Compute the effects of the z similarity map on the fields and particles
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        field_data: data.field_data
            field data class object
        """

        kick_z, kick_r, kick_Q0, kick_Qomega = \
            field_data.compute_S_z_kick(ptcl_data.r, ptcl_data.z, ptcl_data.qOc)

        # Update the particle momenta
        ptcl_data.pz -= kick_z
        ptcl_data.pr -= kick_r

        field_data.delta_P_dc    -= kick_Q0
        field_data.delta_P_omega -= kick_Qomega


    def S_z_inverse(self, field_data, ptcl_data):
        """
        Compute the effects of the z-inverse similarity map on the fields and particles
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        field_data: data.field_data
            field data class object
        """

        kick_z, kick_r, kick_Q0, kick_Qomega = \
            field_data.compute_S_z_kick(ptcl_data.r, ptcl_data.z, ptcl_data.qOc)

        # Update the particle momenta
        ptcl_data.pz += kick_z
        ptcl_data.pr += kick_r

        field_data.delta_P_dc    += kick_Q0
        field_data.delta_P_omega += kick_Qomega