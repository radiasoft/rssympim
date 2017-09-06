#
# Class for handling deposition of sources to the field amplitudes and
# rotations of the particle coordinates.
#

class similarity_drift_maps():


    def A_r(self, field_data, ptcl_data):
        """
        Compute the effects of the r similarity map on the fields and particles
        :param field_data:
        :param ptcl_data:
        :return: nothing
        """

        deltap = field_data.compute_dFrdQ(ptcl_data.r, ptcl_data.z, ptcl_data.qOc)

        field_data.delta_P_dc    -= deltap[0]
        field_data.delta_P_omega -= deltap[1]



    def A_r_inverse(self, field_data, ptcl_data):
        """
        Compute the effects of the r-inverse similarity map on the fields and
        particles
        :param field_data:
        :param ptcl_data:
        :return: nothing
        """
        deltap = field_data.compute_dFrdQ(ptcl_data.r, ptcl_data.z, ptcl_data.qOc)

        field_data.delta_P_dc    += deltap[0]
        field_data.delta_P_omega += deltap[1]

    def A_z(self, field_data, ptcl_data):
        """
        Compute the effects of the z similarity map on the fields and particles
        :param field_data:
        :param ptcl_data:
        :return: nothing
        """
        deltap = field_data.compute_dFzdQ(ptcl_data.r, ptcl_data.z, ptcl_data.qOc)

        field_data.delta_P_dc    -= deltap[0]
        field_data.delta_P_omega -= deltap[1]

    def A_z_inverse(self, field_data, ptcl_data):
        """
        Compute the effects of the z-inverse similarity map on the fields and
        particles
        :param field_data:
        :param ptcl_data:
        :return: nothing
        """
        deltap = field_data.compute_dFzdQ(ptcl_data.r, ptcl_data.z, ptcl_data.qOc)

        field_data.delta_P_dc    += deltap[0]
        field_data.delta_P_omega += deltap[1]