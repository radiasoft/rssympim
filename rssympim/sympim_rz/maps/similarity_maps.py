#
# Class for handling deposition of sources to the field amplitudes and
# rotations of the particle coordinates.
#

class similarity_maps():


    def A_r(self, field_data, ptcl_data):
        """
        Compute the effects of the r similarity map on the fields and particles
        :param field_data:
        :param ptcl_data:
        :return: nothing
        """
        ptcl_data.pr += ptcl_data.qOc*\
                        field_data.compute_Ar(ptcl_data.r, ptcl_data.z)
        ptcl_data.pz += ptcl_data.qOc*\
                        field_data.compute_dFrdz(ptcl_data.r, ptcl_data.z)
        #field_data.delta_P += field_data.compute_dFrdQ(ptcl_data.r, ptcl_data.z)


    def A_r_inverse(self, field_data, ptcl_data):
        """
        Compute the effects of the r-inverse similarity map on the fields and
        particles
        :param field_data:
        :param ptcl_data:
        :return: nothing
        """
        ptcl_data.pr -= ptcl_data.qOc*\
                        field_data.compute_Ar(ptcl_data.r, ptcl_data.z)
        ptcl_data.pz -= ptcl_data.qOc*\
                        field_data.compute_dFrdz(ptcl_data.r, ptcl_data.z)
        #field_data.delta_P -= field_data.compute_dFrdQ(ptcl_data.r, ptcl_data.z)


    def A_z(self, field_data, ptcl_data):
        """
        Compute the effects of the z similarity map on the fields and particles
        :param field_data:
        :param ptcl_data:
        :return: nothing
        """
        ptcl_data.pr += ptcl_data.qOc*\
                        field_data.compute_dFzdr(ptcl_data.r, ptcl_data.z)
        ptcl_data.pz += ptcl_data.qOc*\
                        field_data.compute_Az(ptcl_data.r, ptcl_data.z)
        #field_data.delta_P += field_data.compute_dFzdQ(ptcl_data.r, ptcl_data.z)


    def A_z_inverse(self, field_data, ptcl_data):
        """
        Compute the effects of the z-inverse similarity map on the fields and
        particles
        :param field_data:
        :param ptcl_data:
        :return: nothing
        """
        ptcl_data.pr -= ptcl_data.qOc*\
                        field_data.compute_dFzdr(ptcl_data.r, ptcl_data.z)
        ptcl_data.pz -= ptcl_data.qOc*\
                        field_data.compute_Az(ptcl_data.r, ptcl_data.z)
        #field_data.delta_P -= field_data.compute_dFzdQ(ptcl_data.r, ptcl_data.z)



