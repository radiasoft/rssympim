

class ptcl_maps:

    def __init__(self, _dt):

        self.dt = _dt
        self.halfdt = _dt/2.


    def drift_z(self, ptcl_data):
        """
        Particles drift in z for a full step
        :param ptcl_data:
        :return:
        """

        ptcl_data.z += (ptcl_data.pz/ptcl_data.gamma_mc) * self.dt


    def half_drift_z(self, ptcl_data):
        """
        Particles drift in z for a half step
        :param ptcl_data:
        :return:
        """

        ptcl_data.z += (ptcl_data.pz/ptcl_data.gamma_mc) * self.halfdt