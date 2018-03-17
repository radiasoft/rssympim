
class ptcl_maps:
    """
    Class for handling particle motion and phase-space coordinate updates
        
    Parameters
    ----------
    _dt: float (second)
        Step size for integration and motion  
    
    """

    def __init__(self, _dt):

        self.dt = _dt
        self.halfdt = _dt/ 2.


    def drift_r(self, ptcl_data, fld_data):
        """
        Particles drift in r for a full step
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        """

        ptcl_data.r += (ptcl_data.pr / ptcl_data.gamma_mc) * self.dt
        # Have to wrap particles around if they passed through the central axis
        ptcl_data.r_boundaries(fld_data)


    def drift_z(self, ptcl_data):
        """
        Particles drift in z for a full step
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        """

        ptcl_data.z += (ptcl_data.pz / ptcl_data.gamma_mc) * self.dt


    def angular_momentum(self, ptcl_data):
        """
        Particles get an angular momentum kick
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        """

        ptcl_data.pr += self.dt * \
                        ptcl_data.ell*ptcl_data.ell / (ptcl_data.gamma_mc * (ptcl_data.r ** 3))


    def half_drift_r(self, ptcl_data, fld_data):
        """
        Particles drift in r for a half step
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        """

        ptcl_data.r += (ptcl_data.pr / ptcl_data.gamma_mc) * self.halfdt
        # Have to wrap particles around if they passed through the central axis
        ptcl_data.r_boundaries(fld_data)


    def half_drift_z(self, ptcl_data):
        """
        Particles drift in z for a half step
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        """

        ptcl_data.z += (ptcl_data.pz / ptcl_data.gamma_mc) * self.halfdt


    def half_angular_momentum(self, ptcl_data):
        """
        Particles get half an angular momentum kick
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        """

        ptcl_data.pr += self.halfdt * \
                        ptcl_data.ell*ptcl_data.ell / (ptcl_data.gamma_mc * (ptcl_data.r ** 3))
