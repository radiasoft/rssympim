from rssympim.sympim_rz.maps import ptcl_maps, field_maps, similarity_maps

class integrator:
    """
    Class that contains integration functions for advancing particles and fields
    
    Parameters
    ----------
    dt : float (seconds)
        time step for integration
    
    field_data: data.field_data
        field data class object
    """

    def __init__(self, dt, fld_data):

        self.dt = dt
        self.sim_maps = similarity_maps.similarity_maps()
        self.ptcl_maps = ptcl_maps.ptcl_maps(dt)
        self.field_maps = field_maps.field_maps(fld_data, dt)


    def single_step_fields(self, field_data):
        """
        Advances the field coordinates a full step
        
        Parameters
        ----------
        field_data: data.field_data
            field data class object
        """

        # Update the fields then compute gamma to get a symplectic integrator
        self.field_maps.advance_forward(field_data)


    def single_step_ptcl(self, ptcl_data, field_data):
        """
        Advances the particle coordinates without advancing the fields
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        field_data: data.field_data
            field data class object
        """

        # always compute the new gamma_mc after the field map update
        ptcl_data.compute_gamma_mc(field_data)

        # Update sequence goes
        # M_ell S_r D_r S_r^-1 S_z D_z S_z^-1 S_r D_r S_r^-1 M_ell
        self.ptcl_maps.half_angular_momentum(ptcl_data)

        self.sim_maps.S_r(field_data, ptcl_data)
        self.ptcl_maps.half_drift_r(ptcl_data, field_data)
        self.sim_maps.S_r_inverse(field_data, ptcl_data)

        self.sim_maps.S_z(field_data, ptcl_data)
        self.ptcl_maps.drift_z(ptcl_data)
        self.sim_maps.S_z_inverse(field_data, ptcl_data)

        self.sim_maps.S_r(field_data, ptcl_data)
        self.ptcl_maps.half_drift_r(ptcl_data, field_data)
        self.sim_maps.S_r_inverse(field_data, ptcl_data)

        self.ptcl_maps.half_angular_momentum(ptcl_data)

        # Add the delta-P to each mode
        field_data.finalize_fields()


    def half_field_forward(self, field_data):
        """
        Advances the field coordinates one-half step forward
        
        Parameters
        ----------
        field_data: data.field_data
            field data class object
        """

        self.field_maps.half_advance_forward(field_data)

    def half_field_back(self, field_data):
        """
        Advances the field coordinates one-half step backwards
        
        Parameters
        ----------
        field_data: data.field_data
            field data class object
        """

        self.field_maps.half_advance_back(field_data)
        
    def second_order_step(self, ptcl_data, field_data):
        """
        A comlpete step with second order integration in time
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        field_data: data.field_data
            field data class object
        """
        
        
        # half step field forward
        self.field_maps.half_advance_forward(field_data)
        
        # always compute the new gamma_mc after the field map update
        ptcl_data.compute_gamma_mc(field_data)

        # Particle update sequence : M_ell S_r D_r S_r^-1 S_z D_z S_z^-1 S_r D_r S_r^-1 M_ell
        self.ptcl_maps.half_angular_momentum(ptcl_data)

        self.sim_maps.S_r(field_data, ptcl_data)
        self.ptcl_maps.half_drift_r(ptcl_data, field_data)
        self.sim_maps.S_r_inverse(field_data, ptcl_data)

        self.sim_maps.S_z(field_data, ptcl_data)
        self.ptcl_maps.drift_z(ptcl_data)
        self.sim_maps.S_z_inverse(field_data, ptcl_data)

        self.sim_maps.S_r(field_data, ptcl_data)
        self.ptcl_maps.half_drift_r(ptcl_data, field_data)
        self.sim_maps.S_r_inverse(field_data, ptcl_data)

        self.ptcl_maps.half_angular_momentum(ptcl_data)

        # Add the delta-P to each mode
        field_data.finalize_fields()
        
        # half step field
        self.field_maps.half_advance_forward(field_data)
    
    
    def setup_fourth_order(self, field_data):
        """
        Configures fourth-order Yoshida integration. Must be run before `fourth_order_step` can be called.
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        field_data: data.field_data
            field data class object
        """        
        
        # Create the new integrator w/ Yoshida coefficients
        self.x0 = -(2.**(1./3.)/(2.-2.**(1./3.)))
        self.x1 = (1./(2.-2.**(1./3.)))
        
        self.ptcl_forward = ptcl_maps.ptcl_maps(self.x1*self.dt)
        self.field_forward = field_maps.field_maps(field_data, self.x1*self.dt)
        
        self.ptcl_backward = ptcl_maps.ptcl_maps(self.x0*self.dt)
        self.field_backward = field_maps.field_maps(field_data, self.x0*self.dt)
        
        
    def fourth_order_step(self, ptcl_data, field_data):
        """
        A comlpete step using fourth-order Yoshida integration in time
        
        Parameters
        ----------
        ptcl_data : data.particle_data
            particle data class object
        field_data: data.field_data
            field data class object
        """
        
        #-----------------------------
        # 1. Forward one step of x1*dt
        #-----------------------------
        self.field_forward.half_advance_forward(field_data)
        
        # always compute the new gamma_mc after the field map update
        ptcl_data.compute_gamma_mc(field_data)

        # Particle update sequence : M_ell S_r D_r S_r^-1 S_z D_z S_z^-1 S_r D_r S_r^-1 M_ell
        self.ptcl_forward.half_angular_momentum(ptcl_data)

        self.sim_maps.S_r(field_data, ptcl_data)
        self.ptcl_forward.half_drift_r(ptcl_data, field_data)
        self.sim_maps.S_r_inverse(field_data, ptcl_data)

        self.sim_maps.S_z(field_data, ptcl_data)
        self.ptcl_forward.drift_z(ptcl_data)
        self.sim_maps.S_z_inverse(field_data, ptcl_data)

        self.sim_maps.S_r(field_data, ptcl_data)
        self.ptcl_forward.half_drift_r(ptcl_data, field_data)
        self.sim_maps.S_r_inverse(field_data, ptcl_data)

        self.ptcl_forward.half_angular_momentum(ptcl_data)

        # Add the delta-P to each mode
        field_data.finalize_fields()
        
        # half step field
        self.field_forward.half_advance_forward(field_data)
        
        
        #-----------------------------
        # 2. Backward one step of x0*dt
        #-----------------------------
        self.field_backward.half_advance_forward(field_data)
        
        # always compute the new gamma_mc after the field map update
        ptcl_data.compute_gamma_mc(field_data)

        # Particle update sequence : M_ell S_r D_r S_r^-1 S_z D_z S_z^-1 S_r D_r S_r^-1 M_ell
        self.ptcl_backward.half_angular_momentum(ptcl_data)

        self.sim_maps.S_r(field_data, ptcl_data)
        self.ptcl_backward.half_drift_r(ptcl_data, field_data)
        self.sim_maps.S_r_inverse(field_data, ptcl_data)

        self.sim_maps.S_z(field_data, ptcl_data)
        self.ptcl_backward.drift_z(ptcl_data)
        self.sim_maps.S_z_inverse(field_data, ptcl_data)

        self.sim_maps.S_r(field_data, ptcl_data)
        self.ptcl_backward.half_drift_r(ptcl_data, field_data)
        self.sim_maps.S_r_inverse(field_data, ptcl_data)

        self.ptcl_backward.half_angular_momentum(ptcl_data)

        # Add the delta-P to each mode
        field_data.finalize_fields()
        
        # half step field
        self.field_backward.half_advance_forward(field_data)        
        
        
        #-----------------------------
        # 3. Forward one step of x1*dt
        #-----------------------------
        self.field_forward.half_advance_forward(field_data)
        
        # always compute the new gamma_mc after the field map update
        ptcl_data.compute_gamma_mc(field_data)

        # Particle update sequence : M_ell S_r D_r S_r^-1 S_z D_z S_z^-1 S_r D_r S_r^-1 M_ell
        self.ptcl_forward.half_angular_momentum(ptcl_data)

        self.sim_maps.S_r(field_data, ptcl_data)
        self.ptcl_forward.half_drift_r(ptcl_data, field_data)
        self.sim_maps.S_r_inverse(field_data, ptcl_data)

        self.sim_maps.S_z(field_data, ptcl_data)
        self.ptcl_forward.drift_z(ptcl_data)
        self.sim_maps.S_z_inverse(field_data, ptcl_data)

        self.sim_maps.S_r(field_data, ptcl_data)
        self.ptcl_forward.half_drift_r(ptcl_data, field_data)
        self.sim_maps.S_r_inverse(field_data, ptcl_data)

        self.ptcl_forward.half_angular_momentum(ptcl_data)

        # Add the delta-P to each mode
        field_data.finalize_fields()
        
        # half step field
        self.field_forward.half_advance_forward(field_data)        
