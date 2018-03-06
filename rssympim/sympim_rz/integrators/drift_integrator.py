from rssympim.sympim_rz.maps import ptcl_maps, field_maps, similarity_drift_maps


class integrator:

    def __init__(self, dt, fld_data):

        self.dt = dt
        self.sim_maps = similarity_drift_maps.similarity_drift_maps()
        self.ptcl_maps = ptcl_maps.ptcl_maps(dt)
        self.field_maps = field_maps.field_maps(fld_data, dt)


    def single_step_fields(self, field_data):

        # Update the fields then compute gamma to get a symplectic integrator
        self.field_maps.advance_forward(field_data)


    def single_step_ptcl(self, ptcl_data, field_data):

        # always compute the new gamma_mc after the field map update
        ptcl_data.compute_gamma_mc(field_data)

        # Update sequence goes
        # M_ell A_r D_r A_r^-1 A_z D_z A_z^-1 A_r D_r A_r^-1 M_ell
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

        self.field_maps.half_advance_forward(field_data)


    def half_field_back(self, field_data):

        self.field_maps.half_advance_back(field_data)
