from rssympim.sympim_1d.maps import ptcl_maps, field_maps, similarity_maps


class integrator:

    def __init__(self, dt, frequencies):
        self.dt = dt
        self.sim_maps = similarity_maps.similarity_maps()
        self.ptcl_maps = ptcl_maps.ptcl_maps(dt)
        self.field_maps = field_maps.field_maps(frequencies, dt)


    def field_update(self, field_data):

        # Update the fields then compute gamma to get a symplectic integrator
        self.field_maps.advance_forward(field_data)


    def particle_update(self, ptcl_data, field_data):

        # always compute the new gamma_mc after the field map update
        ptcl_data.compute_gamma_mc(field_data)

        self.sim_maps.A_z(field_data, ptcl_data)
        self.ptcl_maps.drift_z(ptcl_data)
        self.sim_maps.A_z_inverse(field_data, ptcl_data)


    def finalize_fields(self, field_data):

        # Add the delta-P to each mode
        field_data.finalize_fields()


    def half_field_forward(self, field_data):

        self.field_maps.half_advance_forward(field_data)


    def half_field_back(self, field_data):

        self.field_maps.half_advance_back(field_data)
