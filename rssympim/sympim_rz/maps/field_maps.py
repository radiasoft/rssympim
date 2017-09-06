
import numpy as np

class field_maps:

    def __init__(self, _field_data, _dt):

        kr = _field_data.kr
        kz = _field_data.kz
        mode_mass = _field_data.mode_mass
        n_modes_r = self.shape(kr)[0]
        n_modes_z = self.shape(kz)[0]

        self.dt = _dt

        omega = np.sqrt(np.einsum('i,j->ij', kz*kz, kr*kr))
        M_omega = mode_mass*omega

        self.phase_advance = omega*_dt
        self.rotation_matrices = np.zeros((self.n_modes[0], self.n_modes[1],2,2))
        self.half_for_rot_mat = np.zeros((self.n_modes[0], self.n_modes[1],2,2))
        self.half_bac_rot_mat = np.zeros((self.n_modes[0], self.n_modes[1],2,2))

        for idx_1 in range(0, n_modes_z):
            for idx_2 in range(0, n_modes_r):

                # Compute the full rotation matrices
                self.rotation_matrices[idx_1,idx_2,0,0] = \
                    np.cos(self.phase_advance[idx_1,idx_2])
                self.rotation_matrices[idx_1,idx_2,1,0] = \
                    np.sin(self.phase_advance[idx_1,idx_2])/M_omega[idx_1,idx_2]
                self.rotation_matrices[idx_1,idx_2,0,1] = \
                    -np.sin(self.phase_advance[idx_1,idx_2])*M_omega[idx_1,idx_2]
                self.rotation_matrices[idx_1,idx_2,1,1] =\
                    np.cos(self.phase_advance[idx_1,idx_2])

                # Compute the half rotation matrices and their inverses

                self.half_for_rot_mat[idx_1, idx_2, 0,0] = \
                    np.cos(.5*self.phase_advance[idx_1, idx_2])
                self.half_for_rot_mat[idx_1, idx_2, 1,0] = \
                    np.sin(.5*self.phase_advance[idx_1,idx_2])/M_omega[idx_1,idx_2]
                self.half_for_rot_mat[idx_1, idx_2, 0,1] = \
                    -np.sin(.5*self.phase_advance[idx_1,idx_2])*M_omega[idx_1,idx_2]
                self.half_for_rot_mat[idx_1, idx_2, 1,1] =\
                    np.cos(.5*self.phase_advance[idx_1,idx_2])

                self.half_bac_rot_mat[idx_1,idx_2,0,0] = \
                    np.cos(-.5*self.phase_advance[idx_1,idx_2])
                self.half_bac_rot_mat[idx_1,idx_2,1,0] = \
                    np.sin(-.5*self.phase_advance[idx_1,idx_2])/M_omega[idx_1,idx_2]
                self.half_bac_rot_mat[idx_1,idx_2,0,1] = \
                    -np.sin(-.5*self.phase_advance[idx_1,idx_2])*M_omega[idx_1,idx_2]
                self.half_bac_rot_mat[idx_1,idx_2,1,1] = \
                    np.cos(-.5*self.phase_advance[idx_1,idx_2])


    def advance_forward(self, field_data):
        """
        Advance the field data by a full time step.
        :param field_data:
        :return:
        """

        field_data.omega_coords = np.einsum('ijkl, ijl -> ijk',
                                           self.rotation_matrices, field_data.mode_coords)

        field_data.dc_coords[:,1] += field_data.dc_coords[:,0]/field_data.mode_mass*self.dt


    def half_advance_forward(self, field_data):
        """
        Advance the field data by a half time step forward -- for the beginning of a simulation
        :param field_data:
        :return:
        """

        field_data.omega_coords = np.einsum('ijkl, ijl -> ijk',
                                           self.half_for_rot_mat, field_data.mode_coords)

        field_data.dc_coords[:,1] += 0.5*field_data.dc_coords[:,0]/field_data.mode_mass*self.dt


    def half_advance_back(self, field_data):
        """
        Advance the field data by a half time step backward -- for the end of a simulation
        :param field_data:
        :return:
        """

        field_data.omega_coords = np.einsum('ijkl, ijl -> ijk',
                                           self.half_bac_rot_mat, field_data.mode_coords)

        field_data.dc_coords[:,1] -= 0.5*field_data.dc_coords[:,0]/field_data.mode_mass*self.dt