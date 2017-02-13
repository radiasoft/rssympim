
import numpy as np

class field_maps:

    def __init__(self, _frequencies, _dt):

        self.phase_advance = _frequencies*_dt
        self.n_modes = np.shape(_frequencies)
        self.rotation_matrices = np.zeros((self.n_modes[0], self.n_modes[1],2,2))
        self.half_for_rot_mat = np.zeros((self.n_modes[0], self.n_modes[1],2,2))
        self.half_bac_rot_mat = np.zeros((self.n_modes[0], self.n_modes[1],2,2))

        for idx_1 in range(0, self.n_modes[0]):
            for idx_2 in range(0, self.n_modes[1]):

                # Compute the full rotation matrices
                self.rotation_matrices[idx_1,idx_2,0,0] = \
                    np.cos(self.phase_advance[idx_1,idx_2])
                self.rotation_matrices[idx_1,idx_2,0,1] = \
                    -np.sin(self.phase_advance[idx_1,idx_2])/_frequencies[idx_1,idx_2]
                self.rotation_matrices[idx_1,idx_2,1,0] = \
                    np.sin(self.phase_advance[idx_1,idx_2])*_frequencies[idx_1,idx_2]
                self.rotation_matrices[idx_1,idx_2,1,1] =\
                    np.cos(self.phase_advance[idx_1,idx_2])

                # Compute the half rotation matrices and their inverses

                self.half_for_rot_mat[idx_1, idx_2, 0,0] = \
                    np.cos(self.phase_advance[idx_1, idx_2]*.5)
                self.half_for_rot_mat[idx_1, idx_2, 0,1] = \
                    -np.sin(self.phase_advance[idx_1,idx_2]*.5)/_frequencies[idx_1,idx_2]
                self.half_for_rot_mat[idx_1, idx_2, 1,0] = \
                    np.sin(self.phase_advance[idx_1,idx_2]*.5)*_frequencies[idx_1,idx_2]
                self.half_for_rot_mat[idx_1, idx_2, 1,1] =\
                    np.cos(self.phase_advance[idx_1,idx_2]*.5)

                self.half_bac_rot_mat[idx_1,idx_2,0,0] = \
                    np.cos(-.5*self.phase_advance[idx_1,idx_2])
                self.half_bac_rot_mat[idx_1,idx_2,0,1] = \
                    -np.sin(-.5*self.phase_advance[idx_1,idx_2])/_frequencies[idx_1,idx_2]
                self.half_bac_rot_mat[idx_1,idx_2,1,0] = \
                    np.sin(-.5*self.phase_advance[idx_1,idx_2])*_frequencies[idx_1,idx_2]
                self.half_bac_rot_mat[idx_1,idx_2,1,1] = \
                    np.cos(-.5*self.phase_advance[idx_1,idx_2])


    def advance_forward(self, field_data):
        """
        Advance the field data by a full time step.
        :param field_data:
        :return:
        """

        field_data.mode_coords = np.einsum('ijkl, ijl -> ijk',
                                           self.rotation_matrices, field_data.mode_coords)

        #for idx in range(0,self.n_modes):
        #    field_data.mode_coords[idx] = np.dot(self.rotation_matrices[idx],
        #                               field_data.mode_coords[idx])


    def half_advance_forward(self, field_data):
        """
        Advance the field data by a half time step forward -- for the beginning of a simulation
        :param field_data:
        :return:
        """

        field_data.mode_coords = np.einsum('ijkl, ijl -> ijk',
                                           self.half_for_rot_mat, field_data.mode_coords)

        #for idx in range(0,self.n_modes):
        #    field_data.mode_coords[idx] = np.dot(self.half_for_rot_mat[idx],
        #                               field_data.mode_coords[idx])


    def half_advance_back(self, field_data):
        """
        Advance the field data by a half time step backward -- for the end of a simulation
        :param field_data:
        :return:
        """

        field_data.mode_coords = np.einsum('ijkl, ijl -> ijk',
                                           self.half_bac_rot_mat, field_data.mode_coords)

        #for idx in range(0,self.n_modes):
        #    field_data.mode_coords[idx] = np.dot(self.half_bac_rot_mat[idx],
        #                                    field_data.mode_coords[idx])