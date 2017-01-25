
import numpy as np

class field_maps:

    def __init__(self, _frequencies, _dt):

        self.phase_advance = _frequencies*_dt
        self.n_modes = np.shape(_frequencies)[0]
        self.rotation_matrices = np.zeros((self.n_modes,2,2))
        self.half_for_rot_mat = np.zeros((self.n_modes,2,2))
        self.half_bac_rot_mat = np.zeros((self.n_modes,2,2))

        for idx in range(0, self.n_modes):

            # Compute the full rotation matrices
            self.rotation_matrices[idx][0,0]= \
                np.cos(self.phase_advance[idx])
            self.rotation_matrices[idx][0,1]= \
                _frequencies[idx]*np.sin(self.phase_advance[idx])
            self.rotation_matrices[idx][1,0]= \
                -np.sin(self.phase_advance[idx])/_frequencies[idx]
            self.rotation_matrices[idx][1,1]=\
                np.cos(self.phase_advance[idx])

            # Compute the half rotation matrices and their inverses
            self.half_for_rot_mat[idx] = \
                np.sqrt(self.rotation_matrices[idx])
            self.half_bac_rot_mat[idx] = \
                np.linalg.inv(self.half_for_rot_mat[idx])


    def advance_forward(self, field_data):

        for idx in range(0,self.n_modes):
            field_data.mode_coords[idx] = np.dot(self.rotation_matrices[idx],
                                       field_data.mode_coords[idx])


    def half_advance_forward(self, field_data):

        for idx in range(0,self.n_modes):
            field_data.mode_coords[idx] = np.dot(self.half_for_rot_mat[idx],
                                       field_data.mode_coords[idx])


    def half_advance_back(self, field_data):

        for idx in range(0,self.n_modes):
            field_data.mode_coords[idx] = np.dot(self.half_bac_rot_mat[idx],
                                            field_data.mode_coords[idx])