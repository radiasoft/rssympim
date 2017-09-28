
import numpy as np

class field_maps:
    """
    Class for handling field updates and rotations
        
    Parameters
    ----------
    field_data: data.field_data
        field data class object
    _dt: float (second)
        Step size for integration and motion  
    
    """

    def __init__(self, fld_data, _dt):

        kr = fld_data.kr
        kz = fld_data.kz
        mode_mass = fld_data.mode_mass

        n_modes_r = np.shape(kr)[0]
        n_modes_z = np.shape(kz)[0]

        omega = fld_data.omega

        M_omega = mode_mass*omega

        self.dt = _dt
        self.phase_advance = omega*_dt

        self.rotation_matrices = np.zeros((n_modes_z, n_modes_r,2,2))
        self.half_for_rot_mat  = np.zeros((n_modes_z, n_modes_r,2,2))
        self.half_bac_rot_mat  = np.zeros((n_modes_z, n_modes_r,2,2))

        for idx_z in range(0, n_modes_z):
            for idx_r in range(0, n_modes_r):

                # Compute the full rotation matrices
                self.rotation_matrices[idx_z,idx_r,0,0] = \
                    np.cos(self.phase_advance[idx_z,idx_r])
                self.rotation_matrices[idx_z,idx_r,1,0] = \
                    np.sin(self.phase_advance[idx_z,idx_r])/M_omega[idx_z,idx_r]
                self.rotation_matrices[idx_z,idx_r,0,1] = \
                    -np.sin(self.phase_advance[idx_z,idx_r])*M_omega[idx_z,idx_r]
                self.rotation_matrices[idx_z,idx_r,1,1] =\
                    np.cos(self.phase_advance[idx_z,idx_r])

                # Compute the half rotation matrices and their inverses

                self.half_for_rot_mat[idx_z, idx_r, 0,0] = \
                    np.cos(.5*self.phase_advance[idx_z, idx_r])
                self.half_for_rot_mat[idx_z, idx_r, 1,0] = \
                    np.sin(.5*self.phase_advance[idx_z,idx_r])/M_omega[idx_z,idx_r]
                self.half_for_rot_mat[idx_z, idx_r, 0,1] = \
                    -np.sin(.5*self.phase_advance[idx_z,idx_r])*M_omega[idx_z,idx_r]
                self.half_for_rot_mat[idx_z, idx_r, 1,1] =\
                    np.cos(.5*self.phase_advance[idx_z,idx_r])

                self.half_bac_rot_mat[idx_z,idx_r,0,0] = \
                    np.cos(-.5*self.phase_advance[idx_z,idx_r])
                self.half_bac_rot_mat[idx_z,idx_r,1,0] = \
                    np.sin(-.5*self.phase_advance[idx_z,idx_r])/M_omega[idx_z,idx_r]
                self.half_bac_rot_mat[idx_z,idx_r,0,1] = \
                    -np.sin(-.5*self.phase_advance[idx_z,idx_r])*M_omega[idx_z,idx_r]
                self.half_bac_rot_mat[idx_z,idx_r,1,1] = \
                    np.cos(-.5*self.phase_advance[idx_z,idx_r])


    def advance_forward(self, field_data):
        """
        Advance the field data by a full time step.
        
        Parameters
        ----------
        field_data: data.field_data
            field data class object
        """

        field_data.omega_coords = np.einsum('ijkl, ijl -> ijk',
                                           self.rotation_matrices, field_data.omega_coords)

        field_data.dc_coords[:,:,1] += field_data.dc_coords[:,:,0]/field_data.mode_mass*self.dt


    def half_advance_forward(self, field_data):
        """
        Advance the field data by a half time step forward -- for the beginning of a simulation
        
        Parameters
        ----------
        field_data: data.field_data
            field data class object
        """

        field_data.omega_coords = np.einsum('ijkl, ijl -> ijk',
                                           self.half_for_rot_mat, field_data.omega_coords)

        field_data.dc_coords[:,:,1] += 0.5*field_data.dc_coords[:,:,0]/field_data.mode_mass*self.dt


    def half_advance_back(self, field_data):
        """
        Advance the field data by a half time step backward -- for the end of a simulation
        
        Parameters
        ----------
        field_data: data.field_data
            field data class object
        """

        field_data.omega_coords = np.einsum('ijkl, ijl -> ijk',
                                           self.half_bac_rot_mat, field_data.omega_coords)

        field_data.dc_coords[:,:,1] -= 0.5*field_data.dc_coords[:,:,0]/field_data.mode_mass*self.dt