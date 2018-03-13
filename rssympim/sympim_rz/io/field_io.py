"""
Class for dumping field data to hdf5 data format.

Author: Stephen Webb
"""

import h5py
from mpi4py import MPI
import numpy as np

class field_io(object):

    def __init__(self, field_name):

        self.field_name = field_name

        self.rank = MPI.COMM_WORLD.rank


    def dump_field(self, field_class, step_number):

        # Copy over the information required to re-assemble the fields

        self.kr = field_class.kr
        self.kz = field_class.kz
        self.mode_mass = field_class.mode_mass
        self.omega = field_class.omega

        # Field data is global, so only one process needs to write its data
        if self.rank == 0:

            file_name = self.field_name + '_' + str(step_number) + '.hdf5'

            dump_file = h5py.File(file_name, 'w')

            dump_file.attrs['R'] = field_class.domain_R
            dump_file.attrs['L'] = field_class.domain_L
            dump_file.attrs['dz'] = field_class.ptcl_width_z
            dump_file.attrs['dr'] = field_class.ptcl_width_r

            modep_dset = dump_file.create_dataset(
                'p_omega', data = field_class.omega_coords[:,:,0]
            )
            modeq_dset = dump_file.create_dataset(
                'q_omega', data = field_class.omega_coords[:,:,1]
            )

            modep_dset = dump_file.create_dataset(
                'p_dc', data = field_class.dc_coords[:,:,0]
            )
            modeq_dset = dump_file.create_dataset(
                'q_dc', data = field_class.dc_coords[:,:,1]
            )

            # All the other coefficients can be built from kr and kz
            kr_dset = dump_file.create_dataset(
                'kr', data = self.kr
            )
            kz_dset = dump_file.create_dataset(
                'kz', data = self.kz
            )
            mm_dset = dump_file.create_dataset(
                'mode_mass', data = self.mode_mass
            )

            om_dset = dump_file.create_dataset(
                'omega', data = self.omega
            )

            dump_file.close()


    def read_field(self, file_name):

        read_file = h5py.File(file_name, 'r',
                              driver=mpio, comm=self.comm)

        n_modes_r = np.shape(read_file.get('kr'))[0]
        n_modes_z = np.shape(read_file.get('kz'))[0]

        R = read_file.attrs['R']
        L = read_file.attrs['L']

        P_omega = np.array(read_file.get('p_omega'))
        Q_omega = np.array(read_file.get('q_omega'))

        P_dc = np.array(read_file.get('p_dc'))
        Q_dc = np.array(read_file.get('q_dc'))

        return n_modes_z, n_modes_r, L, R, P_omega, Q_omega, P_dc, Q_dc
