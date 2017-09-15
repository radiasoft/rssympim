"""
Class for dumping field data to hdf5 data format.

Author: Stephen Webb
"""

import h5py
#from mpi4py import MPI

class field_io:

    def __init__(self, field_name, field_class):

        self.field_name = field_name

        #self.rank = MPI.COMM_WORLD.rank

        # Copy over the information required to re-assemble the fields

        self.kr = field_class.kr
        self.kz = field_class.kz
        self.mode_mass = field_class.mode_mass


    def dump_field(self, field_class, step_number):

        # Field data is global, so only one process needs to write its data
        #if self.rank == 0:

        file_name = self.field_name + '_' + str(step_number) + '.hdf5'

        dump_file = h5py.File(file_name, 'w')

        dump_file.attrs['R'] = field_class.domain_R
        dump_file.attrs['L'] = field_class.domain_L

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

        dump_file.close()
