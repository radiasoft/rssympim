"""
Class for dumping particle data to hdf5 data format.

Author: Stephen Webb
"""

import h5py
from mpi4py import MPI

class field_io:

    def __init__(self, particle_name, particle_class):

        self.ptcl_name = particle_name

        self.rank = MPI.COMM_WORLD.rank

    def dump_ptcls(self, ptcl_class, step_number):

        # Particle data is local, and the h5 file must be put
        # together using MPI
        file_name = self.ptcl_name + '_' + str(step_number) + '.hdf5'

        dump_file = h5py.File(file_name, 'w',
                              driver='mpio', comm=MPI.COMM_WORLD)

        dump_file.attrs['charge'] = ptcl_class.charge
        dump_file.attrs['mass'] = ptcl_class.mass

        ptcl_pr = dump_file.create_dataset(
            'pr', data = ptcl_class.pr
        )
        ptcl_r = dump_file.create_dataset(
            'r', data = ptcl_class.r
        )
        ptcl_pz = dump_file.create_dataset(
            'pz', data = ptcl_class.pz
        )
        ptcl_z = dump_file.create_dataset(
            'z', data = ptcl_class.z
        )
        ptcl_pl = dump_file.create_dataset(
            'pl', data = ptcl_class.ell
        )
        ptcl_weight = dump_file.create_dataset(
            'weight', data = ptcl_class.weight
        )

        dump_file.close()