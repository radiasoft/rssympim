"""
Class for dumping particle data to hdf5 data format.

Author: Stephen Webb
"""

import h5py
from mpi4py import MPI
import numpy as np

class particle_io(object):

    def __init__(self, particle_name, parallel_hdf5 = False):

        self.ptcl_name = particle_name

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size

        self.parallel = False
        if parallel_hdf5:
            self.parallel = True


    def dump_ptcls(self, ptcl_class, step_number):

        # Particle data is local, and the h5 file must be put
        # together using MPI
        file_name = self.ptcl_name + '_' + str(step_number) + '.hdf5'

        if self.parallel:

            dump_file = h5py.File(file_name, 'w',
                                  driver='mpio', comm=self.comm)

            dump_file.attrs['charge'] = ptcl_class.charge
            dump_file.attrs['mass'] = ptcl_class.mass
            dump_file.attrs['n_ptcls'] = np.shape(ptcl_class.pr)[0]

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

        elif self.rank == 0:
            dump_file = h5py.File(file_name, 'w')

            pr = np.zeros(np.shape(ptcl_class.pr))
            pz = np.zeros(np.shape(ptcl_class.pz))
            pl = np.zeros(np.shape(ptcl_class.ell))

            r = np.zeros(np.shape(ptcl_class.r))
            z = np.zeros(np.shape(ptcl_class.z))

            wgt = np.zeros(np.shape(ptcl_class.weight))

            pr[:] = ptcl_class.pr[:]
            pz[:] = ptcl_class.pz[:]
            pl[:] = ptcl_class.ell[:]

            r[:] = ptcl_class.r[:]
            z[:] = ptcl_class.z[:]

            wgt[:] = ptcl_class.weight[:]

            if self.size > 1:

                for sndr in range(1, self.size):

                    ptclclass = self.comm.recv(source=sndr, tag=11)

                    np.append(pr, ptclclass.pr)
                    np.append(pz, ptclclass.pz)
                    np.append(pl, ptclclass.ell)
                    np.append(r, ptclclass.r)
                    np.append(z, ptclclass.z)
                    np.append(wgt, ptclclass.weight)

            dump_file.attrs['charge'] = ptcl_class.charge
            dump_file.attrs['mass'] = ptcl_class.mass
            dump_file.attrs['n_ptcls'] = np.shape(ptcl_class.pr)[0]

            ptcl_pr = dump_file.create_dataset(
                'pr', data = pr
            )
            ptcl_r = dump_file.create_dataset(
                'r', data = r
            )
            ptcl_pz = dump_file.create_dataset(
                'pz', data = pz
            )
            ptcl_z = dump_file.create_dataset(
                'z', data = z
            )
            ptcl_pl = dump_file.create_dataset(
                'pl', data = pl
            )
            ptcl_weight = dump_file.create_dataset(
                'weight', data = wgt
            )

        else:
            # send the data to rank 0
            self.comm.send(ptcl_class, dest=0, tag=11)


    def read_ptcls(self, file_name):
        """
        Read in files from the hdf5 file file_name, a string.
        Designed to work in parallel to read in the particle
        data uniformly distributed across processes.
        :param file_name:
        :return: r, pr, z, pz, pl, weight, charge, mass
        """

        read_file = h5py.File(file_name, 'r',
                              driver=mpio, comm=self.comm)

        # Read in the particle data uniformly across processes

        n_ptcls = read_file.attrs['n_ptcls']

        ptcls_per_rank = n_ptcls/self.size


        start_index = self.rank*ptcls_per_rank

        # dump any remainder particles on the last rank
        if self.rank < self.size - 1:
            end_index = start_index + ptcls_per_rank
        else:
            end_index = -1

        r  = np.array(read_file.get('r')[start_index:end_index])
        z  = np.array(read_file.get('z')[start_index:end_index])

        pr = np.array(read_file.get('pr')[start_index:end_index])
        pz = np.array(read_file.get('pz')[start_index:end_index])
        pl = np.array(read_file.get('pz')[start_index:end_index])

        weight = np.array(read_file.get('r')[start_index:end_index])

        charge = read_file.attrs['charge']
        mass   = read_file.attrs['mass']

        return r, pr, z, pz, pl, weight, charge, mass