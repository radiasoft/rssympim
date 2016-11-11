import h5py
import numpy as np

class io:

    def __init__(self, _simulationname):

        self.sim_name = _simulationname


    def dump_data(self, _step, _ptcldata, _fielddata):

        filename = str(self.sim_name) + '_step_' + str(_step)+'.hdf5'
        my_file = h5py.File(filename,'w')

        # Store the particle data
        ptcl_grp = my_file.create_group('particles')

        n_ptcls = np.shape(_ptcldata.r)[0]

        dset_qs = ptcl_grp.create_dataset('coordinates',(n_ptcls,2),
                                         data=[_ptcldata.r, _ptcldata.z])
        dset_ps = ptcl_grp.create_dataset('momenta',(n_ptcls,2),
                                         data=[_ptcldata.pr, _ptcldata.pz])

        # Store the field data
        field_grp = my_file.create_group('fields')

        n_modes = np.shape(_fielddata.mode_coords)[0]

        dset_modes = field_grp.create_dataset('modes',(n_modes,2),
                                              data = _fielddata.mode_coords)

        my_file.close()