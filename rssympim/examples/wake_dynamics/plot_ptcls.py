from rssympim.sympim_rz.analysis import ptcl_analysis

file_name = 'wake_ptcls_0.hdf5'

analysis = ptcl_analysis.ptcl_analysis()

analysis.open_file(file_name)
analysis.get_particle_data(file_name)
analysis.get_field_quantities('wake_flds_0.hdf5')

analysis.plot_particles('density.png')

analysis.close_file()