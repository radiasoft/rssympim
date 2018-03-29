from rssympim.sympim_rz.analysis import ptcl_analysis

file_name = './data/wake_ptcls_580.hdf5'

analysis = ptcl_analysis.ptcl_analysis()

analysis.open_file(file_name)
analysis.get_particle_data(file_name)
analysis.get_field_quantities('./data/wake_flds_800.hdf5')

analysis.plot_particles('density.png')
analysis.plot_z_pr_phase('z_pr.png')
analysis.plot_r_pz_phase('r_pz.png')
analysis.plot_z_phase('z_pz.png')
analysis.plot_r_phase('r_pr.png')

analysis.close_file()