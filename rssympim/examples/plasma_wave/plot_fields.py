from rssympim.sympim_rz.analysis import field_analysis

analysis = field_analysis.field_analysis()

file_name = 'wave_flds_0.hdf5'

analysis.open_file(file_name)

analysis.plot_Ez('Ez_test.png') #, rmax=.02, zmin=0.03, zmax=0.07)
analysis.plot_Er('Er_test.png') #, rmax=.02, zmin=0.03, zmax=0.07)

analysis.close_file()