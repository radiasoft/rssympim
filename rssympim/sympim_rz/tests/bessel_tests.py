from rssympim.sympim_rz.data import field_data
import numpy as np

# The field data class has a special implementation of J_0. This tests those implementations.

my_field_data = field_data.field_data(1.,2.,10,10)

x_test = np.arange(0.,10.,1.)
j0_return = my_field_data.my_j0(x_test)

j0_answer = np.array([ 1., 0.76519769, 0.22389078, -0.26005195, -0.39714981, -0.17759677,
                    0.15064526, 0.30007927, 0.17165081, -0.09033361])

int_j0_return = my_field_data.int_my_j0(x_test)

int_j0_answer = np.array([0., 0.91973, 1.42577, 1.38757, 1.02473, 0.715312, 0.706221,
                          0.95464, 1.21075, 1.25227])

# Test for relative errors greater than a part in 10^2
tol = 1.e-2

test_j0 = np.abs(j0_return - j0_answer)
test_int_j0 = np.abs(int_j0_return - int_j0_answer)

msg = "Test results: \n"

j0_results = test_j0 > tol
int_j0_results = test_int_j0 > tol

failed = False

if j0_results.any():
    msg += "j0 function fails \n"
    failed = True

if int_j0_results.any():
    msg += "int_j0 function fails \n"
    failed = True

if failed:
    print msg

else:
    msg += "All tests pass! Good job!"
    print msg