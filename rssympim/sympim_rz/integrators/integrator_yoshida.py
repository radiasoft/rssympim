"""
Uses the method due to Yoshida to generate arbitrary even-order integrators,
with the 4th order integrator hard-coded since it is likely to be the most useful.


H. Yoshida, "Construction of higher order symplectic integrators",
 Phys. Lett. A, 150, 5,6,7 (1990)
"""

from rssympim.sympim_rz.integrators.integrator import integrator

class integrator_y4(object):
    """
    Class for 4th order Yoshida integrator, constructing higher order integrators
    from lower order ones.

    Parameters
    ----------
    dt : float (seconds)
        time step for integration

    field_data: data.field_data
        field data class object
    """

    def __init__(self, dt, fld_data):

        # create the two second order integrators needed for the fourth order integrator.
        x0 = - 2.**(1./3.)/(2. - 2.**(1./3.))
        x1 = 1./(2. - 2.**(1./3.))

        self.integrator0 = integrator(x0 * dt, fld_data)
        self.integrator1 = integrator(x1 * dt, fld_data)


    def update(self, ptcl_data, fld_data):

        self.integrator1.update(ptcl_data, fld_data)
        self.integrator0.update(ptcl_data, fld_data)
        self.integrator1.update(ptcl_data, fld_data)


class integrator_yn(object):
    """
    Class for nth order Yoshida integrator, constructing higher order integrators
    from lower order ones.

    Note that this uses object recursion, which may end up being very slow.

    Algorithmically, an order-n integrator will be 3x slower than an order-(n-2)
    integrator, therefore it will take 3^(n-2) times longer than a second order
    integrator.

    Parameters
    ----------
    dt : float (seconds)
        time step for integration

    field_data: data.field_data
        field data class object

    n: the value of n, must be an even integer
    """

    def __init__(self, dt, fld_data, n):

        if n%2 != 0:
            print ' the order of the integrator must be even'
            Exception

        self.dt = dt

        # create the two second order integrators needed for the fourth order integrator.
        a = 2.**(1./(2.*n+1.))
        z0 = - a / (2. - a)
        z1 = 1. / (2. - a)

        if n-2 > 4:
            self.integrator0 = integrator_yn(z0 * dt, fld_data, n-2)
            self.integrator1 = integrator_yn(z1 * dt, fld_data, n-2)
        else:
            self.integrator0 = integrator_y4(z0 * dt, fld_data)
            self.integrator1 = integrator_y4(z1 * dt, fld_data)


    def update(self, ptcl_data, fld_data):

        self.integrator1.update(ptcl_data, fld_data)
        self.integrator0.update(ptcl_data, fld_data)
        self.integrator1.update(ptcl_data, fld_data)