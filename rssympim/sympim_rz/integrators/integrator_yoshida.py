from integrator import integrator

class integrator_y4:
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
        self.dt = dt

        # create the two second order integrators needed for the fourth order integrator.
        x1 = - 2.**(1./3.)/(2. - 2.**(1./3.))
        x2 = 1./(2. - 2.**(1./3.))
        self.integrator1 = integrator.integrator(x1*self.dt, fld_data)
        self.integrator2 = integrator.integrator(x2*self.dt, fld_data)

    def update(self, ptcl_data, fld_data):

        self.integrator2.update(ptcl_data, fld_data)
        self.integrator1.update(ptcl_data, fld_data)
        self.integrator2.update(ptcl_data, fld_data)


class integrator_y6:
    """
    Class for 6th order Yoshida integrator, constructing higher order integrators
    from lower order ones.

    Parameters
    ----------
    dt : float (seconds)
        time step for integration

    field_data: data.field_data
        field data class object
    """

    def __init__(self, dt, fld_data):
        self.dt = dt

        # create the two second order integrators needed for the fourth order integrator.
        y1 = - 2. ** (1. / 5.) / (2. - 2. ** (1. / 5.))
        y2 = 1. / (2. - 2. ** (1. / 5.))
        self.integrator1 = integrator_y4.integrator_y4(y1 * self.dt, fld_data)
        self.integrator2 = integrator_y4.integrator_y4(y2 * self.dt, fld_data)

    def update(self, ptcl_data, fld_data):
        self.integrator2.update(ptcl_data, fld_data)
        self.integrator1.update(ptcl_data, fld_data)
        self.integrator2.update(ptcl_data, fld_data)


class integrator_yn:
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
        z1 = - 2. ** (1. / (2.*n + 1.)) / (2. - 2. ** (1. / (2.*n + 1.)))
        z2 = 1. / (2. - 2. ** (1. / (2.*n + 1.)))

        if n-2 > 6:
            self.integrator1 = integrator_yn.integrator_yn(z1 * self.dt, fld_data, n-2)
            self.integrator2 = integrator_yn.integrator_yn(z2 * self.dt, fld_data, n-2)
        else:
            self.integrator1 = integrator_y6.integrator_y6(z1 * self.dt, fld_data)
            self.integrator2 = integrator_y6.integrator_y6(z2 * self.dt, fld_data)


    def update(self, ptcl_data, fld_data):

        self.integrator2.update(ptcl_data, fld_data)
        self.integrator1.update(ptcl_data, fld_data)
        self.integrator2.update(ptcl_data, fld_data)