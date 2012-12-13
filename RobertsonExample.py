# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <markdowncell>

# #Example for ICERM 2012 Workshop on Reproducibility in Computational and Experimental Mathematics#

# <markdowncell>

# ##Integrating the Robertson problem using a Python interface to DASSL##

# <markdowncell>

# The Robertson problem is a classical ordinary differential equation (ODE) used to test stiff ODE and differential-algebraic equation (DAE) solvers. DASSL, written by Linda Petzold in Fortran, is one example of a DAE solver. Josh Allen, a colleague of mine at MIT, wrote a Python interface to DASSL called PyDAS. In order to use it, and to plot the results, we need to import some Python modules.

# <codecell>

import numpy
import scipy.linalg
import pydas
import matplotlib.pyplot
import copy

# <markdowncell>

# The Robertson problem takes the following form as an ODE:
# 
# \begin{align}
# \dot{y}\_{0}(t) &= -4 \cdot 10^{-2}y\_{0}(t) + 10^{4}y\_{1}(t)y\_{2}(t), \\\
# \dot{y}\_{1}(t) &= 4 \cdot 10^{-2}y\_{0}(t) - 10^{4}y\_{1}(t)y\_{2}(t) - 3 \cdot 10^{7}y\_{1}^{2}(t), \\\
# \dot{y}\_{2}(t) &= 3 \cdot 10^{7}y\_{1}^{2}(t)
# \end{align}
# 
# However, we need to pose it instead as a DAE by moving the derivative terms into the right-hand side:
# 
# \begin{align}
# 0 &= -\dot{y}\_{0}(t) - 4 \cdot 10^{-2}y\_{0}(t) + 10^{4}y\_{1}(t)y\_{2}(t), \\\
# 0 &= -\dot{y}\_{1}(t) + 4 \cdot 10^{-2}y\_{0}(t) - 10^{4}y\_{1}(t)y\_{2}(t) - 3 \cdot 10^{7}y\_{1}^{2}(t), \\\
# 0 &= -\dot{y}\_{2}(t) + 3 \cdot 10^{7}y\_{1}^{2}(t)
# \end{align}
# 
# We define the DAE right-hand side (and its Jacobian matrix) using the `pydas.DASSL` class defined in `pydas`. This class should have two member functions:
# 
# - `residual`: defines the DAE right-hand side
# - `jacobian` (optional): defines Jacobian matrix information of the DAE right-hand side

# <codecell>

# Define the residual and optional Jacobian matrix
class Problem(pydas.DASSL):
    def residual(self, t, y, dydt):
        ode_rhs = numpy.asarray([-0.04 * y[0] + 1e4 * y[1] * y[2],
         0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
         3e7 * y[1] ** 2])
        res = ode_rhs - dydt
        return res, 0
    def jacobian(self, t, y, dydt, cj):
        # This function actually returns:
        # cj * (Jacobian matrix wrt dydt) + 
        # (Jacobian matrix wrt y), because that's how the solver
        # DASSL is written; cj is a numerical term determined
        # by DASSL.
        ode_jac = numpy.asarray([[-0.04, 1e4*y[2],1e4*y[1]],
        [0.04, -1e4*y[2] - 3e7*2*y[1], -1e4*y[1]],
        [0,    -3e7*2*y[1],                   0]])
        jac = -cj * numpy.eye(3) + ode_jac
        return jac

# <markdowncell>

# Next, we need to set up data structures to hold the solution to the DAE:

# <codecell>

t = [] # Time [s]

# <codecell>

y = [] # State variables; these are supposed to be concentrations

# <markdowncell>

# Define the end of the time interval over which we want a solution, and the maximum number of solver iterations:

# <codecell>

t_final = 1e6 #It's common to solve this problem for long times
maxiter = 5000

# <markdowncell>

# We need to set initial conditions for the problem.

# <codecell>

t_initial = 0
y_initial = numpy.array([1.0, 0, 0])
dydt_initial = numpy.array([-0.04, 0.04, 0])

# <codecell>

dassl = Problem()

# <codecell>

dassl.initialize(t_initial, y_initial, dydt_initial,
                 atol=1e-12, rtol=1e-12)

# <markdowncell>

# Then we can calculate a solution to the problem by using the `pydas.DASSL.step` method, which advances the time step automatically based on the solver (rather than manually selecting values of time at which to return data).

# <codecell>

iter = 0
while iter < maxiter and dassl.t < t_final:
    dassl.step(t_final)
    t.append(dassl.t)
    # Must make copy of dassl.y because DASSL overwrites it at
    # each call to pydas.DASSL.step
    y.append(dassl.y.copy())

# <markdowncell>

# Then we convert the solution arrays to NumPy arrays and plot the solution: 

# <codecell>

t = numpy.array(t)
y = numpy.array(y)

# <codecell>

t.shape

# <codecell>

y.shape

# <codecell>

fig1 = matplotlib.pyplot.figure()
ax = fig1.gca()
ax.plot(t,y[:,0],'r-')
ax.plot(t,y[:,1],'b-')
ax.plot(t,y[:,2],'k-')

# <codecell>

fig1.show()

# <codecell>


