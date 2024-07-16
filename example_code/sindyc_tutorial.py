
import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps
from pysindy.utils import lorenz_control

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12


# Control input
def u_fun(t):
    return np.column_stack([np.sin(2 * t), t**2])

# Generate measurement data
dt = 0.002
t_end_train = 10

t_train = np.arange(0, t_end_train, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [-8, 8, 27]
x_train = solve_ivp(
    lorenz_control,
    t_train_span,
    x0_train,
    t_eval=t_train,
    args=(u_fun,),
    **integrator_keywords,
).y.T
u_train = u_fun(t_train)
# Instantiate and fit the SINDYc model
model = ps.SINDy()
#default, polynomial, default optimiser is  STLSQ()
model.fit(x_train, u=u_train, t=dt)
model.print()
