import numpy as np
from numpy.linalg import solve
from CS_4link_dynamics import mass_matrix, coriolis_vector, gravity_vector
from CS_4link_consts import get_constants
from scipy.integrate import solve_ivp
# jax.config.update("jax_platform_name", "cpu")
def dynamics(states, inputs, const_vals):
    """
    states = (u, qd1, q1, qd2, q2, qdh, qh)
    inputs = (alpha, dalpha, ddalpha, ddphi)
    """
    args = (*states, *inputs, *const_vals)

    M = mass_matrix(*args)
    C = coriolis_vector(*args)
    G = gravity_vector(*args)
    return solve(M, -C - G).flatten()
def get_ordered_states(x):
    return np.array([x[3],x[4],x[0],x[5],x[1],x[6],x[2]])
def f(x, inp, const_vals):
    """
    x : state vector [q1, q2, qh, u, qd1,  qd2, qdh]
    u : input vector [alpha, dalpha, ddalpha, ddphi]
    """
    q_dot = np.array([x[4],x[5],x[6]])
    states = get_ordered_states(x)
    return np.concatenate([q_dot,
                     dynamics(states, inp, const_vals)])

def rk4_step(x, inp, const_vals, dt):
    k1 = f(x, inp, const_vals)
    k2 = f(x + 0.5 * dt * k1, inp, const_vals)
    k3 = f(x + 0.5 * dt * k2, inp, const_vals)
    k4 = f(x + dt * k3, inp, const_vals)

    return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
I = 90e-5
A = 4000
w = 2.0
def input_law(t):
    return np.array([0.0, 0.0, 0.0, -A])

def rhs(t, x, const_vals):
    inp = input_law(t)
    return f(x, inp, const_vals)
import time

# Constants
const_vals = get_constants()
x0 = np.zeros(7)
# Time span
t0, tf = 0.0, 40.0
t_eval = np.linspace(t0, tf, 1000)
ts = time.time()
sol = solve_ivp(
    rhs,
    (t0, tf),
    x0,
    method="RK45",
    t_eval=t_eval,
    args=(const_vals,)

)
te  = time.time()

print("time taken ",te-ts)
import matplotlib.pyplot as plt
t = sol.t              # shape (N,)
x = sol.y.T            # shape (N, 7)

q1  = x[:, 0]
q2  = x[:, 1]
qh  = x[:, 2]
u   = x[:, 3]
qd1 = x[:, 4]
qd2 = x[:, 5]
qdh = x[:, 6]

plt.plot(u,qdh)

# I = 90e-5
# A = 12.0/I
# w = 3.0
plt.show()
plt.plot(t,u)
plt.show()
plt.plot(t,qh)
plt.show()
