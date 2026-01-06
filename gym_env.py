import numpy as np
from numpy.linalg import solve
from CS_4link_dynamics import mass_matrix, coriolis_vector, gravity_vector
from CS_4link_consts import get_constants

import gymnasium as gym
from gymnasium import spaces


# jax.config.update("jax_platform_name", "cpu")
###########
# Dyanmics
###########
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

###########
# gym env
###########
I = 90e-5
A = 5.0/I
w = 3.0
class fish_env(gym.Env):

    def __init__(self,const_vals,dt = 0.01,max_steps = 1000):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.const_vals = const_vals

        self.state_dim = 7
        self.obs_dim = 2   # [u, qh]
        self.action_dim = 1  # only ddphi
        self.prev_action = 0.0

        # ---- Spaces ----
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([ 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.step_count = 0
        self.x = None

    # --------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.prev_action = 0.0
        self.x = np.zeros(7, dtype=np.float32)
        # self.x += 0.001 * np.random.randn(7).astype(np.float32)

        return self._get_obs(), {}

    # --------------------------------------------------
    def step(self, action):
        self.step_count += 1
        prev_action = self.prev_action
        # action = float(np.clip(action[0], -A, A))
        action  = action[0]*A
        self.prev_action = action
        # ---- build full input vector ----
        inp = np.array([0.0, 0.0, 0.0, action], dtype=np.float32)

        # ---- RK4 integration ----
        self.x = rk4_step(self.x, inp, self.const_vals, self.dt).astype(np.float32)


        obs = self._get_obs()
        reward = self._reward(self.x, action,prev_action)
        terminated = self._terminated(self.x)
        truncated = (self.step_count >= self.max_steps)

        return obs, reward, terminated, truncated, {}

    # --------------------------------------------------
    def _get_obs(self):
        # obs = [u, qh]
        return np.array([self.x[3], self.x[2]], dtype=np.float32)
    # --------------------------------------------------
    def _reward(self, x, action , prev_action):
        u = x[3]
        qh = x[2]
        qdh = x[6]
        xd = u*np.cos(qh)
        yd = u*np.sin(qh)
        if u < 0.0:
            r_u = 0.0
        elif u < 0.3:
            r_u =u #1 - abs(xd - 1.2)/2.4
        elif u <= 1.5:
            r_u = 1.0
        # else:
        #     r_u = 0.0  # or clamp, your choice

        reward = r_u  #- yd**2 - 0.1*((1/A)*(action - prev_action))**2

        reward*=1
        # example: stabilize upright & slow spin
        return reward

    # --------------------------------------------------
    def _terminated(self, x):
        if not np.isfinite(x).all():
            return True
        if abs(x[2]) > 1.0:
            return True
        return False


