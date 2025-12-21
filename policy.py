import numpy as np
from gym_env import fish_env
from CS_4link_consts import get_constants

import jax

class Replay_Buffer:
    def __init__(self, obs_dim, act_dim, capacity):
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.ptr = 0
        self.size = 0
    def add(self, obs,next_obs, action, reward, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )
























# buffer = Replay_Buffer(2,1,100000)

# const_vals = get_constants()
# env = fish_env(const_vals=const_vals, dt=0.01, max_steps=1000)

# obs, _ = env.reset()

# for step in range(1000):

#     action = env.action_space.sample()  # or random during warmup

#     next_obs, reward, terminated, truncated, _ = env.step(action)

#     done = terminated or truncated

#     buffer.add(obs, next_obs, action, reward, done)

#     obs = next_obs

#     if done:
#         obs, _ = env.reset()

# import numpy as np
# import jax
# import jax.numpy as jnp
# import time


# obs, acts, rews, next_obs, dones = buffer.sample(256)
# t0 = time.time()

# batch_gpu = {
#     "obs": jax.device_put(obs),
#     "acts": jax.device_put(acts),
#     "rews": jax.device_put(rews),
#     "next_obs": jax.device_put(next_obs),
#     "dones": jax.device_put(dones),
# }