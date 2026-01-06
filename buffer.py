import numpy as np

class Replay_Buffer:
    def __init__(self, obs_dim, act_dim, capacity=100000):
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity,), dtype=np.float32)
        self.dones    = np.zeros((capacity,), dtype=np.float32)

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
