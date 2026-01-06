import jax
import jax.numpy as jnp
from flax import nnx

class Actor(nnx.Module):
  def __init__(self,obs_dim,act_dim,*,rngs):
    self.fc1 = nnx.Linear(obs_dim,256,rngs=rngs)
    self.fc2 = nnx.Linear(256,256,rngs=rngs)
    self.mean = nnx.Linear(256,act_dim,rngs=rngs)
    self.log_std = nnx.Linear(256,act_dim,rngs=rngs)

  def __call__(self,obs):
    x = nnx.relu(self.fc1(obs))
    x = nnx.relu(self.fc2(x))
    mean = self.mean(x)
    log_std = self.log_std(x)
    log_std = jnp.clip(log_std,-25,2)
    return mean , log_std

class Critic(nnx.Module):
  def __init__(self,obs_dim,act_dim,*,rngs):
    self.fc1 = nnx.Linear(obs_dim+act_dim,256,rngs=rngs)
    self.fc2 = nnx.Linear(256,256,rngs=rngs)
    self.q = nnx.Linear(256,1,rngs=rngs)

  def __call__(self,obs,act):
    x = jnp.concatenate([obs,act],axis=-1)
    x = nnx.relu(self.fc1(x))
    x = nnx.relu(self.fc2(x))
    return self.q(x)

def sample_action(rng, mean, log_std, action_scale=1.0):
    std = jnp.exp(log_std)
    eps = jax.random.normal(rng, mean.shape)
    pre_tanh = mean + eps * std
    u = jnp.tanh(pre_tanh)

    log_prob = -0.5 * (jnp.square(eps) + 2 * log_std + jnp.log(2 * jnp.pi)).sum(axis=-1)

    # correction = 2.0 * (jnp.log(2.0) - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh))
    # log_prob -= jnp.sum(correction, axis=-1)

    # # scale correction
    # log_prob -= u.shape[-1] * jnp.log(action_scale)

    return u * action_scale, log_prob
