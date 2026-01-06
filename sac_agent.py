import jax
import jax.numpy as jnp
from flax import nnx
import optax
from networks import Actor, Critic, sample_action

class SAC(nnx.Module):
    def __init__(self, obs_dim, act_dim, rng):
        key_a, key_c1, key_c2 = jax.random.split(rng, 3)
        # Main networks
        self.actor = Actor(obs_dim, act_dim, rngs=nnx.Rngs(key_a))
        self.critic1 = Critic(obs_dim, act_dim, rngs=nnx.Rngs(key_c1))
        self.critic2 = Critic(obs_dim, act_dim, rngs=nnx.Rngs(key_c2))

        # Target networks - initialized with same weights
        self.target_critic1 = Critic(obs_dim, act_dim, rngs=nnx.Rngs(key_c1))
        self.target_critic2 = Critic(obs_dim, act_dim, rngs=nnx.Rngs(key_c2))

        # Copy weights from main to target
        nnx.update(self.target_critic1, nnx.state(self.critic1, nnx.Param))
        nnx.update(self.target_critic2, nnx.state(self.critic2, nnx.Param))

        # Learnable log(alpha) for automatic entropy tuning
        self.log_alpha = nnx.Param(jnp.zeros(()))

        # Target entropy: -dim(action_space) is a common heuristic
        self.target_entropy = -act_dim

        # Optimizers
        self.actor_opt = nnx.Optimizer(self.actor, optax.adam(3e-4), wrt=nnx.Param)
        self.critic1_opt = nnx.Optimizer(self.critic1, optax.adam(3e-4), wrt=nnx.Param)
        self.critic2_opt = nnx.Optimizer(self.critic2, optax.adam(3e-4), wrt=nnx.Param)
        self.alpha_opt = nnx.Optimizer(self, optax.adam(3e-4), wrt=lambda path, node: path == ('log_alpha',))

        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005

    @property
    def alpha(self):
        """Return current alpha value (exp of log_alpha)"""
        return jnp.exp(self.log_alpha.value)

@jax.jit
def update_actor(graphdef, state, batch, key):
    model = nnx.merge(graphdef, state)

    def actor_loss_fn(actor, batch, key_s):
        obs, act, rew, next_obs, done = batch
        mean, log_std = actor(obs)
        act_pi, logp = sample_action(key_s, mean, log_std)

        q1 = model.critic1(obs, act_pi).squeeze()
        q2 = model.critic2(obs, act_pi).squeeze()
        min_q = jnp.minimum(q1, q2)

        # SAC actor loss: maximize Q - alpha * entropy
        return (model.alpha * logp - min_q).mean()

    loss, grads = nnx.value_and_grad(actor_loss_fn)(model.actor, batch, key)
    model.actor_opt.update(model.actor, grads)

    graphdef, state = nnx.split(model)
    return state, loss

@jax.jit
def update_alpha(graphdef, state, batch, key):
    """Update temperature parameter alpha based on entropy"""
    model = nnx.merge(graphdef, state)

    def alpha_loss_fn(log_alpha_val, batch, key_s):
        obs, _, _, _, _ = batch
        mean, log_std = model.actor(obs)
        _, logp = sample_action(key_s, mean, log_std)

        # Alpha loss: we want entropy to match target_entropy
        # Loss = -log_alpha * (entropy + target_entropy)
        # Since entropy = -logp, we have:
        alpha_val = jnp.exp(log_alpha_val)
        return -(alpha_val * (logp + model.target_entropy)).mean()

    # Extract just the log_alpha parameter value
    log_alpha_val = model.log_alpha.value

    loss, grad = jax.value_and_grad(alpha_loss_fn)(log_alpha_val, batch, key)

    # Update using optimizer
    model.alpha_opt.update(model, {'log_alpha': nnx.Param(grad)})

    graphdef, state = nnx.split(model)
    return state, loss, model.alpha

@jax.jit
def update_critics(graphdef, state, batch, rng_key):
    """Update both critics simultaneously"""
    model = nnx.merge(graphdef, state)

    obs, act, rew, next_obs, done = batch
    rew = rew.flatten()
    done = done.flatten()

    # Compute target Q-value (shared for both critics)
    rng_next = jax.random.split(rng_key, 2)[1]
    next_mean, next_log_std = model.actor(next_obs)
    next_act, next_logp = sample_action(rng_next, next_mean, next_log_std)

    qt1 = model.target_critic1(next_obs, next_act).squeeze()
    qt2 = model.target_critic2(next_obs, next_act).squeeze()
    min_target_q = jnp.minimum(qt1, qt2)

    # Target with entropy bonus
    target = rew + model.gamma * (1.0 - done) * (min_target_q - model.alpha * next_logp)
    target = jax.lax.stop_gradient(target)

    # Update critic 1
    def critic1_loss_fn(critic1):
        q1 = critic1(obs, act).squeeze()
        return ((q1 - target) ** 2).mean()

    loss1, grads1 = nnx.value_and_grad(critic1_loss_fn)(model.critic1)
    model.critic1_opt.update(model.critic1, grads1)

    # Update critic 2
    def critic2_loss_fn(critic2):
        q2 = critic2(obs, act).squeeze()
        return ((q2 - target) ** 2).mean()

    loss2, grads2 = nnx.value_and_grad(critic2_loss_fn)(model.critic2)
    model.critic2_opt.update(model.critic2, grads2)

    # Soft update target networks
    def soft_update(target, online, tau):
        t_state = nnx.state(target, nnx.Param)
        o_state = nnx.state(online, nnx.Param)
        new_state = jax.tree.map(
            lambda t, o: (1 - tau) * t + tau * o,
            t_state, o_state
        )
        nnx.update(target, new_state)

    soft_update(model.target_critic1, model.critic1, model.tau)
    soft_update(model.target_critic2, model.critic2, model.tau)

    graphdef, state = nnx.split(model)
    return state, loss1 + loss2