import gymnasium as gym
import time
from networks import Actor, Critic, sample_action
import jax
from flax import nnx
import optax
from buffer import Replay_Buffer
import numpy as np
from sac_agent import SAC, update_actor, update_critics ,update_alpha
from tensorboardX import SummaryWriter
import orbax.checkpoint as ocp
from pathlib import Path
from gym_env import fish_env
from CS_4link_consts import get_constants
# env = gym.make("Pendulum-v1", render_mode="human")
const_vals = get_constants()
env = fish_env(const_vals=const_vals, dt=0.01, max_steps=200)


obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
buffer = Replay_Buffer(obs_dim, act_dim, 1000000)

key = jax.random.PRNGKey(0)
sac = SAC(obs_dim, act_dim, key)

# Split State for JIT
graphdef, state = nnx.split(sac)
obs, _ = env.reset()

# Warmup buffer
print("Warming up replay buffer...")
for step in range(1000):
    action = env.action_space.sample()
    # action = np.array([np.cos(3*step*0.01)])
    # print(action)

    next_obs, reward, terminated, truncated, _ = env.step(action)
    # print(next_obs)
    done = terminated or truncated
    buffer.add(obs, next_obs, action, reward, done)
    if done:
        obs, _ = env.reset()
    else:
        obs = next_obs



print(buffer.ptr)
N_STEPS = 100000
print(f"Starting training for {N_STEPS} steps...")
print("-" * 40)

# JIT warmup
batch = buffer.sample(10)
key, c_key, a_key = jax.random.split(key, 3)

t0 = time.time()
state, c_loss = update_critics(graphdef, state, batch, c_key)
state, a_loss = update_actor(graphdef, state, batch, a_key)
t1 = time.time()

print(f"JIT compilation time: {t1 - t0:.4f} s")

episode_return = 0.0
episode_length = 0
returns_log = []
lengths_log = []
best_return = -np.inf

writer = SummaryWriter(log_dir="runs/fish")

# Execution Loop
start_time = time.time()
obs, _ = env.reset()

for step in range(N_STEPS):
    key, act_key, c_key, a_key ,alpha_key= jax.random.split(key, 5)

    # Sample action from policy
    mean, log_std = sac.actor(obs)
    action, logp = sample_action(act_key, mean, log_std)

    # Execute action in environment
    next_obs, reward, terminated, truncated, _ = env.step(np.array(action))
    done = terminated or truncated

    buffer.add(obs, next_obs, action, reward, done)
    episode_return += reward
    episode_length += 1
    for i in range(2):
    # Update networks
        batch = buffer.sample(64)
        state, c_loss = update_critics(graphdef, state, batch, c_key)

    # Update actor every step (you had step%1==0 which is always true)
    if step%2 == 0:
        state, a_loss = update_actor(graphdef, state, batch, a_key)
        state, alpha_loss, current_alpha = update_alpha(graphdef, state, batch, alpha_key)

    # Merge state back to get updated SAC model
    sac = nnx.merge(graphdef, state)

    # Logging
    if step % 1 == 0:
        writer.add_scalar("loss/critic", c_loss, step)
        writer.add_scalar("loss/actor", a_loss, step)
        writer.add_scalar("velocity", obs[0], step)
        writer.add_scalar("head_angle", obs[1], step)
        writer.add_scalar("input",action[0], step)
        writer.add_scalar("alpha",current_alpha, step)

    if done:
        returns_log.append(episode_return)
        lengths_log.append(episode_length)
        writer.add_scalar("Reward/Episode", episode_return, step)

        # Save best policy (convert to absolute path)
        if episode_return > best_return:
            best_return = episode_return
            checkpointer = ocp.PyTreeCheckpointer()
            actor_state = nnx.state(sac.actor, nnx.Param)
            save_path = Path("saved_models/best_policy").resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            checkpointer.save(str(save_path), actor_state, force=True)
            print(f"Step {step} | New best! Return={episode_return:.2f} | Saved to {save_path}")
        else:
            print(f"Step {step} | Episode done | return={episode_return:.2f} | len={episode_length}")

        episode_return = 0.0
        episode_length = 0
        obs, _ = env.reset()
    else:
        obs = next_obs

end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / N_STEPS

print(f"\nTraining completed!")
print(f"Total time for {N_STEPS} steps: {total_time:.4f} s")
print(f"Average time per step: {avg_time*1000:.4f} ms")
print(f"Best return achieved: {best_return:.2f}")
print("-" * 40)

# ============================================================================
# SAVE FINAL MODEL AND TRAINING STATS
# ============================================================================
print("\nSaving final model and training statistics...")

checkpointer = ocp.PyTreeCheckpointer()

# Save final policy
actor_state = nnx.state(sac.actor, nnx.Param)
final_path = Path("saved_models/final_policy").resolve()
final_path.parent.mkdir(parents=True, exist_ok=True)
checkpointer.save(str(final_path), actor_state, force=True)
print(f"✓ Final policy saved to: {final_path}")

# Save full model for resuming training
full_state = nnx.state(sac)
full_path = Path("saved_models/full_checkpoint").resolve()
checkpointer.save(str(full_path), full_state, force=True)
print(f"✓ Full checkpoint saved to: {full_path}")

# Save with metadata
metadata = {
    'returns': returns_log,
    'lengths': lengths_log,
    'total_steps': N_STEPS,
    'best_return': float(best_return),
    'final_return': float(returns_log[-1]) if returns_log else 0,
    'avg_return_last_10': float(np.mean(returns_log[-10:])) if len(returns_log) >= 10 else 0,
    'hyperparameters': {
        'alpha': float(sac.alpha),
        'gamma': float(sac.gamma),
        'tau': float(sac.tau)
    }
}

checkpoint_with_meta = {
    'actor_state': actor_state,
    'metadata': metadata
}
meta_path = Path("saved_models/policy_with_metadata").resolve()
checkpointer.save(str(meta_path), checkpoint_with_meta, force=True)
print(f"✓ Policy with metadata saved to: {meta_path}")
print(f"\nTraining Summary:")
print(f"  Total episodes: {len(returns_log)}")
print(f"  Best return: {best_return:.2f}")
print(f"  Final return: {returns_log[-1]:.2f}" if returns_log else "  No episodes completed")
print(f"  Avg last 10 episodes: {np.mean(returns_log[-10:]):.2f}" if len(returns_log) >= 10 else "")

writer.close()
env.close()

print("\n" + "="*50)
print("To evaluate the trained policy, run:")
print("  python -c 'from jax_save_load import run_saved_policy; run_saved_policy(\"saved_models/best_policy\")'")
print("="*50)