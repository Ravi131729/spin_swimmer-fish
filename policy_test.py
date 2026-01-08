from pathlib import Path
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp
from networks import Actor,sample_action
from gym_env import fish_env
from CS_4link_consts import get_constants
from tensorboardX import SummaryWriter

# import matplotlib.pyplot as plt
# env = gym.make("Pendulum-v1", render_mode="human")
import jax
const_vals = get_constants()
env = fish_env(const_vals=const_vals, dt=0.01, max_steps=200)


obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
key = jax.random.PRNGKey(10)

actor = Actor(obs_dim=obs_dim,act_dim=act_dim,rngs=nnx.Rngs(key))
checkpointer = ocp.StandardCheckpointer()

abstract_state = nnx.state(actor, nnx.Param)
final_path = Path("saved_models/final_policy").resolve()
actor_state = checkpointer.restore(
    str(final_path),
    abstract_state  # Pass the abstract state structure
)
nnx.update(actor, actor_state)
nsteps = 1000
obs, _ = env.reset()
writer = SummaryWriter(log_dir="final_runs/fish")
u = [obs[0]]
t = []
x0 = 0.0
y0 = 0.0
dt = 0.01
xt = [x0]
yt = [y0]
inp=[]
for step in range(nsteps):
    key, act_key= jax.random.split(key, 2)

    # Sample action from policy
    mean, log_std = actor(obs)
    action, logp = sample_action(act_key, mean, log_std)
    # print(mean)
    # action = np.tanh(mean)
    # print(action)
    next_obs, reward, terminated, truncated, _ = env.step(np.array(action))


    xd = obs[0]*np.cos(obs[1])
    yd = obs[0]*np.sin(obs[1])
    x0 = x0 + xd*0.01
    y0 = y0 + yd*0.01
    xt.append(x0)
    yt.append(y0)
    inp.append(action)
    t.append(step*dt)
    writer.add_scalar("velocity", obs[0], step)
    writer.add_scalar("action", action, step)
    writer.add_scalar("head", obs[1], step)
    writer.add_scalar("xd", obs[0]*np.cos(obs[1]), step)
    writer.add_scalar("yd", obs[0]*np.sin(obs[1]), step)
    writer.add_scalar("trajectoryy", y0, step)
    writer.add_scalar("trajectoryx", x0, step)
    obs = next_obs
writer.close()
env.close()
# plt.plot(yt,xt)
# plt.show
np.savez("trajectory.npz", xt=xt, yt=yt , inp=inp , t = t)