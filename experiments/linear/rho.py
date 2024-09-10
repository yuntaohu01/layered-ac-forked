import torch

import sys
sys.path.append('../..')

from agent.layeredargs import LayeredArgs
from agent.layered import LayeredAgent
from agent.policy import LinearActor
from agent.qnetwork import QuadraticQNetwork

from linear_exp_helper import *


######################## Parameters #################################
T = 20
""" Length of the episode """
A_norm = 1.
""" Spectral radius of the linear system """
RQratio = 1e-2
""" Ratio of control penalty over state cost """
u_min, u_max = -10, 10
""" Actuation bound """
u_range = u_max - u_min

args = LayeredArgs(
    seed=1,
    torch_deterministic=True,
    total_timesteps=100_000, #orignally 70_000
    finetune_steps=10_000, #originally 5_000
    learning_starts=2_000, # have one batch before 
    buffer_size=10_000,
    gamma=1., # No discount here
    policy_noise= 0.01 / u_range,
    noise_clip= 0.02 / u_range,
    exploration_noise=0.012, #originally 0.
    dual_optimizer='SGD',
    learning_rate=1e-3, #originally 3e-3
    dual_learning_rate=1e-2, #originally 1e-1
    dual_batch_size=5,
    cuda=True,
    rho=2,
    tau=0.005,
)
####################################################################

rhos = [5e-1, 1, 2, 4, 8]
Nx, Nu = 4, 2
num_reps = 15

for rho in rhos:
    for seed in range(num_reps):
        nominal_env = generate_env(Nx, Nu, T, u_min, u_max, A_norm, RQratio, seed)
        A = nominal_env.dynamics.A.numpy()
        B = nominal_env.dynamics.B.numpy()
        Q = nominal_env.cost_fn.Q.numpy()
        R = nominal_env.cost_fn.R.numpy()
        nominal_P, nominal_K = riccati(A, B, Q, R, T)
        opt_cost_fn = lambda x0: x0 @ (nominal_P[0] @ x0)

        args.exp_name = f"rho_{rho}_{seed}"
        args.rho = rho
        args.exploration_noise = Nx * 0.003
        agent = LayeredAgent(
            nominal_env,
            actor_creation_fn,
            qnetwork_creation_fn,
            dual_network_creation_fn,
            ref_traj_gen_fn,
            opt_cost_fn,
            [],
            args
        )
        agent.learn()
