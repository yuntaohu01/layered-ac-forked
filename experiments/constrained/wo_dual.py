import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp

import sys
sys.path.append('../..')

from agent.layeredargs import LayeredArgs
from agent.layered import LayeredAgent
from agent.policy import LinearActor
from agent.qnetwork import QuadraticQNetwork

from constrained_exp_helper import *


######################## Parameters #################################
T = 20
""" Length of the episode """
A_norm = .995
""" Spectral radius of the linear system """
RQratio = 1e-2
""" Ratio of control penalty over state cost """
u_min, u_max = -10, 10
""" Actuation bound """
u_range = u_max - u_min

args = LayeredArgs(
    seed=1,
    torch_deterministic=True,
    exp_name=f"actor_critic",
    total_timesteps=150_000,
    finetune_steps=250_000,
    learning_starts=2_000, # have one batch before 
    buffer_size=10_000,
    gamma=1., # No discount here
    policy_noise=0.01 / u_range,
    noise_clip=0.02 / u_range,
    exploration_noise=0.01 / u_range,
    learning_rate=3e-3,
    dual_optimizer='Adam',
    dual_learning_rate=3e-4,
    dual_batch_size=40,
    cuda=True,
    rho=2,
    tau=0.005,
    no_dual=True
)

actor_creation_fn = lambda obs_dim, act_dim: LinearActor(obs_dim, act_dim)
qnetwork_creation_fn = lambda obs_dim, act_dim: QuadraticQNetwork(obs_dim, act_dim)

####################################################################
Nx = 2
num_reps = 10
for seed in range(1, num_reps):
    Nu = Nx
    nominal_env = generate_env(Nx, Nu, T, u_min, u_max, A_norm, RQratio, seed)
    A = nominal_env.dynamics.A.numpy()
    B = nominal_env.dynamics.B.numpy()
    Q = nominal_env.cost_fn.Q.numpy()
    R = nominal_env.cost_fn.R.numpy()

    def opt_cost_fn(x0):
        x_opt = cp.Variable((T+1, B.shape[0]))
        u_opt = cp.Variable((T, B.shape[1]))
        objective = 0
        constr = [x_opt[0] == x0]
        for t in range(T):
            objective += cp.quad_form(x_opt[t], Q) + cp.quad_form(u_opt[t], R)
            constr += [
                x_opt[t+1] == A @ x_opt[t] + B @ u_opt[t]
            ]
        constr += [x_opt[1:] >= -0.05]
        prob = cp.Problem(cp.Minimize(objective), constr)
        prob.solve()
        return objective.value

    args.exp_name = f"wo_dual_{seed}"
    agent = LayeredAgent(
        nominal_env,
        actor_creation_fn,
        qnetwork_creation_fn,
        dual_network_creation_fn,
        ref_traj_gen_fn,
        opt_cost_fn,
        [('val_constr_vio', constr_violation)],
        args
    )
    agent.learn()

    # c_costs, l_costs = [], []
    # for _ in range(30):
    #     # Sample x0 and reference trajectory
    #     x0ref, _ = td3agent.envs.reset()
    #     diff = x0ref[0, :Nx] - x0ref[0, Nx:Nx*2]
    #     dx0ref = np.concatenate([np.zeros_like(diff), x0ref[0]])

    #     # Find the optimal trajectory
    #     x_opt = cp.Variable((T+1, B.shape[0]))
    #     u_opt = cp.Variable((T, B.shape[1]))
    #     objective = 0
    #     constr = [x_opt[0] == x0ref[0, :Nx]]
    #     for t in range(T):
    #         objective += cp.quad_form(x_opt[t], Q) + cp.quad_form(u_opt[t], R)
    #         constr += [
    #             x_opt[t+1] == A @ x_opt[t] + B @ u_opt[t]
    #         ]
    #     constr += [x_opt[1:] >= -0.05]
    #     prob = cp.Problem(cp.Minimize(objective), constr)
    #     prob.solve()

    #     x_l = [x0ref]  # learned
    #     u_l = []
    #     tracking_cost = 0
    #     for t in range(T):
    #         u_l.append(td3agent.actor(torch.tensor(x_l[-1])).detach().numpy())
    #         x_l_, r, term, _, info = td3agent.envs.step(u_l[-1])
    #         tracking_cost += r
    #         x_l.append(x_l_ if not term else info['final_observation'][0])
    #     
    #     c_costs.append(objective.value)
    #     l_costs.append(-info['final_info'][0]['nominal_return'])
    # print(np.sum(l_costs) / np.sum(c_costs))
