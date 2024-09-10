import torch
import torch.nn as nn
import numpy as np
import random
import cvxpy as cp

import sys
sys.path.append('../..')

from env.dynamics import linear
from env.costs import quadratic
from env.gymenv import GymEnv


def generate_env(Nx, Nu, T, u_min, u_max, A_norm, RQratio, seed):
    # Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dynamics = linear.sample_linear_dynamics(Nx, Nu, A_norm=A_norm)
    cost = quadratic.QuadraticCost(
        torch.eye(dynamics.Nx),
        RQratio * torch.eye(dynamics.Nu)
    )
    reset_fn = lambda: torch.randn(dynamics.Nx)

    return GymEnv(dynamics, cost, reset_fn, T, ulim=[u_min, u_max])

def riccati(A, B, Q, R, T):
    P = [None] * (T + 1)
    K = [None] * (T)
    P[T] = Q  # Final cost
    for t in range(T - 1, -1, -1):
        K[t] = -np.linalg.inv(R + B.T @ P[t + 1] @ B) @ B.T @ P[t + 1] @ A
        P[t] = Q + A.T @ P[t + 1] @ A + A.T @ P[t + 1] @ B @ K[t]
    return P, K

def dual_network_creation_fn(nominal_obs_dim, obs_dim):
    dual_network = nn.Sequential(
        nn.Linear(nominal_obs_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, obs_dim - nominal_obs_dim),
    )
    dual_network[-1].weight.data.mul_(0.1)
    dual_network[-1].bias.data.mul_(0.01)
    return dual_network

##########################################################################
# The next two functions assume that the constraint is x[2:] >= -0.05
##########################################################################

def ref_traj_gen_fn(x0s, T, agent, global_step):
    b, Nx = x0s.shape
    assert b == 1, "Can only handle one initial condition at a time"
    x0 = x0s[0]
    Q = agent.nominal_env.cost_fn.Q
    K = agent.actor.gain.weight.data.cpu().numpy()
    P = agent.qf1.get_P().cpu().numpy()
    P = (P + P.T) / 2

    r = cp.Variable(Nx * (T + 1))
    v = agent.dual_network(torch.tensor(x0, device=agent.device)).detach().cpu().numpy()
    if agent.args.no_dual:
        v = np.zeros_like(v)
    rtilde = r + v
    xr = cp.hstack([x0, rtilde])
    u = K @ xr
    xru = cp.hstack([xr, u])
    objective = cp.quad_form(r, np.kron(np.eye(T+1), Q.numpy()))
    objective += cp.quad_form(xru, cp.psd_wrap(P))
    prob = cp.Problem(
        cp.Minimize(objective), [r[2:] >= -0.05]
    )
    try:
        prob.solve()
    except Exception as e:
        print(f"Error occured when solving the QP: {e}")
        print(np.linalg.svd(P))
    return torch.tensor(rtilde.value).float().unsqueeze(0)


def constr_violation(agent, actual_traj):
    return (-0.05 - actual_traj[:, 2:]).clip(0, torch.inf).mean().item()
