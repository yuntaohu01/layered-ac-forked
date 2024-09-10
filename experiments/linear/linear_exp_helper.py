import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import cvxpy as cp

import sys
sys.path.append('../..')

# Linear env
from env.dynamics import linear
from env.costs import quadratic
from env.gymenv import GymEnv

from agent.layeredargs import LayeredArgs
from agent.layered import LayeredAgent
from agent.policy import LinearActor
from agent.qnetwork import QuadraticQNetwork


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
    """
    Solve the discrete-time algebraic Riccati equation backward in time.

    Parameters:
        A (ndarray): State transition matrix.
        B (ndarray): Control matrix.
        Q (ndarray): State cost matrix.
        R (ndarray): Control cost matrix.
        T (int): Time horizon.

    Returns:
        P (list of ndarrays): List of solution matrices P_t for t = T, ..., 0.
    """
    P = [None] * (T + 1)
    K = [None] * (T)
    P[T] = Q  # Final cost

    for t in range(T - 1, -1, -1):
        # Compute the optimal feedback gain
        K[t] = -np.linalg.inv(R + B.T @ P[t + 1] @ B) @ B.T @ P[t + 1] @ A

        # Update the value matrix P
        P[t] = Q + A.T @ P[t + 1] @ A + A.T @ P[t + 1] @ B @ K[t]

    return P, K

def optimal_dual_map(A, B, Q, R, T, rho):
    # Compute Theta^* of the system (optimal dual map)
    Nx, Nu = B.shape
    E = np.zeros(((T + 1) * Nx, T * Nu))
    for i in range(T+1):
        for j in range(0, i):
            E[i*Nx:(i+1)*Nx, j*Nu:(j+1)*Nu] = np.linalg.matrix_power(A, i-j-1) @ B
    F = np.vstack([np.linalg.matrix_power(A, i) for i in range(T+1)])
    QQ = np.kron(np.eye(T+1), Q)
    RR = np.kron(np.eye(T), R)
    return torch.tensor(
        -2 / rho * (-QQ @ E @ np.linalg.pinv(E.T @ QQ @ E + RR) @ E.T @ QQ @ F + F)
    )

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
    prob = cp.Problem(cp.Minimize(objective), [])
    try:
        prob.solve()
    except Exception as e:
        print(f"Error occured when solving the QP: {e}")
        print(np.linalg.svd(P))
    return torch.tensor(rtilde.value).float().unsqueeze(0)

actor_creation_fn = lambda obs_dim, act_dim: LinearActor(obs_dim, act_dim)
qnetwork_creation_fn = lambda obs_dim, act_dim: QuadraticQNetwork(obs_dim, act_dim)

def dual_network_creation_fn(nominal_obs_dim, obs_dim):
    dual_network = nn.Linear(
        nominal_obs_dim, obs_dim - nominal_obs_dim,
        bias=False
    )
    dual_network.weight.data.zero_()
    return dual_network
