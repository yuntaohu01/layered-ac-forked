import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from typing import Callable, Optional
from .gymenv import GymEnv


class TrajConditionedEnv(gym.Env):
    """ Tracking version of a nominal system. Augments the state
    to be the nominal state concatenated with the reference trajectory.
    """

    def __init__(self,
                 nominal_env: GymEnv,
                 tracking_cost: Callable[
                     [torch.Tensor, torch.Tensor, torch.Tensor],
                     torch.Tensor
                 ],
                 ref_traj_len: int,
                 ref_traj_gen_fn: Callable[[torch.Tensor, int], torch.Tensor],
                 partial_tracking_mask: Optional[torch.Tensor]=None
    ) -> None:
        """
        :Parameter:
            - nominal_env:      a nominal environment that specifies dynamics
                                and state initialization
            - tracking_cost:    a function that takes in batches of
                                (x_prev, x, u) and returns batch tracking costs
            - ref_traj_len:     length of the reference trajectory
            - ref_traj_gen_fn:  how to generate reference trajectory based on
                                initial state
            - tracking_mask:    a boolean tensor with the same shape as the nominal
                                state that specifies which states need to be tracked
        """
        self.nominal_env = nominal_env
        self.nominal_dynamics = nominal_env.dynamics
        self.nominal_Nx = nominal_env.dynamics.Nx
        self.nominal_reset_fn = nominal_env.reset_fn
        if partial_tracking_mask is not None:
            self.tracking_mask = partial_tracking_mask.numpy()
        else:
            self.tracking_mask = np.ones(self.nominal_Nx, dtype=np.bool_)
        self.tracked_Nx = self.tracking_mask.sum()
        self.augmented_Nx = self.nominal_Nx + self.tracked_Nx * (ref_traj_len + 1)
        self.T = nominal_env.T
        self.tracking_cost = tracking_cost
        self.ref_traj_len = ref_traj_len
        self.ref_traj_gen_fn = ref_traj_gen_fn

        # Augment the state space
        self.observation_space = spaces.Box(
            low=np.concatenate([
                nominal_env.observation_space.low,
                np.ones(self.tracked_Nx * (ref_traj_len + 1)) * -np.inf
            ]),
            high=np.concatenate([
                nominal_env.observation_space.high,
                np.ones(self.tracked_Nx * (ref_traj_len + 1)) * np.inf
            ]),
            shape=(self.augmented_Nx,),
            dtype=nominal_env.observation_space.dtype
        )
        self.action_space = nominal_env.action_space

        # We keep x to be a batched torch.Tensor
        self.x = None
        self.t = None
        self.ref = None
        self.xtraj = None

    def reset(self, options=None, seed=None):
        """ Resets the state back to initial state """
        # Set manual seed
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)

        x0, _ = self.nominal_env.reset(seed)
        nominal_x = torch.tensor(x0).unsqueeze(0)
        ref_traj = self.ref_traj_gen_fn(nominal_x.numpy(), self.ref_traj_len).cpu()
        self.x = torch.hstack([nominal_x, ref_traj])
        self.t = 0
        self.ref = ref_traj
        self.xtraj = [self.x[:, :self.nominal_Nx]]
        self.utraj = []
        self.nominal_return = 0
        return self.x[0].numpy(), {}

    def step(self, u: np.ndarray):
        """ Step the state forward """
        u_tensor = torch.tensor(u).unsqueeze(0).float()
        with torch.no_grad():
            nex_, nominal_reward, _, _, _ = self.nominal_env.step(u)
            nominal_x_ = torch.tensor(nex_).unsqueeze(0)
            ref_traj = self.x[:, self.nominal_Nx:]
            ref_traj_ = torch.hstack(
                [ref_traj[:, self.tracked_Nx:], ref_traj[:, -self.tracked_Nx:]]
            )
            x_ = torch.hstack([nominal_x_, ref_traj_])
            cost = self.tracking_cost.stage_cost(self.x, x_, u_tensor, self.t+1==self.T)
            reward = -cost
        self.x = x_
        self.xtraj.append(self.x[:, :self.nominal_Nx])
        self.utraj.append(u)
        self.nominal_return += nominal_reward
        self.t += 1
        if self.t < self.T:
            return self.x[0].numpy(), reward.item(), False, False, {}
        else:
            final_info = {
                "reference": self.ref,
                "actual": torch.tensor(np.hstack(self.xtraj)),
                "nominal_return": self.nominal_return,
                "control": np.vstack(self.utraj)
            }
            return self.x[0].numpy(), reward.item(), True, False, final_info
