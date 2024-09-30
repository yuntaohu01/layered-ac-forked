# Adapted from vwxyzjn/cleanrl
import sys
import os
import random
import time
from dataclasses import dataclass
from typing import Callable
from copy import deepcopy

import cvxpy as cp
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
from env.trajcondenv import TrajConditionedEnv
from env.costs.quadratictracking import QuadraticTrackingCost
from .policy import SimpleActor, LinearActor, TransformerUnicycleActor
from .qnetwork import SimpleQNetwork, QuadraticQNetwork, UnicycleQNetwork
from .layeredargs import LayeredArgs


class LayeredAgent:
    def __init__(self,
                 nominal_env,
                 actor_creation_fn,
                 qnetwork_creation_fn,
                 dual_network_creation_fn,
                 ref_traj_gen_fn,
                 opt_cost_fn,
                 extra_validation_fns,
                 args: LayeredArgs,
                 tracking_mask=None,
                 warmstart_steps=0):
        """
        extra_validation_fns:   a list of tuples (name, function to evaluate)
        """
        # Store the args
        self.args = args
        self.nominal_env = nominal_env
        self.seed(self.args.seed, self.args.torch_deterministic)
        self.opt_cost_fn = opt_cost_fn
        self.extra_validation_fns = extra_validation_fns
        self.warmstart_steps = warmstart_steps

        # Create a tracking env
        if tracking_mask is None:
            tracking_Q = torch.eye(nominal_env.dynamics.Nx) * self.args.rho / 2
        else:
            tracking_Q = torch.eye(tracking_mask.sum()) * self.args.rho / 2
        tracking_cost = QuadraticTrackingCost(
            tracking_Q, nominal_env.cost_fn.R, tracking_mask=tracking_mask
        )
        tracking_env_creation_fn = lambda: TrajConditionedEnv(
                nominal_env, tracking_cost, nominal_env.T, None, tracking_mask
        )
        self.envs = self.construct_vec_env(
            tracking_env_creation_fn, self.args.seed
        )
        self.validation_env = TrajConditionedEnv(
            deepcopy(nominal_env), tracking_cost, nominal_env.T, None, tracking_mask
        )
        observation_dim = np.array(self.envs.observation_space.shape).prod()
        action_dim = np.array(self.envs.action_space.shape).prod()
        nominal_obs_dim = np.array(nominal_env.observation_space.shape).prod()
        self.tracking_mask = torch.tensor(self.validation_env.tracking_mask)

        # Initialize the actor
        self.actor = actor_creation_fn(observation_dim, action_dim)
        self.target_actor = actor_creation_fn(observation_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=args.learning_rate
        )

        # Initialize the critics
        self.qf1 = qnetwork_creation_fn(observation_dim, action_dim)
        self.qf2 = qnetwork_creation_fn(observation_dim, action_dim)
        self.qf1_target = qnetwork_creation_fn(observation_dim, action_dim)
        self.qf2_target = qnetwork_creation_fn(observation_dim, action_dim)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=args.learning_rate
        )

        # Initialize the dual network
        self.dual_network = dual_network_creation_fn(
            nominal_obs_dim, observation_dim
        )
        if self.args.dual_optimizer == 'Adam':
            self.dual_network_optimizer = optim.Adam(
                list(self.dual_network.parameters()), lr=args.dual_learning_rate
            )
        else:
            self.dual_network_optimizer = optim.SGD(
                self.dual_network.parameters(), lr=args.dual_learning_rate
            )

        # Setup the reference traj generation algorithm
        self.ref_traj_gen_fn = ref_traj_gen_fn
        self.reset_reference_generation(0)

        # Move all models to the specified device
        self.device = (
            "cuda" if torch.cuda.is_available()
            and args.cuda else "cpu"
        )
        self.to(self.device)

        # Set up logger
        run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" \
                for key, value in vars(args).items()])),
        )

        # Existing initialization code
        self.checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize replay buffer to be None
        self.rb = None

    def reset_reference_generation(self, global_step):
        rtgf = lambda x0s, T: self.ref_traj_gen_fn(x0s, T, self, global_step)
        for ev in self.envs.envs:
            ev.env.ref_traj_gen_fn = rtgf
        self.validation_env.ref_traj_gen_fn = rtgf

    def to(self,device):
        self.device = device
        self.actor.to(device)
        self.qf1.to(device)
        self.qf2.to(device)
        self.target_actor.to(device)
        self.qf1_target.to(device)
        self.qf2_target.to(device)
        self.dual_network.to(device)

    def construct_vec_env(self, env_creation_fn, seed):
        def make_env(env_fn, seed):
            env = gym.wrappers.RecordEpisodeStatistics(env_fn())
            env.action_space.seed(seed)
            return env
        envs = gym.vector.SyncVectorEnv([
            lambda: make_env(env_creation_fn, seed)
        ])
        assert isinstance(envs.single_action_space, gym.spaces.Box), \
            "only continuous action space is supported"
        envs.single_observation_space.dtype = np.float32
        return envs

    def seed(self, seed, torch_deterministic):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

    def create_replay_buffer(self, envs, buffer_size):
        return ReplayBuffer(
            buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )

    def validate_current_policy(self, num_x0s=50):
        nominal_rets, opt_rets, rx_diffs = [], [], []
        if self.extra_validation_fns is not None:
            extra_vals = [[]] * len(self.extra_validation_fns)
        for i in range(num_x0s):
            obs, _ = self.validation_env.reset()
            for t in range(self.nominal_env.T):
                actions = self.actor(torch.Tensor(obs).unsqueeze(0).to(self.device))
                next_obs, _, term, _, info = \
                    self.validation_env.step(actions[0].cpu().detach().numpy())
                obs = next_obs
            
            nominal_Nx = np.prod(self.nominal_env.observation_space.shape)
            nominal_x0 = info["actual"][0, :nominal_Nx].numpy()
            ref_traj = info["reference"].to(self.device)
            actual_traj = info["actual"].to(self.device)
            # Mask out states in actual_traj that are not in the tracking mask
            actual_traj = actual_traj.reshape(
                1, -1, nominal_Nx
            )[:, :, self.tracking_mask].flatten(1)
            with torch.no_grad():
                v = self.dual_network(actual_traj[:, :nominal_Nx]).detach()
                if self.args.no_dual:
                    v.zero_()
                rx_diffs.append((ref_traj - v - actual_traj).abs().mean().item())
                nominal_rets.append(info['nominal_return'])
                if self.opt_cost_fn is not None:
                    opt_rets.append(self.opt_cost_fn(nominal_x0))

            for (val_list, (_, val_fn)) in zip(extra_vals, self.extra_validation_fns):
                val_list.append(val_fn(self, actual_traj))

        extra_vals = [np.median(l) for l in extra_vals]
        rel_cost = (
            -np.sum(nominal_rets) / np.sum(opt_rets)
            if self.opt_cost_fn is not None
            else -np.mean(nominal_rets)
        )
        return np.mean(rx_diffs), rel_cost, extra_vals

    def _save_checkpoint(self, global_step):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'qf1_state_dict': self.qf1.state_dict(),
            'qf1_target_state_dict': self.qf1_target.state_dict(),
            'qf2_state_dict': self.qf2.state_dict(),
            'qf2_target_state_dict': self.qf2_target.state_dict(),
            'dual_network_state_dict': self.dual_network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'dual_network_optimizer_state_dict': self.dual_network_optimizer.state_dict(),
            'replay_buffer': self.rb,  # Ensure your replay buffer is serializable
            'global_step': global_step,
            # Add any other necessary state information
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint_step_{global_step}.pth')
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at step {global_step} to {filename}")
    
    def _load_checkpoint(self, filepath):
        """
        Loads the agent's state from a checkpoint file.
        
        Args:
            filepath (str): Path to the checkpoint file.
        
        Returns:
            int: The global step from which to resume training.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.qf1.load_state_dict(checkpoint['qf1_state_dict'])
        self.qf1_target.load_state_dict(checkpoint['qf1_target_state_dict'])
        self.qf2.load_state_dict(checkpoint['qf2_state_dict'])
        self.qf2_target.load_state_dict(checkpoint['qf2_target_state_dict'])
        self.dual_network.load_state_dict(checkpoint['dual_network_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.dual_network_optimizer.load_state_dict(checkpoint['dual_network_optimizer_state_dict'])
        
        self.rb = checkpoint['replay_buffer']
        global_step = checkpoint['global_step']

        print(f"[INFO] Checkpoint loaded from '{filepath}' at step {global_step}")
        return global_step

    def learn(self,
              overwrite_rb=True,
              seed=None,
              load_checkpoint_path=None):

        self.seed(self.args.seed, self.args.torch_deterministic)

        # Optionally load from a checkpoint
        if load_checkpoint_path is not None:
            global_step = self._load_checkpoint(load_checkpoint_path)
        else:
            global_step = self.args.learning_starts  # Start from initial step

        # replay buffer
        if overwrite_rb:
            self.rb = self.create_replay_buffer(self.envs, self.args.buffer_size)
        else:
            assert self.rb is not None, \
                    "Need to first initialize replay buffer"

        # Training loop
        start_time = time.time()

        # Start by random exploration
        obs, _ = self.envs.reset(seed=self.args.seed)
        for exploration_step in range(self.args.learning_starts):
            actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()

            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

        # Learning starts here
        obs, _ = self.envs.reset(seed=self.args.seed)
        ref_buffer, actual_buffer, self.control_buffer = [], [], []      # For recording trajectory execution
        for global_step in range(self.args.learning_starts, self.args.total_timesteps + self.args.finetune_steps):
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                actions = self.actor(torch.Tensor(obs).to(self.device))
                if global_step <= self.args.total_timesteps:
                    actions += torch.normal(0, self.actor.action_scale * self.args.exploration_noise)
                actions = actions.cpu().numpy().clip(
                    self.envs.single_action_space.low,
                    self.envs.single_action_space.high
                )

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

            # If episode terminated: record reward, update dual network
            if "final_info" in infos:
                for info in infos["final_info"]:
                    # Record reward and information for plotting
                    self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                    # Record the reference and executed trajectories
                    nominal_Nx = np.prod(self.nominal_env.observation_space.shape)
                    nominal_x0 = info["actual"][0, :nominal_Nx].numpy()
                    ref_traj = info["reference"].to(self.device)
                    actual_traj = info["actual"].to(self.device)
                    actual_traj = actual_traj.reshape(
                        1, -1, nominal_Nx
                    )[:, :, self.tracking_mask].flatten(1)
                    ref_buffer.append(ref_traj)
                    actual_buffer.append(actual_traj)
                    with torch.no_grad():
                        v = self.dual_network(actual_traj[:, :nominal_Nx]).detach()
                        diff = ref_traj - v - actual_traj
                        self.writer.add_scalar("charts/rx_diff", diff.abs().mean().item(), global_step)
                        # Register nominal
                        rel_cost = -info["nominal_return"]
                        if self.opt_cost_fn:
                            rel_cost /= self.opt_cost_fn(nominal_x0)
                        self.writer.add_scalar(
                            "charts/rel_ret", rel_cost, global_step
                        )

                    # Update the dual network if we have collected enough trajectories
                    if len(ref_buffer) >= self.args.dual_batch_size:
                        if global_step > self.warmstart_steps:
                            ref_buffer = torch.vstack(ref_buffer).to(self.device)
                            actual_buffer = torch.vstack(actual_buffer).to(self.device)
                            nominal_Nx = np.prod(self.nominal_env.observation_space.shape)
                            v = self.dual_network(actual_buffer[:, :nominal_Nx])
                            diff = (ref_buffer - v.detach().clone() - actual_buffer)
                            v_obj = -(v * diff).sum() / self.args.dual_batch_size # Negative bc maximize
                            self.dual_network_optimizer.zero_grad()
                            v_obj.backward()
                            self.dual_network_optimizer.step()
                        self.reset_reference_generation(global_step)
                        ref_buffer, actual_buffer = [], []  # Clear the dual buffer 
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step <= self.args.total_timesteps:
                data = self.rb.sample(self.args.batch_size)
                with torch.no_grad():
                    clipped_noise = (
                        torch.randn_like(data.actions, device=self.device) *
                        self.args.policy_noise
                    ).clamp(
                        -self.args.noise_clip, self.args.noise_clip
                    ) * self.target_actor.action_scale

                    next_state_actions = (
                        self.target_actor(data.next_observations) + clipped_noise
                    ).clamp(
                        self.envs.single_action_space.low[0],
                        self.envs.single_action_space.high[0]
                    )

                    qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if global_step % self.args.policy_frequency == 0:
                    actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # update the target network
                    tau = self.args.tau
                    for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                if global_step % 200 == 0:
                    self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if global_step % 500 == 0 or global_step == self.args.total_timesteps + self.args.finetune_steps - 1:
                val_diff, val_rel_ret, extra_vals = self.validate_current_policy()
                self.writer.add_scalar("charts/val_rx_diff", val_diff, global_step)
                self.writer.add_scalar("charts/val_rel_ret", val_rel_ret, global_step)
                for (val_name, _), val_value in zip(self.extra_validation_fns, extra_vals):
                    self.writer.add_scalar(f"charts/{val_name}", val_value, global_step)

            # Saving checkpoints every defaultly 100,000 steps
            checkpoint_interval = self.args.ckpt_steps  # Adjust as needed
            if global_step % checkpoint_interval == 0:
                self._save_checkpoint(global_step)

        # Optionally, save final models outside the loop
        self._save_checkpoint(global_step)
        
    def close(self):
        self.envs.close()
        self.writer.close()
