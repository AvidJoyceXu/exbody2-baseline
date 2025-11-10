# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Evaluation script for X2 motion tracking performance
# Computes success rate, mpbpe (mean position body error), and mpbve (mean velocity body error)

# IMPORTANT: isaacgym must be imported before torch
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
from pathlib import Path

from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.envs.x2.x2_mimic_priv import global_to_local

import torch

class MotionEvaluator:
    def __init__(self, env, policy, device, window_length=8):
        self.env = env
        self.policy = policy
        self.device = device
        self.window_length = window_length
        
        # Metrics storage
        self.metrics = {
            'single_motion': defaultdict(lambda: {'success': 0, 'total': 0, 'mpbpe': [], 'mpbve': []}),
            'motion_switch': defaultdict(lambda: {'success': 0, 'total': 0, 'mpbpe': [], 'mpbve': []})
        }
        
        # Episode tracking
        self.episode_failed = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
        self.current_motion_ids = None
        self.switch_info = {}  # Track motion switches: {env_id: [(motion1, motion2, switch_step)]}
        
    def compute_window_errors(self, env_ids=None):
        """Compute window-based position and velocity errors"""
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.device)
        
        if len(env_ids) == 0:
            return None, None
        
        # Get current motion times
        motion_times = self.env._motion_times[env_ids]
        motion_ids = self.env._motion_ids[env_ids]
        
        # Create time window: [current - window_length, current + window_length]
        time_offsets = torch.arange(-self.window_length, self.window_length, device=self.device, dtype=torch.float32) * self.env.dt
        motion_time_window = (motion_times.unsqueeze(1) + time_offsets.unsqueeze(0)).reshape(-1)
        
        # Get reference motion states for the window
        motion_ids_window = motion_ids.unsqueeze(1).repeat(1, self.window_length * 2).reshape(-1)
        motion_state_window = self.env._motion_lib.get_motion_state(motion_ids_window, motion_time_window)
        
        # Get reference body positions and velocities (all bodies)
        # MotionLibRobotWTS returns rg_pos as [num_samples, num_bodies, 3]
        ref_rg_pos = motion_state_window['rg_pos']  # [num_samples, num_bodies, 3]
        ref_root_pos = motion_state_window['root_pos']  # [num_samples, 3]
        ref_root_rot = motion_state_window['root_rot']  # [num_samples, 4]
        
        # Try to get body velocities - check different possible keys
        if 'body_vel_t' in motion_state_window:
            ref_body_vel = motion_state_window['body_vel_t']
        elif 'body_vel' in motion_state_window:
            ref_body_vel = motion_state_window['body_vel']
        else:
            # Fallback: use root velocity for all bodies
            ref_body_vel = ref_root_pos.unsqueeze(1).repeat(1, ref_rg_pos.shape[1], 1)
        
        # Reshape to [num_envs, window_length*2, num_bodies, 3]
        # ref_rg_pos and ref_body_vel are already [num_samples, num_bodies, 3]
        num_samples_expected = len(env_ids) * self.window_length * 2
        
        # Ensure we have the correct number of samples
        if ref_rg_pos.shape[0] != num_samples_expected:
            # This shouldn't happen, but handle it gracefully
            actual_samples = min(ref_rg_pos.shape[0], num_samples_expected)
            ref_rg_pos = ref_rg_pos[:actual_samples]
            ref_root_pos = ref_root_pos[:actual_samples]
            ref_root_rot = ref_root_rot[:actual_samples]
            ref_body_vel = ref_body_vel[:actual_samples]
            num_samples_expected = actual_samples
        
        num_bodies_pos = ref_rg_pos.shape[1]
        num_bodies_vel = ref_body_vel.shape[1]
        num_bodies = max(num_bodies_pos, num_bodies_vel)  # Use the larger number
        
        # Reshape: [num_samples, num_bodies, 3] -> [num_envs, window_length*2, num_bodies, 3]
        ref_rg_pos = ref_rg_pos.view(len(env_ids), self.window_length * 2, num_bodies_pos, 3)
        ref_root_pos = ref_root_pos.view(len(env_ids), self.window_length * 2, 3)
        ref_root_rot = ref_root_rot.view(len(env_ids), self.window_length * 2, 4)
        ref_body_vel = ref_body_vel.view(len(env_ids), self.window_length * 2, num_bodies_vel, 3)
        
        # Pad or trim body positions and velocities to match num_bodies
        if num_bodies_pos < num_bodies:
            padding = torch.zeros(len(env_ids), self.window_length * 2, num_bodies - num_bodies_pos, 3,
                                device=ref_rg_pos.device, dtype=ref_rg_pos.dtype)
            ref_rg_pos = torch.cat([ref_rg_pos, padding], dim=2)
        elif num_bodies_pos > num_bodies:
            ref_rg_pos = ref_rg_pos[:, :, :num_bodies]
        
        if num_bodies_vel < num_bodies:
            padding = torch.zeros(len(env_ids), self.window_length * 2, num_bodies - num_bodies_vel, 3,
                                device=ref_body_vel.device, dtype=ref_body_vel.dtype)
            ref_body_vel = torch.cat([ref_body_vel, padding], dim=2)
        elif num_bodies_vel > num_bodies:
            ref_body_vel = ref_body_vel[:, :, :num_bodies]
        
        # Get current robot body positions and velocities
        robot_root_pos = self.env.root_states[env_ids, :3]  # [num_envs, 3]
        robot_root_rot = self.env.root_states[env_ids, 3:7]  # [num_envs, 4]
        
        # Get robot body positions - need to match number of bodies
        num_robot_bodies = min(self.env.rigid_body_states.shape[1], num_bodies)
        robot_body_pos = self.env.rigid_body_states[env_ids, :num_robot_bodies, :3]  # [num_envs, num_robot_bodies, 3]
        robot_body_vel = self.env.rigid_body_states[env_ids, :num_robot_bodies, 7:10]  # [num_envs, num_robot_bodies, 3]
        
        # Pad or trim to match reference
        if num_robot_bodies < num_bodies:
            # Pad with zeros
            padding = torch.zeros(len(env_ids), num_bodies - num_robot_bodies, 3, device=self.device)
            robot_body_pos = torch.cat([robot_body_pos, padding], dim=1)
            robot_body_vel = torch.cat([robot_body_vel, padding], dim=1)
        elif num_robot_bodies > num_bodies:
            # Trim
            robot_body_pos = robot_body_pos[:, :num_bodies]
            robot_body_vel = robot_body_vel[:, :num_bodies]
        
        # Convert reference positions to local frame (relative to root)
        ref_local_pos = global_to_local(
            ref_root_rot.view(-1, 4),
            (ref_rg_pos - ref_root_pos.unsqueeze(2)).view(-1, num_bodies, 3),
            ref_root_pos.view(-1, 3)
        ).view(len(env_ids), self.window_length * 2, num_bodies, 3)
        
        # Convert robot positions to local frame
        robot_local_pos = global_to_local(
            robot_root_rot,
            robot_body_pos - robot_root_pos.unsqueeze(1),
            robot_root_pos
        )  # [num_envs, num_bodies, 3]
        
        # Convert robot velocities to local frame
        robot_local_vel = quat_rotate_inverse(
            robot_root_rot.unsqueeze(1).repeat(1, num_bodies, 1).view(-1, 4),
            robot_body_vel.view(-1, 3)
        ).view(len(env_ids), num_bodies, 3)
        
        # Convert reference velocities to local frame
        ref_local_vel = quat_rotate_inverse(
            ref_root_rot.view(-1, 4).unsqueeze(1).repeat(1, num_bodies, 1).view(-1, 4),
            ref_body_vel.view(-1, 3)
        ).view(len(env_ids), self.window_length * 2, num_bodies, 3)
        
        # Compute position differences: [num_envs, window_length*2, num_bodies, 3]
        pos_diff = robot_local_pos.unsqueeze(1) - ref_local_pos
        
        # Compute velocity differences: [num_envs, window_length*2, num_bodies, 3]
        vel_diff = robot_local_vel.unsqueeze(1) - ref_local_vel
        
        # Compute mean error per body, then min across window: [num_envs]
        pos_error_per_window = pos_diff.norm(dim=-1).mean(dim=-1)  # [num_envs, window_length*2]
        vel_error_per_window = vel_diff.norm(dim=-1).mean(dim=-1)  # [num_envs, window_length*2]
        
        # Take minimum across window
        min_pos_error = pos_error_per_window.min(dim=-1).values  # [num_envs]
        min_vel_error = vel_error_per_window.min(dim=-1).values  # [num_envs]
        
        return min_pos_error, min_vel_error
    
    def check_success(self, env_ids, pos_error):
        """Check if episode is successful: not fallen and position error < 0.2m"""
        # Check if robot has fallen (height < threshold)
        height = self.env.root_states[env_ids, 2]
        not_fallen = height > 0.2  # Same threshold as in check_termination
        
        # Check position error
        pos_ok = pos_error < 0.2
        
        success = not_fallen & pos_ok
        return success.cpu() if isinstance(success, torch.Tensor) else success
    
    def evaluate_single_motion(self, motion_name, num_episodes=10, episode_length_s=30):
        """Evaluate a single motion without switching"""
        print(f"Evaluating single motion: {motion_name}")
        
        # Initialize metrics for this motion if not already done
        if motion_name not in self.metrics['single_motion']:
            self.metrics['single_motion'][motion_name] = {'success': 0, 'total': 0, 'mpbpe': [], 'mpbve': []}
        
        # Find motion ID
        motion_id = None
        motion_keys = self.env._motion_lib._motion_data_keys
        if isinstance(motion_keys, list):
            keys_list = motion_keys
        else:
            keys_list = motion_keys.tolist()
        
        print(f"  Searching for motion '{motion_name}' in {len(keys_list)} motion keys...")
        print(f"  First few keys: {keys_list[:5]}")
        
        for i, key in enumerate(keys_list):
            # Try multiple matching strategies
            key_str = str(key)
            # Match motion name (key format: "motion_name_1", "motion_name_2", etc.)
            if motion_name == key_str.split('_')[0] or motion_name in key_str or key_str.startswith(motion_name):
                motion_id = i
                print(f"  Found motion '{motion_name}' at index {i} with key '{key_str}'")
                break
        
        if motion_id is None:
            print(f"  ERROR: Motion '{motion_name}' not found in motion keys!")
            print(f"  Available keys (first 10): {keys_list[:10]}")
            return
        
        episode_count = 0
        step_count = 0
        max_steps = int(episode_length_s / self.env.dt)
        
        print(f"  Running {num_episodes} episodes, max {max_steps} steps per episode")
        
        # Reset environment with specific motion
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        self.env._motion_ids[env_ids] = motion_id
        self.env.update_motion_ids(env_ids)
        self.env.reset_idx(env_ids)
        obs = self.env.get_observations()
        
        print(f"  Starting evaluation loop...")
        
        while episode_count < num_episodes:
            # Run policy
            actions = self.policy(obs, hist_encoding=True)
            
            obs, _, _, dones, _ = self.env.step(actions.detach())
            
            # Compute metrics for non-reset environments
            active_envs = ~dones
            if active_envs.any() and step_count > self.window_length:  # Wait for window to be valid
                active_env_ids = env_ids[active_envs]
                pos_error, vel_error = self.compute_window_errors(active_env_ids)
                if pos_error is not None:
                    success = self.check_success(active_env_ids, pos_error)
                    
                    for i, env_idx in enumerate(active_env_ids):
                        self.metrics['single_motion'][motion_name]['mpbpe'].append(pos_error[i].item())
                        self.metrics['single_motion'][motion_name]['mpbve'].append(vel_error[i].item())
            
            # Check for resets
            reset_envs = dones.nonzero(as_tuple=False).flatten()
            if len(reset_envs) > 0:
                for env_idx in reset_envs:
                    # Check if episode was successful before reset
                    if not self.episode_failed[env_idx]:
                        self.metrics['single_motion'][motion_name]['success'] += 1
                    self.metrics['single_motion'][motion_name]['total'] += 1
                    episode_count += 1
                    if episode_count % 5 == 0:
                        print(f"  Completed {episode_count}/{num_episodes} episodes")
                
                # Reset failed flag
                self.episode_failed[reset_envs] = False
                
                # Reset with same motion
                self.env._motion_ids[reset_envs] = motion_id
                self.env.update_motion_ids(reset_envs)
                self.env.reset_idx(reset_envs)
            
            # Check for failures (falling)
            height = self.env.root_states[:, 2]
            self.episode_failed = self.episode_failed | (height < 0.2)
            
            step_count += 1
            if step_count >= max_steps:
                # Timeout - count remaining episodes as incomplete
                print(f"  Timeout reached at step {step_count}, completed {episode_count}/{num_episodes} episodes")
                break
        
        # Final summary for this motion
        final_data = self.metrics['single_motion'][motion_name]
        print(f"  Completed evaluation: {final_data['total']} episodes, {final_data['success']} successful, "
              f"{len(final_data['mpbpe'])} error measurements")
    
    def evaluate_motion_switch(self, motion1_name, motion2_name, num_episodes=10, switch_time_s=5.0):
        """Evaluate motion switching from motion1 to motion2"""
        print(f"Evaluating motion switch: {motion1_name} -> {motion2_name}")
        
        switch_key = f"{motion1_name}->{motion2_name}"
        # Initialize metrics for this switch if not already done
        if switch_key not in self.metrics['motion_switch']:
            self.metrics['motion_switch'][switch_key] = {'success': 0, 'total': 0, 'mpbpe': [], 'mpbve': []}
        
        # Find motion IDs
        motion1_id = None
        motion2_id = None
        motion_keys = self.env._motion_lib._motion_data_keys
        if isinstance(motion_keys, list):
            keys_list = motion_keys
        else:
            keys_list = motion_keys.tolist()
        
        for i, key in enumerate(keys_list):
            key_str = str(key)
            key_base = key_str.split('_')[0]
            if motion1_name == key_base or motion1_name in key_str or key_str.startswith(motion1_name):
                motion1_id = i
            if motion2_name == key_base or motion2_name in key_str or key_str.startswith(motion2_name):
                motion2_id = i
        
        if motion1_id is None or motion2_id is None:
            print(f"  ERROR: Motions not found (motion1_id={motion1_id}, motion2_id={motion2_id}), skipping")
            return
        
        print(f"  Found motion1_id={motion1_id}, motion2_id={motion2_id}")
        
        switch_step = int(switch_time_s / self.env.dt)
        
        episode_count = 0
        step_count = 0
        
        # Reset environment with motion1
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        self.env._motion_ids[env_ids] = motion1_id
        self.env.update_motion_ids(env_ids)
        self.env.reset_idx(env_ids)
        obs = self.env.get_observations()
        
        switched = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        
        while episode_count < num_episodes:
            # Switch motion at specified time
            if step_count == switch_step:
                switch_mask = ~switched
                if switch_mask.any():
                    self.env._motion_ids[switch_mask] = motion2_id
                    self.env.update_motion_ids(switch_mask)
                    switched[switch_mask] = True
                    
                    # Record switch info
                    for env_idx in env_ids[switch_mask]:
                        if env_idx.item() not in self.switch_info:
                            self.switch_info[env_idx.item()] = []
                        self.switch_info[env_idx.item()].append((motion1_id, motion2_id, step_count))
            
            # Run policy
            actions = self.policy(obs, hist_encoding=True)
            
            obs, _, _, dones, _ = self.env.step(actions.detach())
            
            # Compute metrics after switch (wait for window to be valid)
            if step_count >= switch_step + self.window_length:
                active_envs = switched & ~dones
                if active_envs.any():
                    active_env_ids = env_ids[active_envs]
                    pos_error, vel_error = self.compute_window_errors(active_env_ids)
                    if pos_error is not None:
                        success = self.check_success(active_env_ids, pos_error)
                        
                        for i, env_idx in enumerate(active_env_ids):
                            self.metrics['motion_switch'][switch_key]['mpbpe'].append(pos_error[i].item())
                            self.metrics['motion_switch'][switch_key]['mpbve'].append(vel_error[i].item())
            
            # Check for resets
            reset_envs = dones.nonzero(as_tuple=False).flatten()
            if len(reset_envs) > 0:
                for env_idx in reset_envs:
                    if switched[env_idx]:
                        # Check if episode was successful before reset
                        if not self.episode_failed[env_idx]:
                            self.metrics['motion_switch'][switch_key]['success'] += 1
                        self.metrics['motion_switch'][switch_key]['total'] += 1
                        episode_count += 1
                
                # Reset failed flag and switched flag
                self.episode_failed[reset_envs] = False
                switched[reset_envs] = False
                
                # Reset with motion1
                self.env._motion_ids[reset_envs] = motion1_id
                self.env.update_motion_ids(reset_envs)
                self.env.reset_idx(reset_envs)
                step_count = -1  # Will be incremented to 0
            
            # Check for failures
            height = self.env.root_states[:, 2]
            self.episode_failed = self.episode_failed | (height < 0.2)
            
            step_count += 1
    
    def get_summary(self):
        """Get summary of all metrics"""
        summary = {}
        
        # Single motion metrics
        summary['single_motion'] = {}
        for motion_name, data in self.metrics['single_motion'].items():
            if data['total'] > 0:
                summary['single_motion'][motion_name] = {
                    'success_rate': data['success'] / data['total'],
                    'mpbpe': np.mean(data['mpbpe']) if data['mpbpe'] else 0.0,
                    'mpbve': np.mean(data['mpbve']) if data['mpbve'] else 0.0,
                    'num_episodes': data['total']
                }
        
        # Motion switch metrics
        summary['motion_switch'] = {}
        for switch_key, data in self.metrics['motion_switch'].items():
            if data['total'] > 0:
                summary['motion_switch'][switch_key] = {
                    'success_rate': data['success'] / data['total'],
                    'mpbpe': np.mean(data['mpbpe']) if data['mpbpe'] else 0.0,
                    'mpbve': np.mean(data['mpbve']) if data['mpbve'] else 0.0,
                    'num_episodes': data['total']
                }
        
        return summary


def load_config_overrides(config_path):
    """Load config overrides from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def eval_main(args):
    # Load environment and policy
    log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.proj_name, args.exptid)
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Override configs based on evaluation mode
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 30
    env_cfg.noise.add_noise = False
    
    # Load domain randomization config
    if args.domain_rand_config:
        from omegaconf import OmegaConf, ListConfig, DictConfig
        dr_config = OmegaConf.load(args.domain_rand_config)
        # Merge domain randomization settings, converting OmegaConf types to native Python types
        for key, value in dr_config.domain_rand.items():
            # Convert OmegaConf types to native Python types
            if isinstance(value, (ListConfig, DictConfig)):
                value = OmegaConf.to_container(value, resolve=True)
            
            # Special handling for push_interval_s: if it's a list (range), use first element
            # since legged_gym expects a scalar, not a range
            if key == 'push_interval_s' and isinstance(value, list) and len(value) == 2:
                value = value[0]  # Use first element of range
            
            setattr(env_cfg.domain_rand, key, value)
    
    # Load terrain config
    if args.terrain_config:
        from omegaconf import OmegaConf, ListConfig, DictConfig
        terrain_config = OmegaConf.load(args.terrain_config)
        # Merge terrain settings, converting OmegaConf types to native Python types
        for key, value in terrain_config.terrain.items():
            # Convert OmegaConf types to native Python types
            if isinstance(value, (ListConfig, DictConfig)):
                value = OmegaConf.to_container(value, resolve=True)
            
            # Skip incompatible terrain settings for plane terrain
            if key == 'mesh_type' and value == 'trimesh':
                # Force plane terrain for evaluation to avoid terrain creation issues
                value = 'plane'
            
            setattr(env_cfg.terrain, key, value)
        
        # Ensure num_goals is set for plane terrain compatibility
        if not hasattr(env_cfg.terrain, 'num_goals') or env_cfg.terrain.num_goals is None:
            env_cfg.terrain.num_goals = 1
    
    # Create environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # Load policy
    train_cfg.runner.resume = True
    # Set resumeid to use the correct log path for model loading
    args.resumeid = args.exptid
    ppo_runner, _, _ = task_registry.make_alg_runner(
        log_root=log_pth, env=env, name=args.task, args=args, 
        train_cfg=train_cfg, return_log_dir=True
    )
    
    if_distill = ppo_runner.if_distill
    
    # Create policy wrapper
    class PolicyWrapper:
        def __init__(self, ppo_runner, if_distill, env):
            self.ppo_runner = ppo_runner
            self.if_distill = if_distill
            self.env = env
        
        def __call__(self, obs, obs_hist=None, hist_encoding=False):
            if self.if_distill:
                if obs_hist is None:
                    obs_hist = self.env.get_extra_hist_obs()
                return self.ppo_runner.alg.student_actor(
                    obs.detach(), obs_hist, hist_encoding=hist_encoding
                )
            else:
                policy_fn = self.ppo_runner.get_inference_policy(device=self.env.device)
                return policy_fn(obs.detach(), hist_encoding=hist_encoding)
    
    policy = PolicyWrapper(ppo_runner, if_distill, env)
    
    # Create evaluator
    evaluator = MotionEvaluator(env, policy, env.device, window_length=args.window_length)
    
    # Get motion list
    motion_names = []
    if args.motion_list:
        with open(args.motion_list, 'r') as f:
            motion_names = [line.strip() for line in f.readlines()]
    else:
        # Default: get all motions from motion library
        # Extract unique motion names from keys
        if hasattr(env._motion_lib, '_motion_data_keys'):
            keys = env._motion_lib._motion_data_keys
            if isinstance(keys, list):
                keys_list = keys
            else:
                keys_list = keys.tolist()
            
            print(f"Extracting motion names from {len(keys_list)} motion keys...")
            
            # Try to extract motion names - handle different key formats
            motion_names_set = set()
            for key in keys_list:
                key_str = str(key)
                # Try different extraction strategies
                # Format 1: "motion_name_1", "motion_name_2" -> "motion_name"
                # Format 2: "motion_name" -> "motion_name"
                # Format 3: Full path or complex name
                parts = key_str.split('_')
                if len(parts) > 1 and parts[-1].isdigit():
                    # Has numeric suffix, remove it
                    motion_name = '_'.join(parts[:-1])
                else:
                    # No numeric suffix, use full name or first part
                    motion_name = parts[0] if len(parts) > 0 else key_str
                motion_names_set.add(motion_name)
            
            motion_names = sorted(list(motion_names_set))
        else:
            # Fallback: use number of unique motions
            num_motions = env._motion_lib._num_unique_motions
            motion_names = [f"motion_{i}" for i in range(num_motions)]
    
    print(f"Found {len(motion_names)} unique motions: {motion_names}")
    
    # Evaluate single motions
    if args.eval_single:
        print("\n" + "="*50)
        print("Evaluating single motions")
        print("="*50)
        for motion_name in motion_names:
            evaluator.evaluate_single_motion(motion_name, num_episodes=args.num_episodes)
    
    # Evaluate motion switches
    if args.eval_switch:
        print("\n" + "="*50)
        print("Evaluating motion switches")
        print("="*50)
        for i, motion1 in enumerate(motion_names):
            for motion2 in motion_names[i+1:]:
                evaluator.evaluate_motion_switch(motion1, motion2, num_episodes=args.num_episodes)
    
    # Get summary
    summary = evaluator.get_summary()
    
    # Save results
    output_path = args.output_path or f"{log_pth}/eval_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    # Parse custom arguments
    parser = argparse.ArgumentParser(description='Evaluate X2 motion tracking performance')
    parser.add_argument('--task', type=str, required=True, help='Task name')
    parser.add_argument('--exptid', type=str, required=True, help='Experiment ID')
    parser.add_argument('--proj_name', type=str, default='x2_mimic_priv', help='Project name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--domain_rand_config', type=str, help='Path to domain randomization config YAML')
    parser.add_argument('--terrain_config', type=str, help='Path to terrain config YAML')
    parser.add_argument('--motion_list', type=str, help='Path to file with list of motion names (one per line)')
    parser.add_argument('--eval_single', action='store_true', help='Evaluate single motions')
    parser.add_argument('--eval_switch', action='store_true', help='Evaluate motion switches')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes per evaluation')
    parser.add_argument('--window_length', type=int, default=8, help='Window length for error computation')
    parser.add_argument('--output_path', type=str, help='Path to save evaluation results')
    
    # Parse custom args
    custom_args = parser.parse_args()
    
    # Create args object with required attributes for task_registry
    class Args:
        def __init__(self, custom_args):
            # Copy custom args
            for key, value in vars(custom_args).items():
                setattr(self, key, value)
            
            # Set device-related args
            self.sim_device = custom_args.device
            if ':' in custom_args.device:
                self.sim_device_type, device_id_str = custom_args.device.split(':')
                self.sim_device_id = int(device_id_str)
            else:
                self.sim_device_type = custom_args.device
                self.sim_device_id = 0
            
            # Set defaults for other required args
            self.headless = True
            self.compute_device_id = self.sim_device_id
            
            # Attributes needed by update_cfg_from_args
            self.use_camera = False
            self.motion_name = None
            self.motion_type = None
            self.num_envs = None
            self.seed = None
            self.task_both = False
            self.rows = None
            self.cols = None
            self.delay = False
            self.resume = False
            self.record_video = False
            self.record_frame = False
            self.fix_base = False
            self.regen_pkl = False
            self.max_iterations = None
            self.experiment_name = None
            self.run_name = None
            self.load_run = None
            self.checkpoint = -1
            
            # Attributes needed by parse_sim_params
            self.physics_engine = gymapi.SIM_PHYSX  # Default to PhysX
            self.use_gpu = (self.sim_device_type == 'cuda')
            self.subscenes = 0
            self.use_gpu_pipeline = (self.sim_device_type == 'cuda')
            self.num_threads = 0
            
            # Other attributes
            self.horovod = False
            self.rl_device = custom_args.device
            self.debug = False
            self.entity = None
            self.resumeid = None
            self.daggerid = None
            self.no_wandb = False
            self.nographics = False
            self.flex = False
            self.slices = None
            self.pipeline = 'GPU' if self.sim_device_type == 'cuda' else 'CPU'
    
    args = Args(custom_args)
    
    eval_main(args)

