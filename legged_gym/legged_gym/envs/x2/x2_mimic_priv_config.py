# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class X2MimicPrivCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096

        n_demo_steps = 2
        n_demo = 23 + 3 + 3 + 3 + 11*3  #observe height: 23 DOFs + 3 vel + 3 ang_vel + 3 (roll, pitch, height) + 11 key bodies * 3
        interval_demo_steps = 0.1

        n_scan = 0#132
        n_priv = 3
        n_priv_latent = 4 + 1 + 23*2
        n_proprio = 3 + 2 + 2 + 23*3 + 2 # one hot
        history_len = 10

        extra_history_len = 25

        prop_hist_len = 4
        n_feature = prop_hist_len * n_proprio

        n_teacher_priv = 11*3 + 11*3 + 3  # key_body_diff (11 bodies * 3) + cur_local_key_body_pos (11 bodies * 3) + base_lin_vel (3) = 69

        num_observations = n_feature + n_proprio + n_teacher_priv + n_demo + history_len*n_proprio + n_priv_latent + n_priv

        episode_length_s = 50 # episode length in seconds
        
        num_actions = 23
        
        num_policy_actions = 23
    
    class motion:
        motion_curriculum = True
        motion_type = "yaml"
        motion_name = "motions_autogen_all_no_run_jump.yaml"
        motion_folder = "/home/descfly/Lingyun/exbody2/x2_t2_23dof_retarget_motion_dynamic_adjust"

        global_keybody = False
        global_keybody_reset_time = 2

        num_envs_as_motions = False

        no_keybody = False
        regen_pkl = False

        step_inplace_prob = 0.05
        resample_step_inplace_interval_s = 10


    class terrain( LeggedRobotCfg.terrain ):
        horizontal_scale = 0.1 # [m] influence computation time by a lot
        height = [0., 0.04]
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.68] # x,y,z [m] - aligned with x2_t2_23dof.yaml
        default_joint_angles = { # = target angles [rad] when action = 0.0 - aligned with x2_t2_23dof.yaml
           'left_hip_yaw_joint' : 0.0,   
           'left_hip_roll_joint' : 0.0,               
           'left_hip_pitch_joint' : 0.0,         
           'left_knee_joint' : 0.4,       
           'left_ankle_pitch_joint' : -0.2,    
           'left_ankle_roll_joint' : 0.0, 
           'right_hip_yaw_joint' : 0.0, 
           'right_hip_roll_joint' : 0.0, 
           'right_hip_pitch_joint' : 0.0,                                       
           'right_knee_joint' : 0.4,                                             
           'right_ankle_pitch_joint' : -0.2,
           'right_ankle_roll_joint' : 0.0,                                     
           'waist_yaw_joint' : 0.0, 
           'waist_pitch_joint' : 0.0,
           'waist_roll_joint' : 0.0,
           'left_shoulder_pitch_joint' : 0.0, 
           'left_shoulder_roll_joint' : 0.0, 
           'left_shoulder_yaw_joint' : 0.0,
           'left_elbow_pitch_joint' : 0.0,
           'right_shoulder_pitch_joint' : 0.0,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.0,
           'right_elbow_pitch_joint' : 0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters - aligned with x2_t2_23dof.yaml:
        control_type = 'P'
        stiffness = {'hip_yaw': 200,
                     'hip_pitch': 200,
                     'hip_roll': 200,
                     'knee': 200,
                     'ankle_pitch': 40,
                     'ankle_roll': 40,
                     'waist_yaw': 100,
                     'waist_pitch': 100,
                     'waist_roll': 100,
                     'shoulder_pitch': 100,
                     'shoulder_roll': 100,
                     'shoulder_yaw': 100,
                     "elbow": 100,  # elbow_pitch in YAML
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 4.0,
                     'hip_pitch': 4.0,
                     'hip_roll': 4.0,
                     'knee': 4.0,
                     'ankle_pitch': 2.0,
                     'ankle_roll': 2.0,
                     'waist_yaw': 2.0,
                     'waist_pitch': 2.0,
                     'waist_roll': 2.0,
                     'shoulder_pitch': 2.0,
                     'shoulder_roll': 2.0,
                     'shoulder_yaw': 2.0,
                     "elbow": 2.0,  # elbow_pitch in YAML
                     }  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 10 # 4

    class normalization( LeggedRobotCfg.normalization):
        clip_actions = 10

    class asset( LeggedRobotCfg.asset ):
        urdf_file = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/x2/urdf/x2_t2_jw_collision_kungfu.urdf'
        mjcf_file = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/x2/x2_mc_kungfu.xml'
        torso_name = "waist_roll_link"  # aligned with x2_t2_23dof.yaml
        foot_name = "ankle_roll"
        hip_names = ["left_hip_yaw_joint", "right_hip_yaw_joint"]
        penalize_contacts_on = ["pelvis", "shoulder", "hip", "torso", "waist", "elbow", "knee", "head"]  # aligned with x2_t2_23dof.yaml
        terminate_after_contacts_on = []  # aligned with x2_t2_23dof.yaml (empty list)
        self_collisions = 0 # 0 to enable, 1 to disable - aligned with x2_t2_23dof.yaml
        armature = 0.001  # aligned with x2_t2_23dof.yaml - stablize semi-euler integration for end effectors
    
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # tracking rewards
            alive = 3
            tracking_lin_vel = 10

            tracking_demo_yaw = 1
            tracking_demo_roll_pitch = 1
            orientation = -4
            tracking_demo_dof_pos = 6
            tracking_demo_key_body = 10

            dof_acc = -3e-7
            action_rate = -0.1
            dof_error = -0.1
            feet_stumble = -2
            dof_pos_limits = -10.0
            feet_air_time = 10
            feet_force = -3e-3
            ankle_action = -0.1
            waist_roll_pitch_error = -0.5

        only_positive_rewards = False
        clip_rewards = True
        soft_dof_pos_limit = 0.95
        base_height_target = 0.25
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_gravity = True
        gravity_rand_interval_s = 10
        gravity_range = [-0.1, 0.1]
    
    class noise():
        add_noise = True
        noise_scale = 0.5 # scales other values
        class noise_scales():
            dof_pos = 0.01
            dof_vel = 0.15
            ang_vel = 0.3
            imu = 0.2
            
    class x2_params:
        height_factor = 1.3/1.8
        # height_factor = 1.0
        max_init_height = 0.6
        min_init_height = 0.5
        max_vel = 5.0
    
    class extend_config:
        """Extended body configuration for motion tracking links.
        Compatible with x2_t2_23dof.yaml extend_config structure.
        """
        nums_extend_bodies = 3
        
        class item0:
            """Left hand link extension"""
            joint_name = "left_hand_link"
            parent_name = "left_elbow_pitch_link"
            pos = [0.0, 0.0, -0.21]  # x, y, z [m]
            rot = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z [quat]
        
        class item1:
            """Right hand link extension"""
            joint_name = "right_hand_link"
            parent_name = "right_elbow_pitch_link"
            pos = [0.0, 0.0, -0.21]  # x, y, z [m]
            rot = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z [quat]
        
        class item2:
            """Head link extension"""
            joint_name = "head_link"
            parent_name = "waist_roll_link"
            pos = [0.0, 0.0, 0.4]  # x, y, z [m]
            rot = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z [quat]
        
        def __iter__(self):
            """Make extend_config iterable by yielding all item classes"""
            for i in range(self.nums_extend_bodies):
                item_attr = getattr(self, f'item{i}', None)
                if item_attr is not None:
                    yield item_attr

        def __len__(self):
            return self.nums_extend_bodies

class X2MimicPrivCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimicPriv"
        policy_class_name = 'ActorCriticMimicPriv'
        algorithm_class_name = 'PPOMimicPriv'
    
    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = False
        text_feat_input_dim = X2MimicPrivCfg.env.n_feature
        text_feat_output_dim = 16
        feat_hist_len = X2MimicPrivCfg.env.prop_hist_len
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005
        grad_penalty_coef_schedule = [0.0001, 0.0001, 700, 1000]

    class estimator:
        train_with_estimated_states = False
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = X2MimicPrivCfg.env.n_priv
        priv_start = X2MimicPrivCfg.env.n_feature + X2MimicPrivCfg.env.n_proprio + X2MimicPrivCfg.env.n_teacher_priv + X2MimicPrivCfg.env.n_demo + X2MimicPrivCfg.env.n_scan
        
        prop_start = X2MimicPrivCfg.env.n_feature
        prop_dim = X2MimicPrivCfg.env.n_proprio

class X2MimicPrivDistillCfgPPO( X2MimicPrivCfgPPO ):
    class distill:
        num_demo = X2MimicPrivCfg.env.n_demo
        num_steps_per_env = 24
        
        num_pretrain_iter = 0

        activation = "elu"
        learning_rate = 1.e-4
        student_actor_hidden_dims = [1024, 1024, 512]

        num_mini_batches = 4

        num_student_history = X2MimicPrivCfg.env.extra_history_len

