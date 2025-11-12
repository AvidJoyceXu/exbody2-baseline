from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

import torch.nn.functional as F

import glob
import os.path as osp
import numpy as np
import joblib
import torch
import random
from typing import Dict, Any, Optional

from humanoidverse.utils.motion_lib.motion_utils.flags import flags
from enum import Enum
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from pathlib import Path
from copy import deepcopy
from easydict import EasyDict
from loguru import logger
from rich.progress import track

from isaac_utils.rotations import(
    quat_angle_axis,
    quat_inverse,
    quat_mul_norm,
    get_euler_xyz,
    normalize_angle,
    slerp,
    quat_to_exp_map,
    quat_to_angle_axis,
    quat_mul,
    quat_conjugate,
    calc_heading_quat_inv
)

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

class MotionlibMode(Enum):
    file = 1
    directory = 2


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)


def _calc_frame_blend(time, len, num_frames, dt):
    time = time.clone()
    phase = time / len
    # import ipdb;ipdb.set_trace()
    phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
    time[time < 0] = 0

    frame_idx0 = (phase * (num_frames - 1)).long()
    frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
    blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
    
    return frame_idx0, frame_idx1, blend


def _local_rotation_to_dof_smpl(local_rot):
    B, J, _ = local_rot.shape
    dof_pos = quat_to_exp_map(local_rot[:, 1:])
    return dof_pos.reshape(B, -1)
    

def forbidden(fn):
    def wrapper(*args, **kwargs):
        raise RuntimeError("You are NOT ALLOWED to call it.")
    return wrapper

# @time_prot_cls_dec_mlb
# TimePortion: 6%
class MotionLibBase():
    ############################################################ SETUP ############################################################
    
    def __init__(self, motion_lib_cfg, num_envs, device):

        def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
            self.fix_height = fix_height
            self.multi_thread = multi_thread
            
            #### Termination history
            self._curr_motion_ids = None
            self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
            self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
            self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
            # self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches

        self.m_cfg = motion_lib_cfg
        # self._sim_fps = 1/self.m_cfg.get("step_dt", 1/50)   # 没用上，不要误导人
        
        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None
        self.has_action = False
        self.has_contact_mask: Optional[str] = None
        skeleton_file = self.m_cfg.asset.mjcf_file
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)

        
        logger.info(f"Loaded skeleton from {skeleton_file}")
        # logger.info(f"Loading motion data from {self.m_cfg.motion_file}...")
        
        self.load_data(self.m_cfg.motion.motion_folder)
        setup_constants(self, fix_height = False,  multi_thread = False)
        
        self._curr_motion_ids = torch.zeros(self.num_envs).to(device=self._device,dtype=torch.long)

        if flags.real_traj:
            self.track_idx = self._motion_data_load[next(iter(self._motion_data_load))].get("track_idx", [19, 24, 29])
        return
        

    def load_data(self, motion_folder):
        """
        Load all .pkl files under motion_folder into memory as a list of motion dicts.
        Results:
        - self._motion_data_list: numpy array (dtype=object) of dicts, each dict is motion file content (already sliced per file)
        - self._motion_data_keys: numpy array of keys (string)
        - self._num_unique_motions: int
        """
        pths = sorted(glob.glob(osp.join(motion_folder, "*.pkl")))
        
        assert len(pths) > 0, f"No motion files found in {motion_folder}"

        motion_list = []
        keys = []
        for p in pths:
            data = joblib.load(p)
            # common cases: file contains {key: dict} or directly a dict of motion
            if isinstance(data, dict) and len(data) == 1:
                k, v = list(data.items())[0]
                motion_list.append(v)
                keys.append(k)
            elif isinstance(data, dict) and "pose_aa" in data:
                motion_list.append(data)
                keys.append(Path(p).stem)
            else:
                # fallback: try to take first dict-like entry
                # if data has many entries, user may want to adjust
                try:
                    # if it's dict of many entries, take values
                    if isinstance(data, dict):
                        k, v = next(iter(data.items()))
                        motion_list.append(v)
                        keys.append(k)
                    else:
                        raise RuntimeError("Unsupported pkl structure")
                except Exception as e:
                    raise RuntimeError(f"Cannot parse motion file {p}: {e}")

        self._motion_data_list = np.array(motion_list, dtype=object)
        self._motion_data_keys = keys # np.array(keys, dtype=object)

        self.name2label = {
            'jump': 0,
            'rot_jump': 1,
            'run': 2,
            'run_left': 3,
            'run_right': 4,
            'walk': 5,
            'stand': 6,
            'back_kick': 7,
            'jump_kick': 8,
            'punch': 9
        }
        self._motion_data_label = F.one_hot(
            torch.tensor([self.name2label.get(k.rsplit('_', 1)[0], -1) for k in self._motion_data_keys],device='cuda'),
            num_classes=len(self.name2label)
        ).to(torch.float32)     # 一个list，按照读取的顺序存放对应的label；而curr_id取的是这个list的index，所以curr_id不是和label等价的数字：curr_id是读取时排列的顺序，label是我们定义的动作类别编号
        
        self._num_unique_motions = len(self._motion_data_list)
        logger.info(f"Loaded {self._num_unique_motions} motion files from {motion_folder}")
            
            

    def load_motions(self, 
                    # random_sample=True, 
                    start_idx=0, 
                    max_len=-1, 
                    target_heading = None):
        """
        Load (and concatenate) the sampled motions into per-frame tensors.
        This version DOES NOT use FakeCat and supports loading different motions.
        """
        assert target_heading is None, "Not Allowed to use target_heading!"

        motions_curr = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa_list = []
        has_action = False
        _motion_actions_list = []
        _motion_contact_masks_list = []



        # import ipdb; ipdb.set_trace()
        # self.curr_motion_keys = self._motion_data_keys[sample_idxes.cpu()]
        # self.curr_motion_labels = self._motion_data_label[sample_idxes.cpu()]


        # build motion_data_list for those indices
        motion_data_list = self._motion_data_list# [sample_idxes.cpu().numpy()]


        # load each motion with skeleton (returns dict f -> (file_dict, curr_motion))
        res_acc = self.load_motion_with_skeleton(motion_data_list, self.fix_height, target_heading, max_len)

        # collect per-motion and per-frame tensors
        gts_list = []
        grs_list = []
        lrs_list = []
        grvs_list = []
        gravs_list = []
        gavs_list = []
        gvs_list = []
        dvs_list = []
        
        gts_t_list = []
        grs_t_list = []
        gvs_t_list = []
        gavs_t_list = []

        dof_pos_list = []  # if present

        for f in range(len(res_acc)):
            motion_file_data, curr_motion = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = curr_dt * (num_frames - 1)

            # # per-motion meta
            # if "beta" in motion_file_data:
            #     _motion_aa_list.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
            #     _motion_bodies.append(curr_motion.gender_beta if hasattr(curr_motion, "gender_beta") else torch.zeros(self.num_joints))
            # else:
            #     # fill zeros for aa
            #     _motion_aa_list.append(np.zeros((num_frames, self.num_joints * 3)))
            #     _motion_bodies.append(torch.zeros(self.num_joints))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)   # 有多少帧
            _motion_lengths.append(curr_len)        # 时间（秒）

            # append per-frame tensors (ensure float and moved to device)
            gts_list.append(curr_motion.global_translation.float().to(self._device))
            grs_list.append(curr_motion.global_rotation.float().to(self._device))
            lrs_list.append(curr_motion.local_rotation.float().to(self._device))
            grvs_list.append(curr_motion.global_root_velocity.float().to(self._device))
            gravs_list.append(curr_motion.global_root_angular_velocity.float().to(self._device))
            gavs_list.append(curr_motion.global_angular_velocity.float().to(self._device))
            gvs_list.append(curr_motion.global_velocity.float().to(self._device))
            dvs_list.append(curr_motion.dof_vels.float().to(self._device))

            gts_t_list.append(curr_motion.global_translation_extend.float().to(self._device))
            grs_t_list.append(curr_motion.global_rotation_extend.float().to(self._device))
            gvs_t_list.append(curr_motion.global_velocity_extend.float().to(self._device))
            gavs_t_list.append(curr_motion.global_angular_velocity_extend.float().to(self._device))
        
            dof_pos_list.append(curr_motion.dof_pos.float().to(self._device))

            _motion_contact_masks_list.append(curr_motion.contact_mask.float().to(self._device))

            motions_curr.append(curr_motion)

        # concat per-frame lists into big tensors
        self.gts = torch.cat(gts_list, dim=0)
        self.grs = torch.cat(grs_list, dim=0)
        self.lrs = torch.cat(lrs_list, dim=0)
        self.grvs = torch.cat(grvs_list, dim=0)
        self.gravs = torch.cat(gravs_list, dim=0)
        self.gavs = torch.cat(gavs_list, dim=0)
        self.gvs = torch.cat(gvs_list, dim=0)
        self.dvs = torch.cat(dvs_list, dim=0)
        self.gts_t = torch.cat(gts_t_list, dim=0)
        self.grs_t = torch.cat(grs_t_list, dim=0)
        self.gvs_t = torch.cat(gvs_t_list, dim=0)
        self.gavs_t = torch.cat(gavs_t_list, dim=0)
        self.dof_pos = torch.cat(dof_pos_list, dim=0)
        self.gts_t = torch.cat(gts_t_list, dim=0)
        self.grs_t = torch.cat(grs_t_list, dim=0)
        self.gvs_t = torch.cat(gvs_t_list, dim=0)
        self.gavs_t = torch.cat(gavs_t_list, dim=0)

        # motion-level tensors
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device, dtype=torch.long)

        # self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        # motion_aa: concat per-frame arrays (converted to tensor)
        # aa_concat = np.concatenate(_motion_aa_list, axis=0)
        # self._motion_aa = torch.tensor(aa_concat, device=self._device, dtype=torch.float32)

        # actions / contact masks concatenated per-frame if present


        self.has_contact_mask = "point"
        self._motion_contact_masks = torch.cat(_motion_contact_masks_list, dim=0).float().to(self._device)

        self._num_motions = len(motions_curr)
        # compute length starts (start frame index of each motion in the concatenated per-frame tensors)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)   # 循环右移
        lengths_shifted[0] = 0              # 第一个motion的起始位置是0
        self.length_starts = lengths_shifted.cumsum(0)  # 累加，得到全局索引

        # self.motion_ids = torch.arange(self._num_motions, dtype=torch.long, device=self._device)    # 这里的motion_ids太hard code了，适配于单条轨迹的写法
        # self.num_bodies = self.num_joints

        # # return motions_cur
        # if random_sample:
        #     sample_idxes = torch.multinomial(self._sampling_prob, num_samples=self.num_envs, replacement=True).to(self._device)
        # else:
        #     sample_idxes = torch.remainder(torch.arange(self.num_envs) + start_idx, self._num_unique_motions ).to(self._device)
    
        # self._curr_motion_ids = sample_idxes    # env2motion,感觉这个不该写在这里的，应该写在外边的reset里，回头改一下

    def load_motion_with_skeleton(self,
                                  motion_data_list: np.ndarray,
                                  fix_height,
                                  target_heading,
                                  max_len):
        
        # loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        res = {}
        for f in range(len(motion_data_list)):
            curr_file:Dict[str, Any] = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                forbidden(lambda :0)()
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]

                
            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]     # t,3
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()             # t,26,3(26是什么东西？？)
            # import ipdb; ipdb.set_trace()
            if "action" in curr_file.keys():
                self.has_action = True
            if "contact_mask" in curr_file.keys():
                contact_shape = curr_file['contact_mask'].shape
                assert len(contact_shape) ==2 and contact_shape[0] == seq_len
                self._contact_size = contact_shape[1]
                if contact_shape[1] == 2:
                    self.has_contact_mask = "point"
                else:
                    raise ValueError(f"Contact mask shape {contact_shape} is not supported")
            
            dt = 1/curr_file['fps']

            if self.mesh_parsers is None:
                logger.error("No mesh parser found")
            # trans, trans_fix = fix_trans_height(self, pose_aa, trans, mesh_parsers, fix_height_mode = fix_height)
            curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
            curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })
            # add "action" to curr_motion
            if self.has_action:
                curr_motion.action = to_torch(curr_file['action']).clone()[start:end]
            if self.has_contact_mask:
                curr_motion.contact_mask = to_torch(curr_file['contact_mask']).clone()[start:end]
                
            res[f] = (curr_file, curr_motion)
        return res
    



    def get_motion_state(self, motion_ids, motion_times, offset=None, target2targetaligned=None):
        # motion_ids: (num_envs,)
        # motion_times: (num_envs,) ,以秒为单位
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = _calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        if "dof_pos" in self.__dict__:
            local_rot0 = self.dof_pos[f0l]
            local_rot1 = self.dof_pos[f1l]
        else:
            local_rot0 = self.lrs[f0l]
            local_rot1 = self.lrs[f1l]
            
        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        # breakpoint()
        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        # if offset is None:
        #     rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        # else:
        #     rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset
        rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset


        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1

        if "dof_pos" in self.__dict__: # Robot Joints
            dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
            dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1
        else:
            dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
            local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = _local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}
        
        if "gts_t" in self.__dict__:
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            
            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]
            
            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]
            
            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            # if offset is None:
            #     rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  
            # else:
            #     rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  

            rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            rg_pos_t = rg_pos
            rg_rot_t = rb_rot
            body_vel_t = body_vel
            body_ang_vel_t = body_ang_vel
        
        if flags.real_traj:
            import ipdb; ipdb.set_trace()
            q_body_ang_vel0, q_body_ang_vel1 = self.q_gavs[f0l], self.q_gavs[f1l]
            q_rb_rot0, q_rb_rot1 = self.q_grs[f0l], self.q_grs[f1l]
            q_rg_pos0, q_rg_pos1 = self.q_gts[f0l, :], self.q_gts[f1l, :]
            q_body_vel0, q_body_vel1 = self.q_gvs[f0l], self.q_gvs[f1l]

            q_ang_vel = (1.0 - blend_exp) * q_body_ang_vel0 + blend_exp * q_body_ang_vel1
            q_rb_rot = slerp(q_rb_rot0, q_rb_rot1, blend_exp)
            q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1
            q_body_vel = (1.0 - blend_exp) * q_body_vel0 + blend_exp * q_body_vel1
            
            rg_pos[:, self.track_idx] = q_rg_pos
            rb_rot[:, self.track_idx] = q_rb_rot
            body_vel[:, self.track_idx] = q_body_vel
            body_ang_vel[:, self.track_idx] = q_ang_vel


        if self.has_contact_mask:
            contact0, contact1 = self._motion_contact_masks[f0l], self._motion_contact_masks[f1l]
            contact = (1.0 - blend) * contact0 + blend * contact1
            
            return_dict["contact_mask"] = contact
        
        if target2targetaligned is None:

            if offset is not None:
                rg_pos_t = rg_pos_t + offset[..., None, :]
                rg_pos = rg_pos + offset[..., None, :]
            # xyzw
            return_dict.update({
                "root_pos": rg_pos[..., 0, :].clone(),  # 世界坐标系下根节点位置
                "root_rot": rb_rot[..., 0, :].clone(),  # 世界坐标系下根节点旋转
                "dof_pos": dof_pos.clone(),             # 关节角
                "root_vel": body_vel[..., 0, :].clone(),    # 世界坐标系下根节点线速度
                "root_ang_vel": body_ang_vel[..., 0, :].clone(),    # 世界坐标系下根节点角速度
                "dof_vel": dof_vel.view(dof_vel.shape[0], -1),  # 关节速度
                # "motion_aa": self._motion_aa[f0l],      # smpl相关，与此处机器人的ref motion tracking无关
                # "motion_bodies": self._motion_bodies[motion_ids],       # 同样无关
                "rg_pos": rg_pos,       # 世界坐标系下body位置
                "rb_rot": rb_rot,       # 世界坐标系下的body旋转
                "body_vel": body_vel,   # 世界坐标系下body线速度
                "body_ang_vel": body_ang_vel,   # 世界坐标系下body角速度
                "rg_pos_t": rg_pos_t,  
                "rg_rot_t": rg_rot_t,
                "body_vel_t": body_vel_t,
                "body_ang_vel_t": body_ang_vel_t,
                # "label": self._motion_data_label[motion_ids]
            })
        else:

            def quat_mul(a, b):
                # Hamilton product for quaternions in xyzw order
                # a,b: (...,4) -> out (...,4)
                ax, ay, az, aw = a.unbind(dim=-1)
                bx, by, bz, bw = b.unbind(dim=-1)
                # Hamilton product
                rx = aw * bx + ax * bw + ay * bz - az * by
                ry = aw * by - ax * bz + ay * bw + az * bx
                rz = aw * bz + ax * by - ay * bx + az * bw
                rw = aw * bw - ax * bx - ay * by - az * bz
                return torch.stack([rx, ry, rz, rw], dim=-1)

            def rotate_vec_by_quat(q, v):
                # rotate v by quaternion q (both can be broadcastable)
                # q: (...,4) xyzw, v: (...,3)
                # formula: v' = v + 2*cross(q_xyz, cross(q_xyz, v) + q_w * v)
                q_xyz = q[..., :3]
                q_w = q[..., 3:4]
                # ensure broadcasting shapes align
                # cross expects same trailing dim
                t = 2.0 * torch.cross(q_xyz, v, dim=-1)
                return v + q_w * t + torch.cross(q_xyz, t, dim=-1)
            # target2targetaligned: tensor 或 array，最后一维 7 (tx,ty,tz, qx,qy,qz,qw)，xyzw 顺序
            tt = target2targetaligned
            if not torch.is_tensor(tt):
                tt = torch.as_tensor(tt, dtype=rg_pos_t.dtype, device=rg_pos_t.device)
            else:
                tt = tt.to(dtype=rg_pos_t.dtype, device=rg_pos_t.device)

            batch = rg_pos_t.shape[0]
            if tt.dim() == 1 and tt.numel() == 7:
                tt = tt.unsqueeze(0).expand(batch, 7)
            elif tt.dim() == 2 and tt.shape[0] == 1 and batch > 1:
                tt = tt.expand(batch, 7)
            elif tt.dim() == 2 and tt.shape[0] != batch:
                raise ValueError(f"target2targetaligned batch size ({tt.shape[0]}) doesn't match data batch ({batch})")

            t_rel = tt[..., :3]   # (B,3) 平移
            q_rel = tt[..., 3:]   # (B,4) 旋转 xyzw
            q_rel = q_rel / (q_rel.norm(dim=-1, keepdim=True) + 1e-12)

            # bodies 数量分别
            nb_rb = rg_pos.shape[1]   # motion bodies count
            nb_t = rg_pos_t.shape[1]  # target bodies count

            # 为广播准备 q_rel 扩展（分别用于 motion bodies / target bodies）
            q_rel_b_rb = q_rel.unsqueeze(1).expand(-1, nb_rb, -1)  # (B, nb_rb, 4)
            q_rel_b_t = q_rel.unsqueeze(1).expand(-1, nb_t, -1)    # (B, nb_t, 4)

            # ----------------- 按 align2source 的语义做变换 -----------------
            # 位置： p_aligned = R_rel * p + t_rel
            rg_pos_aligned = rotate_vec_by_quat(q_rel_b_rb, rg_pos) + t_rel.unsqueeze(1)     # (B, nb_rb, 3)
            rg_pos_t_aligned = rotate_vec_by_quat(q_rel_b_t, rg_pos_t) + t_rel.unsqueeze(1)  # (B, nb_t, 3)

            # 旋转： q_aligned = q_rel * q_old
            q_body_aligned_rb = quat_mul(q_rel_b_rb, rb_rot)     # (B, nb_rb, 4)
            q_body_aligned_t = quat_mul(q_rel_b_t, rg_rot_t)     # (B, nb_t, 4)

            # 速度： v_aligned = R_rel * v
            body_vel_aligned = rotate_vec_by_quat(q_rel_b_rb, body_vel)          # (B, nb_rb, 3)
            body_ang_vel_aligned = rotate_vec_by_quat(q_rel_b_rb, body_ang_vel)  # (B, nb_rb, 3)

            body_vel_t_aligned = rotate_vec_by_quat(q_rel_b_t, body_vel_t)           # (B, nb_t, 3)
            body_ang_vel_t_aligned = rotate_vec_by_quat(q_rel_b_t, body_ang_vel_t)   # (B, nb_t, 3)

            if offset is not None:
                rg_pos_aligned = rg_pos_aligned + offset[..., None, :]
                rg_pos_t_aligned = rg_pos_t_aligned + offset[..., None, :]

            # dof 保持不变
            return_dict.update({
                "root_pos": rg_pos_aligned[..., 0, :].clone(),   # 世界坐标系下根节点位置 (aligned)
                "root_rot": q_body_aligned_rb[..., 0, :].clone(),   # 世界坐标系下根节点旋转 (aligned)
                "dof_pos": dof_pos.clone(),     # 关节角 (不变)
                "root_vel": body_vel_aligned[..., 0, :].clone(),   # 世界坐标系下根节点线速度 (aligned)
                "root_ang_vel": body_ang_vel_aligned[..., 0, :].clone(), # 世界坐标系下根节点角速度 (aligned)
                "dof_vel": dof_vel.view(dof_vel.shape[0], -1),  # 关节速度 (不变)
                "rg_pos": rg_pos_aligned,                # 所有 motion body 的位置 (aligned)
                "rb_rot": q_body_aligned_rb,             # 所有 motion body 的旋转 (aligned)
                "body_vel": body_vel_aligned,            # 所有 motion body 线速度 (aligned)
                "body_ang_vel": body_ang_vel_aligned,    # 所有 motion body 角速度 (aligned)
                "rg_pos_t": rg_pos_t_aligned,            # 额外 target body 位置 (aligned)
                "rg_rot_t": q_body_aligned_t,            # 额外 target body 旋转 (aligned)
                "body_vel_t": body_vel_t_aligned,        # 额外 target body 线速度 (aligned)
                "body_ang_vel_t": body_ang_vel_t_aligned,# 额外 target body 角速度 (aligned)
            })
        return return_dict
    


    def get_motion_length(self, motion_ids=None):
        # motion length的单位是秒
        if motion_ids is None:
            # print("motion lengths:", self._motion_lengths)
            return self._motion_lengths
        else:
            # print("motion lengths:", self._motion_lengths[motion_ids])
            return self._motion_lengths[motion_ids]


    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)



    
    
class MotionLibRobotWTS(MotionLibBase):
    def __init__(self, motion_lib_cfg, num_envs, device):
        super().__init__(motion_lib_cfg = motion_lib_cfg, num_envs = num_envs, device = device)
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg)

        # np.save('custom_data/dof_axis.npy',self.mesh_parsers.dof_axis)
        return