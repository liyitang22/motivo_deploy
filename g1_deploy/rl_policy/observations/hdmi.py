from .base import Observation

import time
from typing import Any, Dict, List
import numpy as np
from utils.math import quat_rotate_inverse_numpy, yaw_from_quat, wrap_to_pi, yaw_quat, quat_multiply_numpy, quat_conjugate_numpy, matrix_from_quat
from utils.common import PORTS


class ref_motion_phase(Observation):
    def __init__(self, motion_duration_second: float, **kwargs):
        super().__init__(**kwargs)
        self.ref_motion_phase = np.zeros(1)
        self.start_time = time.time()
        self.motion_duration_second = motion_duration_second
    
    def reset(self):
        """Reset the motion phase to start from 0"""
        self.start_time = time.time()
        self.ref_motion_phase[:] = 0

    def compute(self) -> np.ndarray:
        t = time.time()
        self.ref_motion_phase[:] = (t - self.start_time) / self.motion_duration_second
        self.ref_motion_phase %= 1.0
        print(f"ref_motion_phase: {self.ref_motion_phase}")
        return self.ref_motion_phase

class ref_motion_phase_noise(Observation):
    def compute(self) -> np.ndarray:
        return np.random.normal(0, 1, 1)

from rl_policy.utils.motion import MotionDataset, MotionData

class _motion_obs(Observation):
    def __init__(self, motion_path: str, future_steps: List[int], joint_names: List[str], body_names: List[str], root_body_name: str = "pelvis", **kwargs):
        super().__init__(**kwargs)
        self.motion_dataset = MotionDataset.create_from_path(motion_path)
        assert self.motion_dataset.num_motions == 1, "Only one motion is supported"
        self.motion_ids = np.array([0])
        self.motion_length = self.motion_dataset.num_steps

        self.t = np.array([0])
        self.future_steps = np.array(future_steps)

        self.joint_indices = [self.motion_dataset.joint_names.index(name) for name in joint_names]
        self.body_indices = [self.motion_dataset.body_names.index(name) for name in body_names]
        self.root_body_idx = self.motion_dataset.body_names.index(root_body_name)

        self.n_future_steps = len(self.future_steps)
        self.n_bodies = len(self.body_indices)
    
    def reset(self):
        self.t[:] = 0
    
    def update(self, data: Dict[str, Any]) -> None:
        self.t += 1
        if self.t[0] == self.motion_length:
            self.t[:] = 0
        motion_data: MotionData = self.motion_dataset.get_slice(self.motion_ids, self.t, self.future_steps)
        self.ref_joint_pos_future = motion_data.joint_pos[:, :, self.joint_indices]
        self.ref_joint_vel_future = motion_data.joint_vel[:, :, self.joint_indices]
        self.ref_body_pos_future_w = motion_data.body_pos_w[:, :, self.body_indices]
        self.ref_body_lin_vel_future_w = motion_data.body_lin_vel_w[:, :, self.body_indices]
        self.ref_body_quat_future_w = motion_data.body_quat_w[:, :, self.body_indices]
        self.ref_body_ang_vel_future_w = motion_data.body_ang_vel_w[:, :, self.body_indices]
        self.ref_root_pos_w = motion_data.body_pos_w[:, [0], [self.root_body_idx], :]
        self.ref_root_quat_w = motion_data.body_quat_w[:, [0], [self.root_body_idx], :]


class ref_joint_pos_future(_motion_obs):
    def compute(self) -> np.ndarray:
        print(f"t: {self.t.item()}")
        return self.ref_joint_pos_future.reshape(-1)
    
class ref_joint_vel_future(_motion_obs):
    def compute(self) -> np.ndarray:
        return self.ref_joint_vel_future.reshape(-1)
    
class ref_body_pos_future_local(_motion_obs):
    """
    Reference body position in motion root frame
    """
    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_body_pos_future_w = self.ref_body_pos_future_w
        ref_root_pos_w: np.ndarray = self.ref_root_pos_w # [batch, 1, 1, 3]
        ref_root_quat_w: np.ndarray = self.ref_root_quat_w  # [batch, 1, 1, 4]

        # Expand dimensions to match ref_body_pos_future_w
        ref_root_pos_w = np.tile(ref_root_pos_w, (1, self.n_future_steps, self.n_bodies, 1))  # [batch, future_steps, n_bodies, 3]
        ref_root_quat_w = np.tile(ref_root_quat_w, (1, self.n_future_steps, self.n_bodies, 1))  # [batch, future_steps, n_bodies, 4]

        ref_root_pos_w[..., 2] = 0.0
        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        ref_body_pos_future_local = quat_rotate_inverse_numpy(ref_root_quat_w, ref_body_pos_future_w - ref_root_pos_w)
        self.ref_body_pos_future_local = ref_body_pos_future_local
    
    def compute(self):
        return self.ref_body_pos_future_local.reshape(-1)
    
class ref_body_ori_future_local(_motion_obs):
    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_body_quat_future_w = self.ref_body_quat_future_w
        ref_root_quat_w = self.ref_root_quat_w

        ref_root_quat_w = np.tile(ref_root_quat_w, (1, self.n_future_steps, self.n_bodies, 1))
        
        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        ref_body_quat_future_local = quat_multiply_numpy(
            quat_conjugate_numpy(ref_root_quat_w),
            ref_body_quat_future_w
        )
        self.ref_body_ori_future_local = matrix_from_quat(ref_body_quat_future_local)
    
    def compute(self):
        return self.ref_body_ori_future_local[:, :, :, :2, :3].reshape(-1)

class ref_body_lin_vel_future_local(_motion_obs):
    def __init__(self, root_body_name: str = "pelvis", **kwargs):
        super().__init__(**kwargs)
        self.root_body_idx = self.motion_dataset.body_names.index(root_body_name)
    
    def update(self):
        super().update()
        ref_body_lin_vel_future_w = self.motion_data.body_lin_vel_w[:, :, :, :]
        ref_root_quat_w = self.motion_data.body_quat_w[:, :, self.root_body_idx, :]

        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        ref_body_lin_vel_future_local = quat_rotate_inverse_numpy(ref_root_quat_w, ref_body_lin_vel_future_w)
        self.ref_body_lin_vel_future_local = ref_body_lin_vel_future_local
    
    def compute(self):
        return self.ref_body_lin_vel_future_local.reshape(-1)


class door_pos_b(Observation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Register required subscribers
        self.state_processor.register_subscriber("Wall", PORTS["Wall"])
        self.state_processor.register_subscriber("pelvis", PORTS["pelvis"])
        
        # Give time for connections to establish
        time.sleep(0.5)

    def compute(self) -> np.ndarray:
        # Get mocap data from state processor
        door_pos_w = self.state_processor.get_mocap_data("Wall_pos")
        root_pos_w = self.state_processor.get_mocap_data("pelvis_pos")
        root_quat_w = self.state_processor.get_mocap_data("pelvis_quat")
        
        if door_pos_w is None or root_pos_w is None or root_quat_w is None:
            raise ValueError("Missing mocap data for door_pos_b computation")
            
        root_quat_yaw_w = yaw_quat(root_quat_w)
        door_pos_b = quat_rotate_inverse_numpy(
            root_quat_yaw_w[None, :], (door_pos_w - root_pos_w)[None, :]
        ).squeeze(0)
        print(f"door_pos_b: {door_pos_b}")
        return door_pos_b[:2]


class root_yaw(Observation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Register required subscribers
        self.state_processor.register_subscriber("Wall", PORTS["Wall"])
        self.state_processor.register_subscriber("pelvis", PORTS["pelvis"])
        
        # Give time for connections to establish
        time.sleep(0.5)

    def compute(self) -> np.ndarray:
        # Get mocap data from state processor
        door_quat_w = self.state_processor.get_mocap_data("Wall_quat")
        root_quat_w = self.state_processor.get_mocap_data("pelvis_quat")
        
        if door_quat_w is None or root_quat_w is None:
            raise ValueError("Missing mocap data for root_yaw computation")
            
        root_yaw_w = yaw_from_quat(root_quat_w[None, :]).squeeze(0)
        door_yaw_w = yaw_from_quat(door_quat_w[None, :]).squeeze(0)
        root_yaw = wrap_to_pi(root_yaw_w - door_yaw_w - np.pi)
        print(f"root_yaw: {root_yaw}")
        return np.array(root_yaw)

