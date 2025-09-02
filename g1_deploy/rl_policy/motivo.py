import sys
import time
import threading
import numpy as np
import zmq
from typing import List, Dict, Any, Type
import sched
from termcolor import colored
from sshkeyboard import listen_keyboard
from pathlib import Path
import math
import torch.nn.functional as F
import joblib
import json


sys.path.append(".")
# Ensure g1_interface can be imported
sys.path.append("/home/unitree/haoyang_deploy/unitree_sdk2/build/lib")
# try:
#     import g1_interface  # type: ignore
# except ImportError as e:
#     raise ImportError("g1_interface module not found. Make sure it is built and added to PYTHONPATH.")

from loguru import logger

# Re-use utilities from the existing sim2real package
from utils.strings import resolve_matching_names_values, unitree_joint_names
from utils.onnx_module import Timer
from rl_policy.observations import Observation, ObsGroup
from rl_policy.utils.state_processor import StateProcessor
from rl_policy.utils.command_sender import CommandSender
# -------------------------------------------------------------------------------------------------
# High-level RL policy that plugs into the existing framework
# -------------------------------------------------------------------------------------------------
class MotivoPolicy:
    def __init__(
        self,
        robot_config: Dict[str, Any],
        policy_config: Dict[str, Any],
        exp_config: Dict[str, Any],
        model_path: str,
        rl_rate: int = 50,
    ) -> None:
        robot_type = robot_config["ROBOT_TYPE"]
        if robot_type != "g1_real":
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            if robot_config.get("INTERFACE", None):
                ChannelFactoryInitialize(robot_config["DOMAIN_ID"], robot_config["INTERFACE"])
            else:
                ChannelFactoryInitialize(robot_config["DOMAIN_ID"])
        else:
            sys.path.append("/home/elijah/haoyang_deploy/unitree_sdk2/build/lib")
            import g1_interface
        
            network_interface = robot_config.get("INTERFACE", None)
            # Create shared G1 interface instance
            self.robot = g1_interface.G1Interface(network_interface)
            # Ensure we are in PR mode (most policies work in PR)
            try:
                self.robot.set_control_mode(g1_interface.ControlMode.PR)
            except Exception:
                pass  # Ignore if firmware already in the correct mode
            robot_config["robot"] = self.robot
        # Plug-in our custom state processor & command sender
        self.state_processor = StateProcessor(robot_config, policy_config["isaac_joint_names"])
        self.command_sender = CommandSender(robot_config, policy_config)

        self.rl_dt = 1.0 / rl_rate
        self.t = 0

        self.policy_config = policy_config

        self.setup_policy(model_path)
        self.obs_cfg = policy_config["observation"]

        self.isaac_joint_names = policy_config["isaac_joint_names"]
        self.num_dofs = len(self.isaac_joint_names)

        default_joint_pos_dict = policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            self.isaac_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_dof_angles = np.zeros(len(self.isaac_joint_names))
        self.last_action  = np.zeros(len(self.isaac_joint_names))
        self.default_dof_angles[joint_indices] = default_joint_pos

        action_scale_cfg = policy_config["action_scale"]
        self.action_scale = np.ones((self.num_dofs))
        if isinstance(action_scale_cfg, float):
            self.action_scale *= action_scale_cfg
        elif isinstance(action_scale_cfg, dict):
            joint_ids, joint_names, action_scales = resolve_matching_names_values(
                action_scale_cfg, self.isaac_joint_names, preserve_order=True
            )
            self.action_scale[joint_ids] = action_scales
        else:
            raise ValueError(f"Invalid action scale type: {type(action_scale_cfg)}")

        self.policy_joint_names = policy_config["policy_joint_names"]
        self.num_actions = len(self.policy_joint_names)
        self.controlled_joint_indices = [
            self.isaac_joint_names.index(name)
            for name in self.policy_joint_names
        ]

        # Keypress control state
        self.use_policy_action = False

        self.first_time_init = True
        self.init_count = 0
        self.get_ready_state = False

        # Joint limits
        joint_indices, joint_names, joint_pos_lower_limit = (
            resolve_matching_names_values(
                robot_config["joint_pos_lower_limit"],
                self.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_lower_limit = np.zeros(self.num_dofs)
        self.joint_pos_lower_limit[joint_indices] = joint_pos_lower_limit

        joint_indices, joint_names, joint_pos_upper_limit = (
            resolve_matching_names_values(
                robot_config["joint_pos_upper_limit"],
                self.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_upper_limit = np.zeros(self.num_dofs)
        self.joint_pos_upper_limit[joint_indices] = joint_pos_upper_limit

        # ------------------------------------------------------
        # Joystick / keyboard setup (mirrors base_policy logic)
        # ------------------------------------------------------
        if robot_config.get("USE_JOYSTICK", False):
            print("Using joystick")
            self.use_joystick = True
            self.wc_msg = None  # type: ignore
            self.last_wc_msg = self.robot.read_wireless_controller()
            print("Wireless Controller Initialized")
        else:
            import threading
            print("Using keyboard")
            self.use_joystick = False
            self.key_listener_thread = threading.Thread(
                target=self.start_key_listener, daemon=True
            )
            self.key_listener_thread.start()

        # Setup observations after all processors are ready
        self.setup_observations()

        # for test
        self.exp_config = exp_config
        self.task_type = exp_config['type']
        self.start_motion = False

        if self.task_type == "tracking":
            ctx_path = Path(model_path).parent / exp_config['ctx_path']
            self.ctx = joblib.load(ctx_path)
            self.t_start = exp_config['start']
            self.t_end = exp_config['end']
            self.t_stop = exp_config['stop']
            
        elif self.task_type == "reward":
            self.z_index = 0
            import pickle
            with open(Path(model_path).parent.parent / "reward_inference" / exp_config['ctx_path'], "rb") as f:
                self.z_dict = pickle.load(f)
            if "selected_rewards" in exp_config:
               selected_rewards = exp_config['selected_rewards'] 
            else:
                selected_rewards = self.z_dict.keys()
                
            self.z_dict = {
                k: v for k, v in self.z_dict.items() if k in selected_rewards
            }
            self.num_rewards = len(selected_rewards)
        elif self.task_type == "stiching":
            ctx_path = Path(model_path).parent / exp_config['ctx_path']
            pose_ctx_path = Path(model_path).parent / exp_config['pose_ctx_path']
            self.ctx = joblib.load(ctx_path)
            self.pose_ctx = joblib.load(pose_ctx_path)
            self.t_start = exp_config['start']
            self.t_end = exp_config['end']
            self.t_stop = exp_config['stop']
            self.t_interval = exp_config['t_interval']
            self.pose_time = exp_config['pose_time']
            self.pose_count_time = 0
            self.pose_count = 0
            self.start_pose = False
            self.pose_list = exp_config['pose_list']

            self.rl_log_buffer = []

        import os
        self.rl_log_buffer = []
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.rl_log_file = os.path.join(
            log_dir, f"rl_step_log_sim2sim_walk2sj4.jsonl"
        )

            


    def setup_policy(self, model_path):
        # load onnx policy
        import onnxruntime
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name

        def policy_act(obs):
            return self.onnx_policy_session.run([self.onnx_output_name], {self.onnx_input_name: obs})[0]
        self.policy = policy_act

    def setup_observations(self):
        """Setup observations for policy inference"""
        self.observations: Dict[str, ObsGroup] = {}
        self.reset_callbacks = []
        self.update_callbacks = []
        
        # Create observation instances based on config
        for obs_group, obs_items in self.obs_cfg.items():
            print(f"obs_group: {obs_group}")
            obs_funcs = {}
            for obs_name, obs_config in obs_items.items():
                obs_class: Type[Observation] = Observation.registry[obs_name]
                obs_func = obs_class(env=self, **obs_config)
                obs_funcs[obs_name] = obs_func
                self.reset_callbacks.append(obs_func.reset)
                self.update_callbacks.append(obs_func.update)
                print(f"\t{obs_name}: {obs_config}")
            self.observations[obs_group] = ObsGroup(obs_group, obs_funcs)

    def reset(self):
        for reset_callback in self.reset_callbacks:
            reset_callback()

    def update(self):
        for update_callback in self.update_callbacks:
            update_callback(self.state_dict)

    def prepare_obs_for_rl(self):
        """Prepare observation for policy inference using observation classes"""
        obs_dict: Dict[str, np.ndarray] = {}
        for obs_group in self.observations.values():
            obs = obs_group.compute()
            obs_dict[obs_group.name] = obs[None, :].astype(np.float32)
        
        obs = obs_dict[obs_group.name]
        obs[:, 64:93] = self.last_action
        self.state_dict["action"] = self.last_action
        self.state_dict["dof_pos_minus_default"] = obs[:, 0:29]
        self.state_dict["dof_vel"] = obs[:, 29:58]
        self.state_dict["projected_gravity"] = obs[:, 58:61]
        self.state_dict["root_ang_vel_b"] = obs[:, 61:64]
        self.update()

        if self.task_type == "tracking":
            gamma = 0.8  # 折扣因子
            window = self.ctx[self.t:self.t+1]  # 取出窗口

            discounts = gamma ** np.arange(len(window))  # 生成折扣权重：[1, gamma, gamma^2, ...]
            discounts = discounts / np.sum(discounts)    # 归一化成平均权重

            # 加权平均                                                                                                                                                                         
            discounted_avg = np.sum(window * discounts[:, np.newaxis], axis=0)
            discounted_avg = discounted_avg / np.linalg.norm(discounted_avg, axis=-1) * 16
            
            inputs = np.concatenate([obs, discounted_avg[np.newaxis, :]], axis= -1).astype(np.float32)
            # inputs = np.concatenate([obs, list(self.z_dict.values())[self.z_index].cpu()], axis=-1).astype(np.float32)
            if self.use_policy_action:
                if self.start_motion and self.t < self.t_end:
                    self.t += 1
                    self.t = self.t % self.ctx.shape[0]
                    print(self.t)
                else:
                    self.t = self.t_stop
                    self.start_motion = False
            
        elif self.task_type == "reward":
            inputs = np.concatenate([obs, list(self.z_dict.values())[self.z_index].cpu()], axis=-1).astype(np.float32)
        elif self.task_type == "single":
            inputs = np.concatenate([obs, self.z[np.newaxis, :]], axis=-1).astype(np.float32)
        elif self.task_type == "stiching":
            inputs = np.concatenate([obs, self.ctx[self.t:self.t+1]], axis=-1).astype(np.float32)
            if self.use_policy_action:
                if self.start_pose == True:
                    self.pose_count_time += 1
                    print("pose", self.pose_count)
                    inputs = np.concatenate([obs, self.pose_z], axis=-1).astype(np.float32)
                    if self.pose_count_time % self.pose_time == 0:
                        self.start_pose = False
                        self.start_motion = True
                        self.pose_count += 1
                    
                if self.start_motion and self.t < self.t_end:
                    if self.t % self.t_interval == 0 and self.t != 0:
                        self.start_pose = True
                        self.start_motion = False
                        print(f"start pose {self.pose_count}")
                        self.pose_z = self.pose_ctx[self.pose_list[self.pose_count]:self.pose_list[self.pose_count]+1]
                        inputs = np.concatenate([obs, self.pose_z], axis=-1).astype(np.float32)
                        self.t += 1
                        
                    else:
                        self.t += 1
                        self.t = self.t % self.ctx.shape[0]
                        inputs = np.concatenate([obs, self.ctx[self.t:self.t+1]], axis=-1).astype(np.float32)
                        print(self.t)
                elif self.start_pose == False:
                    self.t = self.t_stop
                    self.start_motion = False
                    inputs = np.concatenate([obs, self.ctx[self.t:self.t+1]], axis=-1).astype(np.float32)
        return obs_dict, inputs

    def get_init_target(self):
        if self.init_count > 500:
            self.init_count = 500

        # interpolate from current dof_pos to default angles
        dof_pos = self.state_processor.joint_pos
        progress = self.init_count / 500
        q_target = dof_pos + (self.default_dof_angles - dof_pos) * progress
        self.init_count += 1
        return q_target

    def run(self):
        total_inference_cnt = 0
        
        # 初始化状态变量
        state_dict = {}
        state_dict["adapt_hx"] = np.zeros((1, 128), dtype=np.float32)
        state_dict["action"] = np.zeros(self.num_actions)
        self.state_dict = state_dict
        self.total_inference_cnt = total_inference_cnt
        self.perf_dict = {}

        try:
            # 使用scheduler进行精确时间控制
            scheduler = sched.scheduler(time.perf_counter, time.sleep)
            next_run_time = time.perf_counter()
            
            while True:
                # 调度下一次执行
                scheduler.enterabs(next_run_time, 1, self._rl_step_scheduled, ())
                scheduler.run()
                
                next_run_time += self.rl_dt
                self.total_inference_cnt += 1

                if self.total_inference_cnt % 100 == 0:
                    # print(f"total_inference_cnt: {self.total_inference_cnt}")
                    # for key, value in self.perf_dict.items():
                    #     print(f"\t{key}: {value/100*1000:.3f} ms")
                    self.perf_dict = {}
        except KeyboardInterrupt:
            pass

    def _rl_step_scheduled(self):
        loop_start = time.perf_counter()

        with Timer(self.perf_dict, "prepare_low_state"):
            if self.use_joystick:
                self.process_joystick_input()

            if not self.state_processor._prepare_low_state():
                print("low state not ready.")
                return
            
        try:
            with Timer(self.perf_dict, "prepare_obs"):
                # Prepare observations
                obs_dict, observations = self.prepare_obs_for_rl()
                self.state_dict["is_init"] = np.zeros(1, dtype=bool)

            with Timer(self.perf_dict, "policy"):  
                # Inference
                action = self.policy(observations)
                # Clip policy action
                action = action.clip(-1, 1)
                action_scaled = 5 * action
                self.last_action = action_scaled
        except Exception as e:
            print(f"Error in policy inference: {e}")
            self.state_dict["action"] = np.zeros(self.num_actions)
            return

        with Timer(self.perf_dict, "rule_based_control_flow"):
            # rule based control flow
            if self.get_ready_state:
                q_target = self.get_init_target()
            elif not self.use_policy_action:
                q_target = self.state_processor.joint_pos
            else:
                policy_action = np.zeros((self.num_dofs))
                policy_action[self.controlled_joint_indices] = action_scaled
                policy_action = policy_action * self.action_scale
                self.rl_log_buffer.append({
                    "timestamp": time.time(),
                    "obs": observations.tolist(),         # numpy → list
                    "action_raw": action.tolist(),
                    "action_scaled": policy_action.tolist(),
                })
                # policy_action *= 0
                # policy_action[0] += 1
                q_target = policy_action + self.default_dof_angles

            # Clip q target
            # print(self.joint_pos_lower_limit)
            q_target = np.clip(
                q_target, self.joint_pos_lower_limit, self.joint_pos_upper_limit
            )

            # Send command
            cmd_q = q_target
            cmd_dq = np.zeros(self.num_dofs)
            cmd_tau = np.zeros(self.num_dofs)
            self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)

        elapsed = time.perf_counter() - loop_start
        if elapsed > self.rl_dt:
            logger.warning(f"RL step took {elapsed:.6f} seconds, expected {self.rl_dt} seconds")


        if len(self.rl_log_buffer) >= 100:
            print("store")
            with open(self.rl_log_file, "a") as f:
                for entry in self.rl_log_buffer:
                    f.write(json.dumps(entry) + "\n")
            self.rl_log_buffer.clear()

    def process_joystick_input(self):
        """Poll current wireless controller state and translate to high-level key events."""
        try:
            self.wc_msg = self.robot.read_wireless_controller()
        except Exception:
            return

        if self.wc_msg is None:
            return

        # print(f"wc_msg.A: {self.wc_msg.A}")
        if self.wc_msg.A and not self.last_wc_msg.A:
            self.handle_joystick_button("A")
        if self.wc_msg.B and not self.last_wc_msg.B:
            self.handle_joystick_button("B")
        if self.wc_msg.X and not self.last_wc_msg.X:
            self.handle_joystick_button("X")
        if self.wc_msg.Y and not self.last_wc_msg.Y:
            self.handle_joystick_button("Y")
        if self.wc_msg.L1 and not self.last_wc_msg.L1:
            self.handle_joystick_button("L1")
        if self.wc_msg.L2 and not self.last_wc_msg.L2:
            self.handle_joystick_button("L2")
        if self.wc_msg.R1 and not self.last_wc_msg.R1:
            self.handle_joystick_button("R1")
        if self.wc_msg.R2 and not self.last_wc_msg.R2:
            self.handle_joystick_button("R2")
        
        self.last_wc_msg = self.wc_msg

    def handle_joystick_button(self, cur_key):
        if cur_key == "R1":
            self.use_policy_action = True
            self.get_ready_state = False
            logger.info(colored("Using policy actions", "blue"))
            self.phase = 0.0  # type: ignore
        elif cur_key == "R2":
            self.use_policy_action = False
            self.get_ready_state = False
            logger.info(colored("Actions set to zero", "blue"))
        elif cur_key == "A":
            self.get_ready_state = True
            self.init_count = 0
            logger.info(colored("Setting to init state", "blue"))
        elif cur_key == "B":
            self.start_motion = True
            self.t = self.t_start
        # elif cur_key == "Y+left":
        #     self.command_sender.kp_level -= 0.1
        # elif cur_key == "Y+right":
        #     self.command_sender.kp_level += 0.1
        # elif cur_key == "A+left":
        #     self.command_sender.kp_level -= 0.01
        # elif cur_key == "A+right":
        #     self.command_sender.kp_level += 0.01

        # Debug print for kp level tuning
        if cur_key in ["Y+left", "Y+right", "A+left", "A+right"]:
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))

    # ----------------------------- Keyboard handling -----------------------------
    def start_key_listener(self):
        """Start a key listener using sshkeyboard (same as BasePolicy)."""

        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except AttributeError as e:
                logger.warning(f"Keyboard key {keycode}. Error: {e}")

        listener = listen_keyboard(on_press=on_press)
        listener.start()
        listener.join()

    def handle_keyboard_button(self, keycode):
        if keycode == "]":
            self.use_policy_action = True
            self.t = self.t_stop
            self.get_ready_state = False
            logger.info("Using policy actions")
            # self.frame_start_time = time.perf_counter()
            self.phase = 0.0
        elif keycode == "[":
            self.start_motion = True
            self.t = self.t_start
        elif keycode == "n":
            if self.task_type == "reward":
                if self.z_index >= self.num_rewards - 1:
                    self.z_index = 0
                else:
                    self.z_index += 1
            elif self.task_type == "single":
                self.z_index += 1
                self.z = np.array(self.exp_config[f"Best z"])
                print({self.z_index})
        elif keycode == "p":
            self.z_index = 0
            self.start_motion = False
            self.t = self.t_stop
            logger.info("Resetting to stop state")
        elif keycode == "o":
            self.use_policy_action = False
            self.get_ready_state = False
            logger.info("Actions set to zero")
        elif keycode == "i":
            self.get_ready_state = True
            self.init_count = 0
            logger.info("Setting to init state")
        elif keycode == "w":
            self.lin_vel_command[0, 0] += 0.1
        elif keycode == "s":
            self.lin_vel_command[0, 0] -= 0.1
        elif keycode == "a":
            self.lin_vel_command[0, 1] += 0.1
        elif keycode == "d":
            self.lin_vel_command[0, 1] -= 0.1
        elif keycode == "q":
            self.ang_vel_command[0, 0] -= 0.1
        elif keycode == "e":
            self.ang_vel_command[0, 0] += 0.1
        elif keycode == "z":
            self.ang_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 1] = 0.0
        elif keycode == "5":
            self.command_sender.kp_level -= 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "6":
            self.command_sender.kp_level += 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "4":
            self.command_sender.kp_level -= 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "7":
            self.command_sender.kp_level += 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "0":
            self.command_sender.kp_level = 1.0
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        if self.task_type == "reward":
            print({self.z_index})
        # import ipdb; ipdb.set_trace()
            print(f"Testing reward {list(self.z_dict.keys())[self.z_index]}")


if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument(
        "--policy_config", type=str, help="policy config file"
    )
    parser.add_argument(
        "--model_path", type=str, help="model path"
    )
    args = parser.parse_args()

    with open(args.policy_config) as file:
        policy_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    with open("config/exp/walk1_sj1-1.yaml") as file:
        exp_config = yaml.load(file, Loader=yaml.FullLoader)
    model_path = args.model_path

    policy = MotivoPolicy(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=model_path,
        exp_config=exp_config,
        rl_rate=50,
    )
    policy.run()
