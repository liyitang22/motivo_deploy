import sys
import time
import threading
import numpy as np
import zmq
from typing import List, Dict, Any, Type
import sched
from termcolor import colored
from sshkeyboard import listen_keyboard


sys.path.append(".")
# Ensure g1_interface can be imported
sys.path.append("/home/unitree/haoyang_deploy/unitree_sdk2/build/lib")
try:
    import g1_interface  # type: ignore
except ImportError as e:
    raise ImportError("g1_interface module not found. Make sure it is built and added to PYTHONPATH.")

from loguru import logger

# Re-use utilities from the existing sim2real package
from utils.strings import resolve_matching_names_values, unitree_joint_names
from utils.onnx_module import Timer
from rl_policy.observations import Observation, ObsGroup

class G1StateProcessor:
    """Light-weight state processor reading data directly from ``g1_interface``.

    This class provides the minimal subset of attributes/methods expected by the
    existing observation modules so that they remain drop-in compatible.
    """

    def __init__(self, robot: "g1_interface.G1Interface", dest_joint_names: List[str]):
        self.robot = robot

        # Mapping from Isaac order (policy / sim order) to the internal Unitree order
        self.num_dof = len(dest_joint_names)
        self.joint_indices_in_source = [unitree_joint_names.index(name) for name in dest_joint_names]

        # Isaac style qpos / qvel buffers (same layout as original ``StateProcessor``)
        self.qpos = np.zeros(3 + 4 + self.num_dof)
        self.qvel = np.zeros(3 + 3 + self.num_dof)

        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]

        self.root_quat_b = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]

        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]

        self.zmq_context = zmq.Context()
        self.mocap_subscribers: Dict[str, zmq.Socket] = {}
        self.mocap_threads: Dict[str, threading.Thread] = {}
        self.mocap_data: Dict[str, Any] = {}
        self.mocap_data_lock = threading.Lock()

    def register_subscriber(self, object_name: str, port: int):
        if object_name in self.mocap_subscribers:
            return
        from utils.common import MOCAP_IP
        socket = self.zmq_context.socket(zmq.SUB)
        socket.connect(f"tcp://{MOCAP_IP}:{port}")
        socket.setsockopt(zmq.SUBSCRIBE, object_name.encode())
        socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms
        self.mocap_subscribers[object_name] = socket

        def _sub_thread(obj_name: str):
            while True:
                try:
                    msg = socket.recv_multipart(zmq.NOBLOCK)
                    if len(msg) == 2:
                        name = msg[0].decode()
                        data = np.frombuffer(msg[1], dtype=np.float64)
                        if len(data) == 7:
                            pos, quat = data[:3].copy(), data[3:].copy()
                            with self.mocap_data_lock:
                                self.mocap_data[f"{name}_pos"] = pos
                                self.mocap_data[f"{name}_quat"] = quat
                except zmq.Again:
                    time.sleep(0.001)
                except Exception as e:
                    logger.warning(f"{obj_name} subscriber error: {e}")
                    time.sleep(0.01)
        th = threading.Thread(target=_sub_thread, args=(object_name,), daemon=True)
        th.start()
        self.mocap_threads[object_name] = th

    def get_mocap_data(self, key: str):
        with self.mocap_data_lock:
            return self.mocap_data.get(key, None)

    # ------------------------------------------------------------------
    # Low-state preparation (called every control / RL step)
    # ------------------------------------------------------------------
    def _prepare_low_state(self) -> bool:
        try:
            state = self.robot.read_low_state()
        except Exception as e:
            logger.warning(f"Failed to read G1 low state: {e}")
            return False

        if state is None:
            return False

        # IMU
        self.root_quat_b[:] = state.imu.quat  # [w, x, y, z]
        self.root_ang_vel_b[:] = state.imu.omega

        # Joints
        for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
            self.joint_pos[dst_idx] = state.motor.q[src_idx]
            self.joint_vel[dst_idx] = state.motor.dq[src_idx]
        return True


# -------------------------------------------------------------------------------------------------
# Command sender wrapping MotorCommand for G1
# -------------------------------------------------------------------------------------------------
class G1CommandSender:
    """Thin wrapper translating Isaace style command arrays to ``MotorCommand``."""

    def __init__(
        self,
        robot: "g1_interface.G1Interface",
        policy_config: Dict[str, Any],
    ):
        """Create a command sender using policy-specified gains.

        Args:
            robot:          Active ``g1_interface`` instance.
            dest_joint_names:  Joint order used by the policy (Isaac order).
            policy_config:  Dict containing ``joint_kp``, ``joint_kd``, and ``default_joint_pos``.
        """

        self.robot = robot
        self.policy_config = policy_config
        # init robot and kp kd
        self._kp_level = 1.0  # 0.1

        joint_kp_dict = self.policy_config["joint_kp"]
        joint_indices, joint_names, joint_kp = resolve_matching_names_values(
            joint_kp_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.joint_kp_unitree_default = np.zeros(len(unitree_joint_names))
        self.joint_kp_unitree_default[joint_indices] = joint_kp
        self.joint_kp_unitree = self.joint_kp_unitree_default.copy()

        joint_kd_dict = self.policy_config["joint_kd"]
        joint_indices, joint_names, joint_kd = resolve_matching_names_values(
            joint_kd_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.joint_kd_unitree = np.zeros(len(unitree_joint_names))
        self.joint_kd_unitree[joint_indices] = joint_kd

        default_joint_pos_dict = self.policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_joint_pos_unitree = np.zeros(len(unitree_joint_names))
        self.default_joint_pos_unitree[joint_indices] = default_joint_pos

        joint_names_isaac = self.policy_config["isaac_joint_names"]
        self.joint_indices_unitree = [unitree_joint_names.index(name) for name in joint_names_isaac]

    # Expose kp_level so it can be tuned by UI callbacks (same API as original)
    @property
    def kp_level(self) -> float:
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value: float):
        self._kp_level = float(value)

    # --------------------------------------------------------------
    # Core API expected by BasePolicy
    # --------------------------------------------------------------
    def send_command(self, cmd_q: np.ndarray, cmd_dq: np.ndarray, cmd_tau: np.ndarray):
        """Construct a ``MotorCommand`` and dispatch it via the interface."""
        cmd = self.robot.create_zero_command()

        # Apply kp_level scaling (kd remains constant, consistent with original implementation)
        kp_scaled = self.joint_kp_unitree * self._kp_level
        kd_scaled = self.joint_kd_unitree

        q_target = list(cmd.q_target)
        dq_target = list(cmd.dq_target)
        tau_ff = list(cmd.tau_ff)
        kp = list(cmd.kp)
        kd = list(cmd.kd)
        for i_policy, idx_unitree in enumerate(self.joint_indices_unitree):
            q_target[idx_unitree] = float(cmd_q[i_policy])
            dq_target[idx_unitree] = float(cmd_dq[i_policy])
            tau_ff[idx_unitree] = float(cmd_tau[i_policy])
            kp[idx_unitree] = float(kp_scaled[idx_unitree])
            kd[idx_unitree] = float(kd_scaled[idx_unitree])

        cmd.q_target = q_target
        cmd.dq_target = dq_target
        cmd.tau_ff = tau_ff
        cmd.kp = kp
        cmd.kd = kd

        self.robot.write_low_command(cmd)
        
from sim2real.utils.robot import Robot
from sim2real.utils.history_handler import HistoryHandler
from sim2real.utils.state_processor import StateProcessor
from sim2real.utils.command_sender import CommandSender
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from multiprocessing import shared_memory
from sim2real.rl_policy.policy_logger import LoggerPolicy


class BasePolicy:
    def __init__(self, config, model_path, use_jit, rl_rate=50, policy_action_scale=0.25, decimation=4, debug_log=False):
        self.config = config
        self.robot = Robot(config)
        self.robot_state_data = None
        self.use_mocap = config.get("use_mocap", False)
        self.upper_body_controller = None
        if config.get("INTERFACE", None):
            if sys.platform == "linux":
                config["INTERFACE"] = "lo"
            elif sys.platform == "darwin":
                config["INTERFACE"] = "lo0"
            else:
                raise NotImplementedError("Only support Linux and MacOS.")
            ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
        else:
            ChannelFactoryInitialize(config["DOMAIN_ID"])

        self.state_processor = StateProcessor(config)
        self.command_sender = CommandSender(config)

        self.setup_policy(model_path, use_jit)

        self.num_dofs = self.robot.NUM_JOINTS
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.default_dof_angles = self.robot.DEFAULT_DOF_ANGLES
        self.policy_action_scale = policy_action_scale
        self.period = 1.0 / rl_rate  # Calculate period in seconds
        self.decimation = decimation

        # Keypress control state
        self.use_policy_action = False
        self.init_count = 0
        self.get_ready_state = False

        self.lin_vel_command = np.array([[0.0, 0.0]])
        self.ang_vel_command = np.array([[0.0]])
        self.stand_command = np.array([[0]])
        self.base_height_command = np.array([[0.78]])

        self.motor_pos_lower_limit_list = self.config.get("motor_pos_lower_limit_list", None)
        self.motor_pos_upper_limit_list = self.config.get("motor_pos_upper_limit_list", None)
        self.motor_vel_limit_list = self.config.get("motor_vel_limit_list", None)
        self.motor_effort_limit_list = self.config.get("motor_effort_limit_list", None)

        self.use_history = self.config["USE_HISTORY"]
        self.obs_scales = self.config["obs_scales"]
        self.history_handler = None
        self.current_obs = None
        if self.use_history:
            self.history_handler = HistoryHandler(self.config["history_config"], self.config["obs_dims"])
            self.current_obs = {key: np.zeros((1, self.config["obs_dims"][key])) for key in self.config["obs_dims"].keys()}

        if self.config.get("USE_ROS", False):
            import rclpy

            rclpy.init(args=None)
            self.node = rclpy.create_node("policy_node")
            self.logger = self.node.get_logger()
            self.rate = self.node.create_rate(rl_rate)
            thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
            thread.start()
        else:
            from loguru import logger

            self.logger = logger
            self.rate = RateLimiter(rl_rate)

        if self.config.get("USE_JOYSTICK", False):
            if sys.platform == "darwin":
                self.logger.warning("Joystick is not supported on Windows or Mac.")
                self.logger.warning("Using keyboard instead")
                self.use_joystick = False
                self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
                self.key_listener_thread.start()
            else:
                # Yuanhang: pygame event can only run in main thread on Mac, so we need to implement it with rl inference
                self.logger.info("Using joystick")
                self.use_joystick = True
                self.key_states = {}
                self.last_key_states = {}
                self.wireless_controller_subscriber = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
                self.wireless_controller_subscriber.Init(self.WirelessControllerHandler, 1)
                self.wc_msg = None
                self.wc_key_map = {
                    1: "R1",
                    2: "L1",
                    3: "L1+R1",
                    4: "start",
                    8: "select",
                    16: "R2",
                    32: "L2",
                    64: "F1",  # not used in sim2sim
                    128: "F2",  # not used in sim2sim
                    256: "A",
                    512: "B",
                    768: "A+B",
                    1024: "X",
                    1280: "A+X",
                    2048: "Y",
                    2304: "A+Y",
                    2560: "B+Y",
                    3072: "X+Y",
                    4096: "up",
                    4097: "R1+up",
                    4352: "A+up",
                    4608: "B+up",
                    5120: "X+up",
                    6144: "Y+up",
                    8192: "right",
                    8193: "R1+right",
                    8448: "A+right",
                    9216: "X+right",
                    10240: "Y+right",
                    16384: "down",
                    16385: "R1+down",
                    16640: "A+down",
                    16896: "B+down",
                    17408: "X+down",
                    18432: "Y+down",
                    32768: "left",
                    32769: "R1+left",
                    33024: "A+left",
                    33792: "X+left",
                    34816: "Y+left",
                }
                print("Wireless Controller Initialized")
        else:
            self.logger.info("Using keyboard")
            self.use_joystick = False
            self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
            self.key_listener_thread.start()

        num_states_digits = 7 + 6 + 6 + 6 + self.num_dofs * 4
        # self.shm_state = shared_memory.SharedMemory(name="robot_state_data_shm", create=True, size=8*num_states_digits)
        # self.robot_state_data_shm = np.ndarray((1, num_states_digits), dtype=np.float64, buffer=self.shm_state.buf)
        self.logger_new = LoggerPolicy(disable=not debug_log)
        self.additiona_info_to_log = {}

    def WirelessControllerHandler(self, msg: WirelessController_):
        # print(f"Raw keys value: {msg.keys}")
        self.wc_msg = msg

    def setup_policy(self, model_path, use_jit):
        # load onnx policy
        if not use_jit:
            self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
            self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
            self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name

            def policy_act(obs):
                return self.onnx_policy_session.run([self.onnx_output_name], {self.onnx_input_name: obs})[0]
        else:
            self.jit_policy = torch.jit.load(model_path)

            def policy_act(obs):
                obs = torch.tensor(obs)
                action_10dof = self.jit_policy(obs)
                action_19dof = torch.cat([action_10dof, torch.zeros(1, 9)], dim=1)
                return action_19dof.detach().numpy()

        self.policy = policy_act

    def prepare_obs_for_rl(self, robot_state_data):
        # robot_state_data [:3]: robot base pos
        # robot_state_data [3:7]: robot base quaternion
        # robot_state_data [7:7+dof_num]: joint angles
        # robot_state_data [7+dof_num: 7+dof_num+3]: base linear velocity
        # robot_state_data [7+dof_num+3: 7+dof_num+6]: base angular velocity
        # robot_state_data [7+dof_num+6: 7+dof_num+6+dof_num]: joint velocities
        raise NotImplementedError

    def get_init_target(self, robot_state_data):
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # interpolate from current dof_pos to default angles
            q_target = dof_pos + (self.default_dof_angles - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        else:
            return dof_pos

    def _get_obs_history(
        self,
    ):
        assert "history_config" in self.config.keys()
        history_config = self.config["history_config"]
        history_list = []

        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_array = self.history_handler.query(key)[:, :history_length]
            history_array = history_array.reshape(history_array.shape[0], -1)  # Shape: [4096, history_length*obs_dim]

            history_list.append(history_array)

        return np.concatenate(history_list, axis=1)

    def rl_inference(self, robot_state_data):
        # Process low states
        obs = self.prepare_obs_for_rl(robot_state_data)
        # Policy inference
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale

        return scaled_policy_action

    def policy_action(self):
        self.additiona_info_to_log = {}

        # Get states
        self.robot_state_data = self.state_processor._prepare_low_state()
        # self.robot_state_data_shm[0] = self.robot_state_data
        # Get policy action
        scaled_policy_action = self.rl_inference(self.robot_state_data)
        if self.get_ready_state:
            # 1. Set to Default Joint Position: interpolate from current dof_pos to default angles
            q_target = self.get_init_target(self.robot_state_data)
            if self.init_count > 500:
                self.init_count = 500
        elif not self.use_policy_action:
            # 2. No Policy Action: set to zero
            q_target = self.robot_state_data[:, 7 : 7 + self.num_dofs]
        else:
            # 3. Policy Action: apply policy action to current joint angles
            if not scaled_policy_action.shape[1] == self.num_dofs:
                if not self.upper_body_controller:
                    scaled_policy_action = np.concatenate(
                        [scaled_policy_action, np.zeros((1, self.num_dofs - scaled_policy_action.shape[1]))], axis=1
                    )
                else:
                    raise NotImplementedError("Upper body controller not implemented")
            q_target = scaled_policy_action + self.default_dof_angles

        # Clip q target
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        # Send command
        cmd_q = q_target[0]
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)
        # log the command
        self.additiona_info_to_log.update(
            dict(
                q_target=q_target.tolist(),
                default_dof_angles=self.default_dof_angles,
                robot_state_data=self.robot_state_data.tolist(),
                scaled_policy_action=scaled_policy_action.tolist(),
            )
        )
        self.logger_new.log_policy_action(
            low_cmd=self.command_sender.low_cmd, low_state=self.state_processor.robot_low_state, additional_info=self.additiona_info_to_log
        )

    def start_key_listener(self):
        """Start a key listener using pynput."""

        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except AttributeError:
                pass  # Handle special keys if needed

        listener = listen_keyboard(on_press=on_press)
        listener.start()
        listener.join()  # Keep the thread alive

    def handle_keyboard_button(self, keycode):
        if keycode == "]":
            self.use_policy_action = True
            self.get_ready_state = False
            self.logger.info("Using policy actions")
            # self.frame_start_time = time.perf_counter()
            self.phase = 0.0
        elif keycode == "o":
            self.use_policy_action = False
            self.get_ready_state = False
            self.logger.info("Actions set to zero")
        elif keycode == "i":
            self.get_ready_state = True
            self.init_count = 0
            self.logger.info("Setting to init state")
        elif keycode == "w" and self.stand_command:
            self.lin_vel_command[0, 0] += 0.1
        elif keycode == "s" and self.stand_command:
            self.lin_vel_command[0, 0] -= 0.1
        elif keycode == "a" and self.stand_command:
            self.lin_vel_command[0, 1] += 0.1
        elif keycode == "d" and self.stand_command:
            self.lin_vel_command[0, 1] -= 0.1
        elif keycode == "q":
            self.ang_vel_command[0, 0] -= 0.1
        elif keycode == "e":
            self.ang_vel_command[0, 0] += 0.1
        elif keycode == "z":
            self.ang_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 1] = 0.0
        elif keycode == "1":
            self.base_height_command += 0.05
        elif keycode == "2":
            self.base_height_command -= 0.05
        elif keycode == "5":
            self.command_sender.kp_level -= 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "6":
            self.command_sender.kp_level += 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "4":
            self.command_sender.kp_level -= 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "7":
            self.command_sender.kp_level += 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "0":
            self.command_sender.kp_level = 1.0
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "=":
            self.stand_command = 1 - self.stand_command
            if self.stand_command == 0:
                self.ang_vel_command[0, 0] = 0.0
                self.lin_vel_command[0, 0] = 0.0
                self.lin_vel_command[0, 1] = 0.0
        print(f"Linear velocity command: {self.lin_vel_command}")
        print(f"Angular velocity command: {self.ang_vel_command}")
        print(f"Base height command: {self.base_height_command}")
        print(f"Stand command: {self.stand_command}")

    def process_joystick_input(self):
        # Process stick
        if self.wc_msg.keys == 0:
            self.lin_vel_command[0, 1] = -(self.wc_msg.lx if abs(self.wc_msg.lx) > 0.1 else 0.0) * self.stand_command[0, 0] * 0.5
            self.lin_vel_command[0, 0] = (self.wc_msg.ly if abs(self.wc_msg.ly) > 0.1 else 0.0) * self.stand_command[0, 0]
            self.ang_vel_command[0, 0] = -(self.wc_msg.rx if abs(self.wc_msg.rx) > 0.1 else 0.0) * self.stand_command[0, 0]
        cur_key = self.wc_key_map.get(self.wc_msg.keys, None)
        self.last_key_states = self.key_states.copy()
        if cur_key:
            self.key_states[cur_key] = True
        else:
            self.key_states = {key: False for key in self.wc_key_map.values()}

        for key, is_pressed in self.key_states.items():
            if is_pressed and not self.last_key_states.get(key, False):
                self.handle_joystick_button(key)

    def handle_joystick_button(self, cur_key):
        # Handle button press
        if cur_key == "start":
            # self.history_handler.reset([0])
            self.use_policy_action = True
            self.get_ready_state = False
            self.logger.info(colored("Using policy actions", "blue"))
            self.phase = 0.0
            self.command_sender.no_action = 0
        elif cur_key == "B+Y":
            self.use_policy_action = False
            self.get_ready_state = False
            self.command_sender.no_action = 1
            self.logger.info(colored("Actions set to zero", "blue"))
        elif cur_key == "A+X":
            self.get_ready_state = True
            self.init_count = 0
            self.command_sender.no_action = 0
            self.logger.info(colored("Setting to init state", "blue"))
        elif cur_key == "B+up" and not self.stand_command:
            self.base_height_command[0, 0] += 0.05
            self.logger.info(colored(f"Base height command: {self.base_height_command[0, 0]}", "green"))
        elif cur_key == "B+down" and not self.stand_command:
            self.base_height_command[0, 0] -= 0.05
            self.logger.info(colored(f"Base height command: {self.base_height_command[0, 0]}", "green"))
        elif cur_key == "Y+left":
            self.command_sender.kp_level -= 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif cur_key == "Y+right":
            self.command_sender.kp_level += 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif cur_key == "A+left":
            self.command_sender.kp_level -= 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif cur_key == "A+right":
            self.command_sender.kp_level += 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif cur_key == "A+Y":
            self.command_sender.kp_level = 1.0
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            self.logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif cur_key == "L2":
            self.lin_vel_command[0, :] *= 0.0
            self.ang_vel_command[0, :] *= 0.0
            self.logger.info(colored("Velocities set to zero", "blue"))
        elif cur_key == "R2":
            self.stand_command = 1 - self.stand_command
            if self.stand_command == 0:
                self.ang_vel_command[0, 0] = 0.0
                self.lin_vel_command[0, 0] = 0.0
                self.lin_vel_command[0, 1] = 0.0
                self.logger.info(colored("Stance command", "blue"))
            else:
                self.base_height_command[0, 0] = 0.78
                self.logger.info(colored("Walk command", "blue"))

    def run(self):
        total_inference_cnt = 0
        start_time = time.time()
        try:
            while True:
                if self.use_joystick and self.wc_msg is not None:
                    self.process_joystick_input()
                self.policy_action()
                end_time = time.time()
                total_inference_cnt += 1

                self.rate.sleep()
        except KeyboardInterrupt:
            pass
