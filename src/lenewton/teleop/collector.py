from __future__ import annotations

"""Socket-based teleoperation utilities for LeNewton."""

import json
import os
import socket
import time
from datetime import datetime
from threading import Event, Thread
from typing import Callable

import numpy as np
from pynput import keyboard

from lenewton.env import DEFAULT_JOINTS, LeNewtonEnv, create_lenewton_env
from lenewton.teleop.dataset import DatasetRecorder

JOINT_OFFSETS = np.array([[0.0, 1.70, -1.70, -0.921, -0.0120, 0.0]])


class DatasetCollector:
    """Handle socket-based teleoperation and dataset recording."""

    def __init__(
        self,
        task_name: str = "BlockStack",
        socket_host: str = "localhost",
        socket_port: int = 12345,
        seed: int = 42,
        use_viewer: bool = True,
        record_frequency: float = 50.0,
        env_factory: Callable[..., LeNewtonEnv] = create_lenewton_env,
        joint_offsets: np.ndarray = JOINT_OFFSETS,
    ):
        self.task_name = task_name
        self.socket_host = socket_host
        self.socket_port = socket_port
        self.record_frequency = record_frequency
        self.record_period = 1.0 / record_frequency
        self.joint_offsets = joint_offsets

        # Initialize LeNewton environment
        self.env = env_factory(
            task_name,
            record_video=False,
            use_viewer=use_viewer,
            teleop_mode=True,
            randomize=True,
        )
        self.env.reset()
        self.env.set_viewer_pose(
            pos=[0.0, -1.25, 0.5],
            pitch=-25.0,
            yaw=90.0,
        )

        self.action_dim = self.env.robot_dof_count
        self._stabilize_environment()

        self.socket = None
        self.running = False
        self.stop_event = Event()

        self.current_joint_targets = None
        self.previous_joint_targets = None
        self.current_gripper = 0.0

        self.control_frequency = 50.0  # Hz
        self.control_period = 1.0 / self.control_frequency

        self.recorder = DatasetRecorder(task_name=task_name, seed=seed)
        self.last_record_time = 0.0

        self.save_requested = False
        self.discard_requested = False
        self.keyboard_listener = None

        self.seed = seed
        self.use_viewer = use_viewer

        print("LeNewton Socket Collector initialized")
        print(f"Recording metadata: task={task_name}, seed={seed}")

    def _stabilize_environment(self) -> None:
        """Warm up the environment with default joint positions."""
        for _ in range(10):
            self.env.step(DEFAULT_JOINTS)

    def _recreate_environment(self) -> None:
        """Close current environment and create new one with a random seed."""
        new_seed = np.random.randint(0, 100000)
        print(f"\n[RECREATE] Closing current environment (seed={self.seed})...")

        if self.env is not None:
            self.env.close()
            print("Environment closed")

        print(f"Creating new environment with seed={new_seed}...")
        self.env = create_lenewton_env(
            self.task_name,
            record_video=False,
            use_viewer=self.use_viewer,
            teleop_mode=True,
            randomize=True,
        )
        self.env.reset()
        print("New environment created")

        self.action_dim = self.env.robot_dof_count
        self.seed = new_seed
        self.recorder = DatasetRecorder(task_name=self.task_name, seed=new_seed)
        print(f"Recorder initialized with new seed: {new_seed}")

        print("Stabilizing new environment...")
        self._stabilize_environment()
        print("Stabilization complete")

        self.current_joint_targets = None
        self.previous_joint_targets = None
        self.current_gripper = 0.0

        print(f"\n[SUCCESS] Ready to collect new episode with seed={new_seed}!\n")

    def _on_key_press(self, key) -> None:
        """Handle keyboard press events."""
        try:
            if hasattr(key, "char"):
                if key.char == "a":
                    print("\n[KEYBOARD] 'a' key pressed - Save requested")
                    self.save_requested = True
                elif key.char == "c":
                    print("\n[KEYBOARD] 'c' key pressed - Discard requested")
                    self.discard_requested = True
        except AttributeError:
            pass

    def _start_keyboard_listener(self) -> None:
        self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self.keyboard_listener.start()
        print("Keyboard listener started")
        print("Press 'a' to save and finish current episode")
        print("Press 'c' to discard and reset episode")

    def _stop_keyboard_listener(self) -> None:
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
            print("Keyboard listener stopped")

    def connect_to_sender(self) -> bool:
        """Connect to the SO100 leader socket sender."""
        print(
            f"Connecting to socket server at {self.socket_host}:{self.socket_port}..."
        )
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.socket_host, self.socket_port))
        self.socket.settimeout(1.0)
        print("Connected to socket server")
        return True

    def parse_joint_message(self, message: str) -> dict | None:
        """Parse JSON joint target message from socket.

        Expected leader format (single arm):
          {
            "timestamp": <float>,
            "end_effector": {...},
            "joints": {"arm": [j0, j1, ..., jN]}
          }

        The joint list is padded/truncated to match the action dimension. If the
        message omits a gripper value, the final slot reuses the last gripper
        command (starts at 0.0).
        """
        data = json.loads(message)

        if not (isinstance(data.get("joints"), dict) and "arm" in data["joints"]):
            print(f"Received message without joints.arm field: {data.keys()}")
            return None

        arm_joints = data["joints"]["arm"]
        joints = np.array(list(arm_joints.values()), dtype=float)

        return {
            "joint_positions": joints,
            "timestamp": data.get("timestamp", time.time()),
        }

    def update_robot_joints(
        self, joint_command: dict
    ) -> tuple[dict, float, bool, dict, dict]:
        """Apply joint targets directly via env.step."""
        joints = np.asarray(joint_command["joint_positions"], dtype=float)

        joints_rad = np.deg2rad(joints) + self.joint_offsets

        observation, reward, done, info = self.env.step(joints_rad)

        self.previous_joint_targets = joints_rad.copy()
        self.current_gripper = joints_rad[-1]

        action_dict = {
            "joint_positions": joints_rad.copy(),
            "timestamp": joint_command.get("timestamp", time.time()),
        }

        return observation, reward, done, info, action_dict

    def record_current_state(
        self, observation: dict, reward: float, done: bool, info: dict, action: dict
    ) -> None:
        """Record current state to dataset."""
        self.recorder.record_step(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            action=action,
        )

    def receive_poses_thread(self) -> None:
        """Thread function to receive joint data from socket."""
        buffer = ""

        while not self.stop_event.is_set():
            try:
                data = self.socket.recv(4096).decode("utf-8")
                if not data:
                    print("Socket connection closed by sender")
                    break

                buffer += data

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        joint_command = self.parse_joint_message(line.strip())
                        if joint_command:
                            self.current_joint_targets = joint_command

            except TimeoutError:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"Error receiving joint data: {e}")
                break

        print("Joint reception thread stopped")

    def _save_current_episode(self, output_dir: str) -> None:
        """Persist the current recording buffers to disk."""
        if self.recorder.step_count == 0:
            print("No data to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        npz_filename = os.path.join(output_dir, f"{self.task_name}_demo_{timestamp}.npz")
        self.recorder.save_dataset(npz_filename)

        json_filename = os.path.join(
            output_dir, f"{self.task_name}_demo_{timestamp}_info.json"
        )
        self.recorder.save_dataset_json(json_filename)

    def control_loop(self, output_dir: str) -> None:
        """Main control loop with dataset recording."""
        print(f"Starting control loop at {self.control_frequency} Hz...")
        print(f"Recording dataset at {self.record_frequency} Hz...")
        print("Use keyboard controls to save or discard data:")
        print("  'a' key: Save current episode and recreate environment")
        print("  'c' key: Discard current episode and recreate environment")
        print("  Ctrl+C: Save and exit program")
        print("\nNote: Teleoperation continues even after task completion (done=True)")
        print(
            f"Expecting messages with 'joints.arm' length {self.action_dim} (last entry = gripper)."
        )

        self._start_keyboard_listener()

        receive_thread = Thread(target=self.receive_poses_thread, daemon=True)
        receive_thread.start()

        last_control_time = time.time()
        last_observation = None
        last_reward = None
        last_info = None
        last_action = None
        last_done = False

        try:
            while self.running and not self.stop_event.is_set():
                current_time = time.time()

                if self.save_requested:
                    print("\n\n[SAVE] Saving dataset...")
                    print(f"Total steps recorded: {self.recorder.step_count}")
                    self._save_current_episode(output_dir)
                    self.save_requested = False
                    self._recreate_environment()
                    last_control_time = time.time()
                    self.last_record_time = time.time()
                    continue

                if self.discard_requested:
                    print("\n\n[DISCARD] Discarding current data...")
                    print(f"Discarded {self.recorder.step_count} steps")
                    self.discard_requested = False
                    self._recreate_environment()
                    last_control_time = time.time()
                    self.last_record_time = time.time()
                    continue

                if (current_time - last_control_time) >= self.control_period:
                    if self.current_joint_targets is not None:
                        (
                            observation,
                            reward,
                            done,
                            info,
                            action,
                        ) = self.update_robot_joints(self.current_joint_targets)

                        last_observation = observation
                        last_reward = reward
                        last_info = info
                        last_action = action
                        last_done = done

                        if (
                            current_time - self.last_record_time
                        ) >= self.record_period and last_observation is not None:
                            self.record_current_state(
                                observation=last_observation,
                                reward=last_reward,
                                done=last_done,
                                info=last_info,
                                action=last_action,
                            )
                            self.last_record_time = current_time

                            if (
                                self.recorder.step_count
                                % (int(self.record_frequency * 2))
                                == 0
                            ):
                                print(f"Recording... Steps: {self.recorder.step_count}")

                        if done:
                            print(
                                f"\n[INFO] Task completed (done=True). Steps: {self.recorder.step_count}"
                            )
                            print(
                                "Continuing teleoperation... Press 'a' to save or 'c' to discard"
                            )

                    last_control_time = current_time

                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\nControl loop interrupted by user (Ctrl+C)")
            if self.recorder.step_count > 0:
                print(f"Saving {self.recorder.step_count} steps before exit...")
                self._save_current_episode(output_dir)

        self._stop_keyboard_listener()
        self.stop_event.set()

        if receive_thread.is_alive():
            receive_thread.join(timeout=2.0)

    def run(self, output_dir: str = "datasets/lenewton") -> bool:
        """Run collection loop continuously until Ctrl+C."""
        if not self.connect_to_sender():
            return False

        os.makedirs(output_dir, exist_ok=True)

        self.running = True
        self.control_loop(output_dir)

        if self.env is not None:
            self.env.close()

        print("\n[EXIT] Collection session ended")
        return True

    def disconnect(self) -> None:
        """Disconnect from socket and cleanup."""
        self.running = False
        self.stop_event.set()

        if self.socket:
            self.socket.close()
            self.socket = None

        print("Disconnected from socket server")
