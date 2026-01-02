from __future__ import annotations

"""LeRobot-based teleoperation utilities for LeNewton."""

import os
import time
from datetime import datetime
from threading import Event
from typing import Callable

from pynput import keyboard

from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lenewton.teleop.follower import (
    LeNewtonFollower,
    LeNewtonFollowerConfig,
)
from lenewton.teleop.dataset import DatasetRecorder


class DatasetCollector:
    """Handle LeRobot-based teleoperation and dataset recording."""

    def __init__(
        self,
        task_name: str,
        leader_port: str = "/dev/ttyUSB0",
        leader_id: str = "leader",
        follower_id: str = "lenewton",
        use_viewer: bool = True,
        record_frequency: float = 50.0,
        follower_factory: Callable[..., LeNewtonFollower] | None = None,
    ):
        self.task_name = task_name
        self.record_frequency = record_frequency
        self.record_period = 1.0 / record_frequency

        self.leader_config = SO100LeaderConfig(port=leader_port, id=leader_id)
        follower_factory = follower_factory or (
            lambda: LeNewtonFollower(
                LeNewtonFollowerConfig(
                    id=follower_id,
                    use_viewer=use_viewer,
                )
            )
        )
        self.follower = follower_factory()
        self.leader = SO100Leader(self.leader_config)

        self.running = False
        self.stop_event = Event()

        self.control_frequency = 50.0  # Hz
        self.control_period = 1.0 / self.control_frequency

        self.recorder = DatasetRecorder(task_name=task_name, seed=int(time.time()))
        self.last_record_time = 0.0

        self.save_requested = False
        self.discard_requested = False
        self.keyboard_listener = None

        self.use_viewer = use_viewer

        print("LeNewton LeRobot Collector initialized")
        print(f"Recording metadata: task={task_name}")

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

    def update_robot_joints(
        self, joint_command: dict
    ) -> tuple[dict, float, bool, dict, dict]:
        """Apply joint targets from leader to follower."""
        applied = self.follower.send_action(joint_command)
        observation = self.follower.get_observation()
        done = bool(applied.get("terminated", False) or applied.get("truncated", False))
        info = {"sim_time": applied.get("sim_time", 0.0)}
        action_dict = {k: v for k, v in applied.items() if k.endswith(".pos")}
        return observation, 0.0, done, info, action_dict

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

        # Connect devices
        print("Connecting leader...")
        self.leader.connect()
        print("Connecting follower (simulation)...")
        self.follower.connect()
        print("Connected. Starting control loop.")

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
                    self.recorder.clear()
                    self.save_requested = False
                    last_control_time = time.time()
                    self.last_record_time = time.time()
                    continue

                if self.discard_requested:
                    print("\n\n[DISCARD] Discarding current data...")
                    print(f"Discarded {self.recorder.step_count} steps")
                    self.recorder.clear()
                    self.discard_requested = False
                    last_control_time = time.time()
                    self.last_record_time = time.time()
                    continue

                if (current_time - last_control_time) >= self.control_period:
                    leader_action = self.leader.get_action()
                    (
                        observation,
                        reward,
                        done,
                        info,
                        action,
                    ) = self.update_robot_joints(leader_action)

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

                        if self.recorder.step_count % (int(self.record_frequency * 2)) == 0:
                            print(f"Recording... Steps: {self.recorder.step_count}")

                    if done:
                        print(
                            f"\n[INFO] Task completed (done=True). Steps: {self.recorder.step_count}"
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

    def run(self, output_dir: str = "datasets/lenewton") -> bool:
        """Run collection loop continuously until Ctrl+C."""
        os.makedirs(output_dir, exist_ok=True)

        self.running = True
        self.control_loop(output_dir)

        self.leader.disconnect()
        self.follower.disconnect()

        print("\n[EXIT] Collection session ended")
        return True

    def disconnect(self) -> None:
        """Disconnect leader/follower and cleanup."""
        self.running = False
        self.stop_event.set()
        self.leader.disconnect()
        self.follower.disconnect()
        print("Disconnected leader and follower")
