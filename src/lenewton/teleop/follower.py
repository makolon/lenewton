from __future__ import annotations

"""LeRobot-compatible follower that drives the LeNewton simulation."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lenewton.env import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    SO100_JOINTS,
    LeNewtonEnv,
)


@dataclass(kw_only=True)
class LeNewtonFollowerConfig(RobotConfig):
    """Configuration for the simulated follower."""

    task_name: str = "BlockStack"
    use_viewer: bool = False
    render_mode: str = "rgb_array"
    image_size: tuple[int, int] = (480, 640)
    include_camera: bool = True


class LeNewtonFollower(Robot):
    """Wrap LeNewtonEnv with the LeRobot Robot interface."""

    config_class = LeNewtonFollowerConfig
    name = "lenewton_follower"

    def __init__(self, config: LeNewtonFollowerConfig):
        super().__init__(config)
        self.config = config
        self.env: LeNewtonEnv | None = None
        self._connected = False
        self._joint_names = list(SO100_JOINTS)
        self._lower: np.ndarray | None = None
        self._upper: np.ndarray | None = None

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        joints = {f"{name}.pos": float for name in self._joint_names}
        if not self.config.include_camera:
            return joints
        return {
            **joints,
            "camera": (*self.config.image_size, 4),
        }

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self._joint_names}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:  # calibrate kept for API symmetry
        if self._connected:
            return

        self.env = LeNewtonEnv(
            task_name=self.config.task_name,
            render_mode=self.config.render_mode,
            image_size=self.config.image_size,
            use_viewer=self.config.use_viewer,
        )
        self.env.reset()

        limits = self.env.robot_view.get_attribute("joint_limit_lower", self.env.model).numpy().flatten()
        self._lower = np.array(limits[: len(self._joint_names)], dtype=np.float32)
        limits = self.env.robot_view.get_attribute("joint_limit_upper", self.env.model).numpy().flatten()
        self._upper = np.array(limits[: len(self._joint_names)], dtype=np.float32)

        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        # Simulation has no calibration step
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def get_observation(self) -> dict[str, Any]:
        if not self._connected or self.env is None:
            raise RuntimeError(f"{self} is not connected.")

        obs = self.env._get_observation()
        joint_pos = obs["joint_positions"]
        obs_dict: dict[str, Any] = {}
        for i, name in enumerate(self._joint_names):
            obs_dict[f"{name}.pos"] = self._normalize_joint(joint_pos[i], i)

        if self.config.include_camera:
            obs_dict["camera"] = obs[self.env.camera_name]

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._connected or self.env is None:
            raise RuntimeError(f"{self} is not connected.")

        targets = []
        for i, name in enumerate(self._joint_names):
            key = f"{name}.pos"
            if key not in action:
                raise KeyError(f"Missing action for joint '{key}'")
            targets.append(self._denormalize_joint(float(action[key]), i))

        # Step the sim with target joint positions
        obs, _, terminated, truncated, info = self.env.step(np.asarray(targets, dtype=np.float32))
        # Return the action actually applied (normalized)
        applied = {}
        for i, name in enumerate(self._joint_names):
            applied[f"{name}.pos"] = self._normalize_joint(obs["joint_positions"][i], i)
        applied["terminated"] = terminated
        applied["truncated"] = truncated
        applied["sim_time"] = info.get("sim_time", 0.0)
        return applied

    def disconnect(self) -> None:
        if self.env is not None:
            self.env.close()
        self.env = None
        self._connected = False

    def _normalize_joint(self, value: float, idx: int) -> float:
        assert self._lower is not None and self._upper is not None
        lo, hi = self._lower[idx], self._upper[idx]
        if idx == len(self._joint_names) - 1:
            # gripper mapped to [0, 100]
            return np.clip((value - GRIPPER_CLOSE) / (GRIPPER_OPEN - GRIPPER_CLOSE) * 100.0, 0.0, 100.0)
        # map to [-100, 100] based on limits
        return float(np.clip(((value - lo) / (hi - lo)) * 200.0 - 100.0, -100.0, 100.0))

    def _denormalize_joint(self, value: float, idx: int) -> float:
        assert self._lower is not None and self._upper is not None
        if idx == len(self._joint_names) - 1:
            val = np.clip(value, 0.0, 100.0) / 100.0
            return GRIPPER_CLOSE + val * (GRIPPER_OPEN - GRIPPER_CLOSE)

        val = np.clip(value, -100.0, 100.0)
        lo, hi = self._lower[idx], self._upper[idx]
        return float((val + 100.0) / 200.0 * (hi - lo) + lo)
