from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lenewton.envs.lenewton.env import LeNewtonEnv

log = logging.getLogger(__name__)


###
# Base Task
###


class LeNewtonTask:
    """Base class for LeNewton tasks."""

    def __init__(
        self, task_name: str, goal_str: str = "", max_steps: int = 200, **kwargs
    ):
        self.task_name = task_name
        self.goal_str = goal_str or f"Complete the {task_name} task"
        self.max_steps = max_steps
        self.step_count = 0

        # Object tracking
        self.object_bodies: dict[str, int] = {}  # name -> body index

    def setup_env(self, env: LeNewtonEnv, **kwargs) -> None:
        """Setup environment for the task."""
        self.env = env

    def get_reward(self, env: LeNewtonEnv) -> float:
        """Get reward for current state."""
        return 0.0

    def get_info(self, env: LeNewtonEnv) -> dict[str, any]:
        """Get additional task information."""
        return {
            "task_name": self.task_name,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }

    def get_done(self, env: LeNewtonEnv) -> bool:
        """Check if task is complete."""
        # Check step limit
        if self.step_count >= self.max_steps:
            return True
        # Default completion check based on reward
        reward = self.get_reward(env)
        return reward > 0.95

    def get_goal(self) -> str:
        """Get the goal description for this task."""
        return self.goal_str

    def get_object_position(self, object_name: str) -> np.ndarray | None:
        """Get the current position of an object by name.

        Args:
            object_name: Name of the object

        Returns:
            Position as numpy array [x, y, z] or None if not found
        """
        if self.env is None or object_name not in self.object_bodies:
            return None

        body_idx = self.object_bodies[object_name]
        # Get body transform from env state
        body_q = self.env.state_0.body_q.numpy()[body_idx]
        # body_q format: [quat_w, quat_x, quat_y, quat_z, pos_x, pos_y, pos_z]
        return np.array([body_q[4], body_q[5], body_q[6]])

    def get_object_orientation(self, object_name: str) -> np.ndarray | None:
        """Get the current orientation (quaternion) of an object by name.

        Args:
            object_name: Name of the object

        Returns:
            Orientation as numpy array [w, x, y, z] or None if not found
        """
        if self.env is None or object_name not in self.object_bodies:
            return None

        body_idx = self.object_bodies[object_name]
        # Get body transform from env state
        body_q = self.env.state_0.body_q.numpy()[body_idx]
        # body_q format: [quat_w, quat_x, quat_y, quat_z, pos_x, pos_y, pos_z]
        return np.array([body_q[0], body_q[1], body_q[2], body_q[3]])
