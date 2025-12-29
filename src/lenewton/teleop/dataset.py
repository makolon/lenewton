from __future__ import annotations

"""Utilities for recording LeNewton teleoperation datasets."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class DatasetRecorder:
    """Records LeNewton demonstration data including observations, rewards, and info."""

    task_name: str = ""
    seed: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str | None = None

    def record_step(
        self,
        observation: dict[str, Any],
        reward: float,
        done: bool,
        info: dict[str, Any],
        action: dict[str, Any] | None = None,
    ) -> None:
        """Record a single step of the demonstration."""
        self.observations.append(observation)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
        self.actions.append(action or {})
        self.step_count += 1

    def _build_metadata(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "seed": self.seed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_steps": self.step_count,
        }

    def save_dataset(self, output_path: str) -> None:
        """Save dataset to NPZ file."""
        self.end_time = datetime.now().isoformat()

        dataset = {
            "observations": self.observations,
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "infos": self.infos,
            "actions": self.actions,
            "num_steps": self.step_count,
            "metadata": self._build_metadata(),
        }

        np.savez_compressed(output_path, **dataset)

        print(f"\nDataset saved to: {output_path}")
        print(f"Total steps recorded: {self.step_count}")
        print("Dataset contains:")
        print(f"  - observations: {len(self.observations)} steps")
        print(f"  - rewards: {len(self.rewards)} values")
        print(f"  - dones: {len(self.dones)} flags")
        print(f"  - infos: {len(self.infos)} dicts")
        print(f"  - actions: {len(self.actions)} commands")
        print(
            f"  - metadata: task={self.task_name}, seed={self.seed}, "
            f"start={self.start_time}, end={self.end_time}"
        )

    def save_dataset_json(self, output_path: str) -> None:
        """Save dataset in JSON format (lighter, skips large arrays like images)."""
        if self.end_time is None:
            self.end_time = datetime.now().isoformat()

        dataset_simplified = {
            "metadata": self._build_metadata(),
            "num_steps": self.step_count,
            "rewards": self.rewards,
            "infos": self.infos,
            "actions": self.actions,
        }

        with open(output_path, "w") as f:
            json.dump(
                dataset_simplified,
                f,
                indent=2,
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x),
            )

        print(f"Simplified dataset (JSON) saved to: {output_path}")

    def clear(self) -> None:
        """Clear recorded data."""
        self.observations = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.actions = []
        self.step_count = 0
        self.start_time = datetime.now().isoformat()
        self.end_time = None
