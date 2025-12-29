from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton
import numpy as np
import warp as wp

from lenewton.tasks.lenewton_task import LeNewtonTask

if TYPE_CHECKING:
    from lenewton.env import LeNewtonEnv

log = logging.getLogger(__name__)


###
# Block Configuration
###


@dataclass
class BlockConfig:
    """Configuration for a colored block."""

    name: str
    color: tuple[float, float, float]  # RGB values (0-1)
    size: tuple[float, float, float] = (0.025, 0.025, 0.025)  # Width, depth, height
    initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0)


###
# Basic LeNewton Tasks
###


class BlockStackTask(LeNewtonTask):
    """Block stacking task with 3 colored blocks (blue, red, yellow).

    The goal is to stack the three blocks vertically in a specific order.
    Default stacking order (bottom to top): blue -> red -> yellow
    """

    # Block definitions
    BLUE_BLOCK = BlockConfig(
        name="blue_block",
        color=(0.2, 0.4, 0.9),  # Blue
        size=(0.025, 0.025, 0.025),
    )
    RED_BLOCK = BlockConfig(
        name="red_block",
        color=(0.9, 0.2, 0.2),  # Red
        size=(0.025, 0.025, 0.025),
    )
    YELLOW_BLOCK = BlockConfig(
        name="yellow_block",
        color=(0.95, 0.85, 0.2),  # Yellow
        size=(0.025, 0.025, 0.025),
    )

    def __init__(
        self,
        task_name: str = "BlockStack",
        goal_str: str = "Stack the three blocks: blue on bottom, red in middle, yellow on top",
        max_steps: int = 300,
        stacking_order: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(
            task_name=task_name, goal_str=goal_str, max_steps=max_steps, **kwargs
        )

        # Stacking order (bottom to top)
        self.stacking_order = stacking_order or [
            "blue_block",
            "red_block",
            "yellow_block",
        ]

        # Block configurations
        self.blocks = {
            "blue_block": self.BLUE_BLOCK,
            "red_block": self.RED_BLOCK,
            "yellow_block": self.YELLOW_BLOCK,
        }

        # Stacking threshold (how close blocks need to be horizontally to count as stacked)
        self.stack_xy_threshold = 0.02  # 2cm
        self.stack_z_threshold = 0.01  # 1cm tolerance for z positioning

    def setup_env(self, env: LeNewtonEnv, **kwargs) -> None:
        """Setup the block stacking environment.

        Adds a table and three colored blocks to the environment.

        Args:
            env: The LeNewtonEnv instance with ModelBuilder available
            **kwargs: Additional parameters (e.g., randomize_positions)
        """
        super().setup_env(env, **kwargs)

        builder = env.builder
        if builder is None:
            log.error("ModelBuilder not available in env")
            return

        # Default positions spread out on the table
        base_positions = [
            (
                0.0,
                -0.292,
                0.02,
            ),
            (
                0.08,
                -0.3,
                0.02,
            ),
            (
                -0.05,
                -0.308,
                0.02,
            ),
        ]

        if env.randomize:
            # Add random offset to positions
            for i, pos in enumerate(base_positions):
                offset_x = np.random.uniform(-0.05, 0.05)
                offset_y = np.random.uniform(-0.05, 0.05)
                base_positions[i] = (pos[0] + offset_x, pos[1] + offset_y, pos[2])

        # Add blocks
        block_list = [self.BLUE_BLOCK, self.RED_BLOCK, self.YELLOW_BLOCK]
        for i, block in enumerate(block_list):
            block.initial_position = base_positions[i]
            self._add_block(builder, block)

        log.info(
            f"BlockStackTask setup complete: table and {len(block_list)} blocks added"
        )

    def _add_table(self, builder) -> None:
        """Add table geometry to the environment."""
        pos = self.table_config.position
        size = self.table_config.size
        height = self.table_config.height  # Table top position (at the top of the legs)
        table_top_pos = (pos[0], pos[1], pos[2] + height)

        # Static shape config
        static_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,  # Static (infinite mass)
            ke=1e4,
            kd=1e2,
            kf=1e3,
            mu=0.5,
        )

        # Add table as static body (body index 0 is ground typically)
        table_body = builder.add_body(
            xform=wp.transform(table_top_pos, wp.quat_identity()), key="table"
        )

        # Add table top shape
        builder.add_shape_box(
            body=table_body,
            hx=size[0] / 2,
            hy=size[1] / 2,
            hz=size[2] / 2,
            cfg=static_shape_cfg,
        )

        # Add table legs (4 corners)
        leg_radius = 0.02
        leg_positions = [
            (size[0] / 2 - leg_radius, size[1] / 2 - leg_radius),
            (size[0] / 2 - leg_radius, -size[1] / 2 + leg_radius),
            (-size[0] / 2 + leg_radius, size[1] / 2 - leg_radius),
            (-size[0] / 2 + leg_radius, -size[1] / 2 + leg_radius),
        ]

        for lx, ly in leg_positions:
            leg_body = builder.add_body(
                xform=wp.transform(
                    (pos[0] + lx, pos[1] + ly, pos[2] + height / 2), wp.quat_identity()
                ),
                key="table_leg",
            )
            builder.add_shape_box(
                body=leg_body,
                hx=leg_radius,
                hy=leg_radius,
                hz=height / 2,
                cfg=static_shape_cfg,
            )

        log.debug(f"Added table at position {table_top_pos}")

    def _add_block(self, builder, block: BlockConfig) -> None:
        """Add a colored block to the environment."""
        # Dynamic shape config for blocks
        block_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=1000.0,  # Lower density for stability
            ke=1e3,  # Lower contact stiffness to reduce vibration
            kd=1e2,  # Higher relative damping for energy dissipation
            kf=1e2,  # Lower frictional stiffness
            mu=0.8,  # Higher friction to prevent sliding
            collision_group=-1,  # Collide with everything
            collision_filter_parent=False,  # Don't filter collisions
        )

        # Create dynamic body for the block
        body_idx = builder.add_body(
            xform=wp.transform(block.initial_position, wp.quat_identity()),
            key=block.name,
        )

        # Add box shape for the block
        builder.add_shape_box(
            body=body_idx,
            hx=block.size[0] / 2,
            hy=block.size[1] / 2,
            hz=block.size[2] / 2,
            cfg=block_shape_cfg,
        )

        # Store body index for tracking
        self.object_bodies[block.name] = body_idx

        log.debug(
            f"Added {block.name} at {block.initial_position}, body_idx={body_idx}"
        )

    def get_reward(self, env: LeNewtonEnv) -> float:
        """Calculate reward based on stacking progress.

        Reward structure:
        - Partial reward for blocks being close to target position
        - Full reward (1.0) when all blocks are stacked correctly
        """
        positions = {}
        for block_name in self.stacking_order:
            pos = self.get_object_position(block_name)
            if pos is None:
                return 0.0
            positions[block_name] = pos

        reward = 0.0
        block_height = self.BLUE_BLOCK.size[2]

        # Check stacking from bottom to top
        for i, block_name in enumerate(self.stacking_order):
            if i == 0:
                # First block (bottom) - just needs to be on the table
                # Give partial reward if it's on the table surface
                if positions[block_name][2] > 0.35:  # Above some minimum height
                    reward += 0.1
                continue

            # Check if this block is stacked on the previous one
            prev_block_name = self.stacking_order[i - 1]
            prev_pos = positions[prev_block_name]
            curr_pos = positions[block_name]

            # Check horizontal alignment
            xy_dist = np.sqrt(
                (curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2
            )

            # Check vertical positioning (should be one block height above)
            expected_z = prev_pos[2] + block_height
            z_diff = abs(curr_pos[2] - expected_z)

            if xy_dist < self.stack_xy_threshold and z_diff < self.stack_z_threshold:
                # Properly stacked
                reward += 0.3
            elif xy_dist < self.stack_xy_threshold * 2:
                # Partially aligned
                reward += 0.1 * (1 - xy_dist / (self.stack_xy_threshold * 2))

        return min(reward, 1.0)

    def get_done(self, env: LeNewtonEnv) -> bool:
        """Check if all blocks are properly stacked."""
        if self.step_count >= self.max_steps:
            return True

        return self.get_reward(env) > 0.95

    def get_info(self, env: LeNewtonEnv) -> dict[str, any]:
        """Get task information including block positions."""
        info = super().get_info(env)

        # Add block positions
        block_positions = {}
        for block_name in self.blocks:
            pos = self.get_object_position(block_name)
            if pos is not None:
                block_positions[block_name] = pos.tolist()

        info["block_positions"] = block_positions
        info["stacking_order"] = self.stacking_order
        info["reward"] = self.get_reward(env)

        return info
