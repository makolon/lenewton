#!/usr/bin/env python3
"""
Example script demonstrating the LeNewton environment with SO100 robot.

This script shows how to:
1. Create a BlockStackTask with colored blocks
2. Initialize the LeNewton environment
3. Control the robot using IK-based move_to_pose
4. Interact with objects in the scene
"""

import logging

import numpy as np

from lenewton.env import LeNewtonEnv
from lenewton.tasks import BlockStackTask

# Set up logging with DEBUG level to see step details
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def main():
    """Main function to run the LeNewton environment example."""
    # Create the block stacking task
    task = BlockStackTask(
        task_name="BlockStack",
        goal_str="Stack the three blocks: blue on bottom, red in middle, yellow on top",
        max_steps=500,
    )

    # Create environment with viewer enabled
    env = LeNewtonEnv(
        task=task,
        task_name="BlockStack",
        render_mode="rgb_array",
        image_size=(480, 640),
        use_viewer=True,  # Set to False to disable OpenGL viewer
        viewer_type="gl",
        record_video=True,
        render_fps=30,
    )

    log.info("Environment created successfully")

    # Reset environment
    obs = env.reset()
    log.info(f"Initial joint positions: {obs['joint_positions']}")
    log.info(f"Initial gripper position: {obs.get('gripper_pos', 'N/A')}")

    viewer_info = env.get_viewer_pose()
    print(f"Viewer info at step: {viewer_info}")

    # Move camera to a better viewpoint
    env.set_viewer_pose(
        pos=[0.0, -1.25, 0.5],
        pitch=-25.0,
        yaw=90.0,
    )
    log.info("Camera moved to new position")

    init_joint_q = obs["joint_positions"]
    print(f"init_joint_q shape: {init_joint_q.shape}, values: {init_joint_q}")

    # Run a few more steps with larger movements for testing
    log.info("Starting motion test...")
    for i in range(100):
        # Maintain current joint positions
        action = init_joint_q
        obs, reward, done, info = env.step(action)

        # Log every 10 steps
        if i % 10 == 0:
            log.info(
                f"Step {i}: joint_pos={obs['joint_positions'][:6]}, target={action[:6]}"
            )

        if done:
            log.info(f"Task completed at step {i}")
            break

    # Clean up
    env.close()
    log.info("Environment closed")


if __name__ == "__main__":
    main()