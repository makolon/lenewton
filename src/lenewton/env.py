from __future__ import annotations

import math
import os
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import newton
import newton.ik
import newton.viewer
import numpy as np
import warp as wp
from newton.selection import ArticulationView
from newton.sensors import SensorTiledCamera

import lenewton  # noqa: F401

if TYPE_CHECKING:
    from lenewton.tasks.lenewton_task import LeNewtonTask


# SO100 robot configuration
ASSETS_PATH = os.path.join(lenewton.LENEWTON_PATH, "assets")

# Simulation parameters
SIM_STEPS = 10
SIM_SUBSTEPS = 8  # More substeps for better stability
FPS = 50
FRAME_DT = 1.0 / FPS

# Image settings
IMAGE_SIZE = (480, 640)

# SO100 joint configuration
SO100_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
NUM_JOINTS = 6  # 5 arm joints + 1 gripper
EE_LINK_NAME = "gripper"
SO100_ARTICULATION_PATTERN = "so_arm100"

# Default joint positions
DEFAULT_JOINTS = np.array([0.0, 1.0, -1.5, 0.0, 0.0, 0.0])

# Gripper values
GRIPPER_OPEN = 2.0
GRIPPER_CLOSE = -0.2

# Default camera settings for overhead view
DEFAULT_CAMERA_POSITION = (0.0, -0.6, 0.6)  # x, y, z in world frame
DEFAULT_CAMERA_PITCH = -45.0  # degrees, looking down
DEFAULT_CAMERA_YAW = 90.0  # degrees
DEFAULT_CAMERA_FOV = 60.0  # degrees

ArrayLike = np.ndarray | tuple[float, ...]


###
# Kinematics
###


class SO100IKSolver:
    def __init__(self, model: newton.Model, ee_link_index: int):
        """Initialize IK solver.

        Args:
            model: Newton model containing the robot
            ee_link_index: Index of end-effector link (gripper)
        """
        self.model = model
        self.ee_link_index = ee_link_index
        self.n_coords = model.joint_coord_count
        self.n_problems = 1
        self.ik_iters = 24

        self.pos_obj = newton.ik.IKPositionObjective(
            link_index=ee_link_index,
            link_offset=wp.vec3(0.0, -0.03, 0.0),  # Offset to gripper tip
            target_positions=wp.zeros((self.n_problems,), dtype=wp.vec3),
            weight=1.0,
        )

        self.rot_obj = newton.ik.IKRotationObjective(
            link_index=ee_link_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.zeros((self.n_problems,), dtype=wp.vec4),
            canonicalize_quat_err=True,
            weight=1.0,
        )

        self.joint_limit_obj = newton.ik.IKJointLimitObjective(
            joint_limit_lower=model.joint_limit_lower,
            joint_limit_upper=model.joint_limit_upper,
            weight=0.1,
        )

        # Create joint_q array for IK solver
        self.joint_q = wp.zeros((self.n_problems, self.n_coords), dtype=float)
        # Initialize with current joint positions
        self.joint_q.assign(model.joint_q.numpy().reshape(1, -1))

        # Create IK solver
        self.solver = newton.ik.IKSolver(
            model=model,
            n_problems=self.n_problems,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limit_obj],
            lambda_initial=0.1,
            jacobian_mode=newton.ik.IKJacobianMode.AUTODIFF,
        )

    def solve_fk(
        self, joint_q: np.ndarray | None = None, state: newton.State | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics.

        Args:
            joint_q: Joint positions (optional, uses model's joint_q if not provided)
            state: Newton state to update (optional)

        Returns:
            tuple[np.ndarray, np.ndarray] with position and quaternion of end-effector
        """
        if state is None:
            state = self.model.state()

        if joint_q is not None:
            self.model.joint_q.assign(joint_q)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)

        # Get body transform
        body_q = state.body_q.numpy()
        ee_transform = body_q[self.ee_link_index]

        # Extract position and quaternion
        pos = ee_transform[4:7]
        quat = ee_transform[0:4]

        return pos, quat

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Solve inverse kinematics.

        Args:
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [w, x, y, z]
            q_init: Initial joint configuration (optional)

        Returns:
            Tuple of (joint positions, achieved pose)
        """
        # Set initial configuration
        if q_init is not None:
            self.joint_q.assign(q_init.reshape(1, -1))
        else:
            self.joint_q.assign(self.model.joint_q.numpy().reshape(1, -1))

        # Set targets
        self.pos_obj.set_target_position(0, wp.vec3(*target_pos))
        self.rot_obj.set_target_rotation(
            0, wp.vec4(target_quat[0], target_quat[1], target_quat[2], target_quat[3])
        )

        # Solve IK using step() method
        self.solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters)

        # Get result
        result_q = self.joint_q.numpy().flatten()

        # Compute FK to get achieved pose
        achieved_pose = self.solve_fk(result_q)

        return result_q, achieved_pose


###
# Environment
###


class LeNewtonEnv:
    """LeNewton simulation environment for SO100 robot arm."""

    def __init__(
        self,
        task: LeNewtonTask | None = None,
        task_name: str = "BlockStack",
        render_mode: str = "rgb_array",
        image_size: tuple[int, int] = IMAGE_SIZE,
        render_fps: int = 30,
        randomize: bool = True,
        use_viewer: bool = False,
        viewer_type: str = "gl",
        **kwargs,
    ):
        """Initialize LeNewton environment.

        Args:
            task: Task object defining the environment setup
            task_name: Name of the task
            render_mode: Rendering mode ('rgb_array' or 'human')
            image_size: Image size (height, width)
            seed: Random seed
            render_fps: Rendering FPS
            randomize: Whether to randomize environment
            use_viewer: Whether to use interactive viewer
            viewer_type: Type of viewer ('gl', 'usd', 'null')
        """
        self.task = task
        self.task_name = task_name
        self.render_mode = render_mode
        self.image_size = image_size
        self.render_fps = render_fps
        self.randomize = randomize
        self.use_viewer = use_viewer
        self.viewer_type = viewer_type

        # Simulation parameters
        self.fps = FPS
        self.frame_dt = FRAME_DT
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Initialize warp device
        self.device = wp.get_device()

        # Initialize viewer
        self.viewer = None
        self.render_buffer = None

        # Initialize model and state
        self.model: newton.Model | None = None
        self.state_0: newton.State | None = None
        self.state_1: newton.State | None = None
        self.control: newton.Control | None = None
        self.contacts: newton.Contacts | None = None
        self.solver: newton.solvers.SolverMuJoCo | None = None
        self.ik_solver: SO100IKSolver | None = None
        self.robot_view: ArticulationView | None = None

        # MuJoCo solver parameters
        self.use_mujoco_contacts = True
        self.solver_type = "newton"
        self.integrator = "implicitfast"
        self.iterations = 10
        self.ls_parallel = False
        self.ls_iterations = 100
        self.nconmax = 1000
        self.njmax = 2000
        self.cone = "elliptic"
        self.impratio = 1000.0

        # Camera sensor
        self.camera_sensor: SensorTiledCamera | None = None
        self.camera_rays = None
        self.camera_color_image = None
        self.camera_depth_image = None
        self.camera_fov = DEFAULT_CAMERA_FOV

        # Robot info
        self.robot_body_count = 0
        self.ee_link_index = 0
        self.body_names: list[str] = []
        self.robot_dof_count = NUM_JOINTS

        # Camera parameters placeholder (filled during build)
        self.camera_params: dict[str, Any] = {}

        # Episode counter
        self.episode_count = 0

        # Initial joint configuration
        self.initial_joint_q: np.ndarray | None = None

        # Current joint target for PD control
        self.joint_target: np.ndarray | None = None

        # Build environment
        self._build_environment()

    def _build_environment(self):
        """Build the Newton simulation environment."""
        # Create viewer based on settings
        if self.use_viewer:
            if self.viewer_type == "gl":
                self.viewer = newton.viewer.ViewerGL()
            elif self.viewer_type == "usd":
                self.viewer = newton.viewer.ViewerUSD()
            else:
                self.viewer = newton.viewer.ViewerNull()
        else:
            self.viewer = newton.viewer.ViewerNull()

        # Create model builder
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # Set default joint configuration
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            friction=1e-5,
            target_ke=200.0,
            target_kd=10.0,
        )

        # Set default shape configuration
        builder.default_shape_cfg.ke = 1.0e3  # Lower contact stiffness
        builder.default_shape_cfg.kd = 1.0e2  # Damping
        builder.default_shape_cfg.kf = 1.0e2  # Lower frictional stiffness
        builder.default_shape_cfg.mu = 0.8  # Higher friction
        builder.default_shape_cfg.collision_group = -1  # Collide with everything

        # Load SO100 robot (fall back to flat assets layout if needed)
        robot_xml_path = os.path.join(ASSETS_PATH, "so_arm100.xml")
        builder.add_mjcf(
            source=robot_xml_path,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, 0.0),
                wp.quat_identity(),
            ),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )

        # Store robot body count for later reference (before adding ground plane)
        self.robot_body_count = builder.body_count
        self.body_names = list(builder.body_key)

        # Add ground plane
        builder.add_ground_plane()

        # Find end-effector link index
        self.ee_link_index = self._find_link_index(builder, EE_LINK_NAME)

        # Set default joint positions
        if builder.joint_coord_count >= NUM_JOINTS:
            for i, val in enumerate(
                DEFAULT_JOINTS[: min(NUM_JOINTS, builder.joint_coord_count)]
            ):
                builder.joint_q[i] = val

        # Store builder reference for task setup
        self.builder = builder

        # Let task setup additional environment elements (table, objects, etc.)
        if self.task is not None:
            self.task.setup_env(self)

        # Finalize model
        self.model = builder.finalize()

        # Set gravity
        self.model.gravity = wp.array([[0.0, 0.0, -9.81]], dtype=wp.vec3)

        # Save initial joint configuration
        self.initial_joint_q = self.model.joint_q.numpy().copy()

        # Create solver with increased constraint limits to handle collisions
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver=self.solver_type,
            integrator=self.integrator,
            iterations=self.iterations,
            use_mujoco_contacts=self.use_mujoco_contacts,
            ls_parallel=self.ls_parallel,
            ls_iterations=self.ls_iterations,
            nconmax=self.nconmax,
            njmax=self.njmax,
        )

        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Copy initial joint positions to state
        self.state_0.joint_q.assign(self.initial_joint_q)
        self.state_0.joint_qd.zero_()

        # Evaluate initial FK
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )

        self.collide_substeps = False
        self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
            self.model,
            rigid_contact_max_per_pair=2**8,
            broad_phase_mode=newton.BroadPhaseMode.NXN,
        )
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # Create IK solver
        self.ik_solver = SO100IKSolver(self.model, self.ee_link_index)

        # Create ArticulationView for robot
        self.robot_view = ArticulationView(
            self.model,
            pattern=SO100_ARTICULATION_PATTERN,
        )
        self.robot_dof_count = self.robot_view.joint_coord_count

        # Set up viewer
        if self.viewer is not None:
            self.viewer.set_model(self.model)

        # Setup SensorTiledCamera for rendering
        self._setup_camera_sensor()

        # Store camera parameters for coordinate transformations
        self._update_camera_params()

    def _setup_camera_sensor(self):
        """Setup the SensorTiledCamera for rendering."""
        self.camera_sensor = SensorTiledCamera(
            model=self.model,
            num_cameras=1,
            width=self.image_size[1],  # width
            height=self.image_size[0],  # height
            options=SensorTiledCamera.Options(
                default_light=True,
                default_light_shadows=True,
                colors_per_world=True,
                colors_per_shape=False,
                checkerboard_texture=True,
            ),
        )

        # Compute camera rays for pinhole camera model
        self.camera_rays = self.camera_sensor.compute_pinhole_camera_rays(
            math.radians(self.camera_fov)
        )

        # Create output buffers for color and depth images
        self.camera_color_image = self.camera_sensor.create_color_image_output()
        self.camera_depth_image = self.camera_sensor.create_depth_image_output()

    def _update_camera_params(self):
        """Update camera intrinsic/extrinsic parameters."""
        # Compute intrinsic matrix from FOV
        fx = fy = self.image_size[1] / (
            2.0 * math.tan(math.radians(self.camera_fov / 2))
        )
        cx = self.image_size[1] / 2.0
        cy = self.image_size[0] / 2.0
        intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        # Compute extrinsic matrix from camera pose
        camera_transform = self._get_camera_transform()
        # Convert to 4x4 extrinsic matrix (world to camera)
        pos = np.array(
            [camera_transform[0][0], camera_transform[0][1], camera_transform[0][2]]
        )
        quat = camera_transform[1]  # wp.quatf
        rot = wp.quat_to_matrix(quat)
        rot_wp = wp.array([rot], dtype=wp.mat33, device="cpu")
        rot_matrix = rot_wp.numpy()[0]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rot_matrix.T  # Transpose for world-to-camera
        extrinsic[:3, 3] = -rot_matrix.T @ pos

        self.camera_params["overhead_cam"] = SimpleNamespace(
            intrinsic_matrix=intrinsic,
            extrinsic_matrix=extrinsic,
            position=pos,
            fov=self.camera_fov,
        )

    def _get_camera_transform(self) -> tuple[tuple[float, float, float], wp.quatf]:
        """Get camera transform (position and orientation).

        Returns:
            Tuple of (position, quaternion)
        """
        # Default overhead camera looking at robot workspace
        pos = DEFAULT_CAMERA_POSITION
        pitch = math.radians(DEFAULT_CAMERA_PITCH)
        yaw = math.radians(DEFAULT_CAMERA_YAW)

        # Build rotation matrix from pitch and yaw
        # Camera looks along -Z in its local frame
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)

        # Rotation matrix: first yaw around Z, then pitch around X
        rot = wp.mat33f(cy * cp, -sy, cy * sp, sy * cp, cy, sy * sp, -sp, 0.0, cp)
        quat = wp.quat_from_matrix(rot)

        return (pos, quat)

    def _find_link_index(self, builder: newton.ModelBuilder, link_name: str) -> int:
        """Find the index of a link by name.

        Args:
            builder: Model builder
            link_name: Name of the link to find

        Returns:
            Index of the link, or 0 if not found
        """
        for i, name in enumerate(builder.body_key):
            if name == link_name:
                return i
        return 0

    def reset(self, **kwargs) -> dict[str, Any]:
        """Reset the environment.

        Returns:
            Initial observation dictionary
        """
        self.sim_time = 0.0

        # Reset joint positions to initial configuration
        if self.initial_joint_q is not None:
            initial_q = self.initial_joint_q
        else:
            initial_q = DEFAULT_JOINTS

        # Reset model joint positions
        self.model.joint_q.assign(initial_q)
        self.model.joint_qd.zero_()

        # Store target joint positions for PD control
        self.joint_target = initial_q.copy()

        # Create fresh states for simulation
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Also set state_0 joint positions directly to ensure consistency
        self.state_0.joint_q.assign(initial_q)
        self.state_0.joint_qd.zero_()

        # Evaluate FK to update body positions using state's joint positions
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )

        # Reset task if available
        if self.task is not None:
            self.task.step_count = 0

        self.episode_count += 1

        return self._get_observation()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute an action in the environment.

        Args:
            action: Joint position targets for the SO100 arm (one value per actuated DOF)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Ensure action is the correct size
        action = np.asarray(action, dtype=np.float32).flatten()

        # Apply action as joint position targets using PD control
        self.robot_view.set_attribute("joint_target_pos", self.control, action)

        # Simulate using MuJoCo solver (uses generalized coordinates)
        for _ in range(self.sim_substeps):
            if self.collide_substeps:
                self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

            self.state_0.clear_forces()

            # Apply forces to the models
            self.viewer.apply_forces(self.state_0)

            # MuJoCo solver with contact handling
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Update body positions from joint positions (FK)
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
        )

        # Sync model.joint_q with state_0 for consistency with other code (e.g., IK)
        self.model.joint_q.assign(self.state_0.joint_q)
        self.model.joint_qd.assign(self.state_0.joint_qd)

        self.sim_time += self.frame_dt

        # Update task step count
        if self.task is not None:
            self.task.step_count += 1

        # Get observation
        observation = self._get_observation()

        # Get reward and done from task
        reward = 0.0
        done = False
        info: dict[str, Any] = {"sim_time": self.sim_time}

        if self.task is not None:
            reward = self.task.get_reward(self)
            done = self.task.get_done(self)
            info.update(self.task.get_info(self))

        # Update viewer
        if self.use_viewer and self.viewer is not None:
            self._update_viewer()

        return observation, reward, done, info

    def move_to_pose(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper: float,
        steps: int = SIM_STEPS,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Move the robot to a target pose using IK.

        Args:
            position: Target end-effector position [x, y, z]
            quaternion: Target end-effector quaternion [w, x, y, z]
            gripper: Gripper value (GRIPPER_OPEN to GRIPPER_CLOSE)
            steps: Number of simulation steps

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Get current joint positions
        current_q = self.model.joint_q.numpy().copy()

        # Solve IK for arm joints
        target_q, _ = self.ik_solver.solve_ik(
            position,
            quaternion,
            q_init=current_q,
        )

        # Extract robot joint targets from IK result
        target_q_robot = target_q[: self.robot_dof_count - 1]  # Exclude gripper
        # Append gripper target
        target_q_robot = np.concatenate((target_q_robot, gripper))

        observation = self._get_observation()
        reward = 0.0
        done = False
        info: dict[str, Any] = {}

        # Execute motion
        for _ in range(steps):
            observation, reward, done, info = self.step(target_q_robot)

            if done:
                break

            # Update viewer
            if self.use_viewer and self.viewer is not None:
                self._update_viewer()

        return observation, reward, done, info

    def render(self) -> np.ndarray | None:
        """Render the environment using SensorTiledCamera.

        Returns:
            RGB image array or None
        """
        if self.model is None or self.state_0 is None:
            return np.zeros((*self.image_size, 3), dtype=np.uint8)

        # Update FK for rendering
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )

        # Compute camera orientation (use default camera settings)
        camera_pos = wp.vec3f(*DEFAULT_CAMERA_POSITION)
        camera_target = wp.vec3f(0.0, 0.0, 0.0)  # Look at robot workspace

        # Compute forward direction
        forward = wp.normalize(camera_target - camera_pos)

        # Compute right and up vectors
        world_up = wp.vec3f(0.0, 0.0, 1.0)
        right = wp.normalize(wp.cross(forward, world_up))
        up = wp.cross(right, forward)

        # Create rotation matrix and convert to quaternion
        # Rotation matrix columns: right, up, -forward (camera looks along -Z)
        m00, m01, m02 = right[0], up[0], -forward[0]
        m10, m11, m12 = right[1], up[1], -forward[1]
        m20, m21, m22 = right[2], up[2], -forward[2]

        # Convert rotation matrix to quaternion
        trace = m00 + m11 + m22
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m21 - m12) * s
            y = (m02 - m20) * s
            z = (m10 - m01) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        camera_quat = wp.quatf(x, y, z, w)
        camera_transform = wp.transformf(camera_pos, camera_quat)

        # Create camera transforms array
        camera_transforms = wp.array(
            [[camera_transform]], dtype=wp.transformf, device=self.model.device
        )

        # Render using SensorTiledCamera
        self.camera_sensor.render(
            self.state_0,
            camera_transforms,
            self.camera_rays,
            self.camera_color_image,
            self.camera_depth_image,
        )
        wp.synchronize()

        # Get flattened color image
        flat_image = self.camera_sensor.flatten_color_image_to_rgba(self.camera_color_image)

        return flat_image

    def _get_observation(self) -> dict[str, Any]:
        """Get current observation.

        Returns:
            Observation dictionary
        """
        observation: dict[str, Any] = {}

        # Joint positions and velocities
        joint_q = (
            self.robot_view.get_attribute("joint_q", self.state_0)
            .numpy()
            .reshape(-1)
            .copy()
        )
        joint_qd = (
            self.robot_view.get_attribute("joint_qd", self.state_0)
            .numpy()
            .reshape(-1)
            .copy()
        )

        observation["joint_positions"] = joint_q
        observation["joint_velocities"] = joint_qd

        # End-effector pose
        ee_pos, ee_orn = self.ik_solver.solve_fk(joint_q, self.state_0)
        observation["gripper_pos"] = ee_pos
        observation["gripper_quat"] = ee_orn

        # Camera image
        image = self.render()
        observation["overhead_cam"] = image

        # Body positions for objects
        body_q = self.state_0.body_q.numpy()
        # Store body transforms
        for i in range(self.robot_body_count, self.model.body_count):
            body_name = self.body_names[i] if i < len(self.body_names) else f"body_{i}"
            observation[body_name] = {
                "position": body_q[i, 4:7].copy(),
                "quaternion": body_q[i, 0:4].copy(),
            }

        return observation

    def close(self):
        """Close the environment and clean up resources."""
        if self.viewer is not None:
            self.viewer = None

        self.model = None
        self.state_0 = None
        self.state_1 = None
        self.control = None
        self.solver = None
        self.ik_solver = None

    @property
    def action_space(self):
        """Get action space dimensions."""
        if self.robot_view is not None:
            return self.robot_dof_count
        if self.model is not None:
            return self.model.joint_dof_count
        return NUM_JOINTS

    @property
    def observation_space(self):
        """Get observation space (placeholder)."""
        return {
            "joint_positions": (self.robot_dof_count,),
            "joint_velocities": (self.robot_dof_count,),
            "gripper_pos": (3,),
            "gripper_quat": (4,),
            "overhead_cam": (*self.image_size, 3),
        }

    def get_object_positions(self) -> dict[str, np.ndarray]:
        """Get positions of all objects in the environment.

        Returns:
            Dictionary mapping object names to positions
        """
        positions: dict[str, np.ndarray] = {}

        if self.state_0 is None:
            return positions

        body_q = self.state_0.body_q.numpy()

        # Get positions of non-robot bodies
        for i in range(self.robot_body_count, self.model.body_count):
            body_name = self.body_names[i] if i < len(self.body_names) else f"body_{i}"
            positions[body_name] = body_q[i, 4:7].copy()

        return positions

    def get_object_orientations(self) -> dict[str, np.ndarray]:
        """Get orientations of all objects in the environment.

        Returns:
            Dictionary mapping object names to quaternions
        """
        orientations: dict[str, np.ndarray] = {}

        if self.state_0 is None:
            return orientations

        body_q = self.state_0.body_q.numpy()

        # Get orientations of non-robot bodies
        for i in range(self.robot_body_count, self.model.body_count):
            body_name = self.body_names[i] if i < len(self.body_names) else f"body_{i}"
            orientations[body_name] = body_q[i, 0:4].copy()

        return orientations

    def set_viewer_pose(
        self,
        pos: ArrayLike | None = None,
        pitch: float | None = None,
        yaw: float | None = None,
    ):
        """Set viewer camera parameters.

        Args:
            pos: Camera position [x, y, z]
            pitch: Camera pitch angle in degrees
            yaw: Camera yaw angle in degrees
        """
        if self.viewer is None:
            return

        # Use provided values or keep current
        if pos is not None:
            new_pos = wp.vec3(*pos)
        if pitch is not None:
            new_pitch = pitch
        if yaw is not None:
            new_yaw = yaw

        # Set camera using viewer's set_camera method
        self.viewer.set_camera(new_pos, new_pitch, new_yaw)

    def get_viewer_pose(self) -> dict[str, float]:
        """Get current viewer camera parameters.

        Returns:
            Dictionary with camera position and orientation
        """
        if self.viewer is None:
            return {}

        cam_pos, cam_pitch, cam_yaw = (
            self.viewer.camera.pos,
            self.viewer.camera.pitch,
            self.viewer.camera.yaw,
        )

        camera_info = {
            "position_x": cam_pos[0],
            "position_y": cam_pos[1],
            "position_z": cam_pos[2],
            "pitch": cam_pitch,
            "yaw": cam_yaw,
        }

        return camera_info

    def _update_viewer(self):
        """Update the viewer display."""
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)

        # Use the simulated state's joint positions for FK visualization
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()
        wp.synchronize()


def _default_task_factory(task_name: str):
    """Create a task instance from a task name."""
    try:
        from lenewton.tasks import BlockStackTask
    except Exception:
        return None

    registry = {
        "BlockStack": BlockStackTask,
    }

    task_cls = registry.get(task_name)
    if task_cls is None:
        return None
    return task_cls(task_name=task_name)


def create_lenewton_env(
    task_name: str = "BlockStack",
    task_factory: Callable[[str], LeNewtonTask | None] | None = None,
    **kwargs,
) -> LeNewtonEnv:
    """Helper to build a LeNewtonEnv from a task name."""
    factory = task_factory or _default_task_factory
    task = factory(task_name) if factory else None

    if task is None:
        print(
            f"[WARN] No task registered for '{task_name}', creating environment without a task."
        )

    return LeNewtonEnv(
        task=task,
        task_name=task_name,
        **kwargs,
    )
