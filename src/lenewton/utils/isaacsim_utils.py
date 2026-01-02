from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class IsaacSimReplayConfig:
    """Configuration for replaying a USD capture inside Isaac Sim."""

    usd_path: str
    renderer: str = "RayTracedLighting"  # or "PathTracing"
    headless: bool = False
    num_frames: int = 240
    frame_rate: float = 24.0
    output_dir: str = "outputs/isaacsim_replay"
    camera_prim: str = "/World/ReplayCamera"
    camera_position: tuple[float, float, float] = (0.0, -1.2, 0.75)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.25)
    camera_fov: float = 60.0  # degrees
    resolution: tuple[int, int] = (1280, 720)  # (width, height)
    write_depth: bool = False
    write_instance_segmentation: bool = False

    def resolve_paths(self) -> tuple[Path, Path]:
        usd_path = Path(self.usd_path).expanduser().resolve()
        output_dir = Path(self.output_dir).expanduser().resolve()
        return usd_path, output_dir


class IsaacSimReplayRunner:
    """Helper to load a USD recording in Isaac Sim and capture high-quality frames."""

    def __init__(self, config: IsaacSimReplayConfig):
        self.config = config
        self.usd_path, self.output_dir = self.config.resolve_paths()
        self.app = None
        self.rep = None
        self.timeline = None
        self.stage = None
        self.writer = None
        self._Usd = None
        self._UsdGeom = None
        self._Gf = None
        self.metadata_path: Path | None = None

    def __enter__(self) -> IsaacSimReplayRunner:
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _launch_app(self):
        if not self.usd_path.exists():
            raise FileNotFoundError(f"USD file not found: {self.usd_path}")

        # SimulationApp must be instantiated before other Omni imports.
        from isaacsim import SimulationApp
        self.app = SimulationApp(
            {"renderer": self.config.renderer, "headless": self.config.headless}
        )

    def _import_isaac_modules(self):
        import importlib

        omni_usd = importlib.import_module("omni.usd")
        omni_timeline = importlib.import_module("omni.timeline")
        self.rep = importlib.import_module("omni.replicator.core")
        from pxr import Gf, Usd, UsdGeom

        self.timeline = omni_timeline.get_timeline_interface()
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._Gf = Gf

        ctx = omni_usd.get_context()
        ctx.open_stage(str(self.usd_path))
        self.stage = ctx.get_stage()
        if self.stage is None:
            raise RuntimeError(f"Failed to open stage: {self.usd_path}")

    def _configure_camera(self):
        camera = self._UsdGeom.Camera.Get(self.stage, self.config.camera_prim)
        if not camera:
            camera = self._UsdGeom.Camera.Define(self.stage, self.config.camera_prim)

        width, height = self.config.resolution
        aspect_ratio = width / float(height)

        # Aperture values in millimeters (USD camera convention).
        horiz_aperture = 20.955
        vert_aperture = horiz_aperture / aspect_ratio
        focal_length = horiz_aperture / (
            2.0 * math.tan(math.radians(self.config.camera_fov) / 2.0)
        )

        camera.CreateHorizontalApertureAttr(horiz_aperture)
        camera.CreateVerticalApertureAttr(vert_aperture)
        camera.CreateFocalLengthAttr(focal_length)
        camera.CreateFocusDistanceAttr(
            max(
                0.1,
                float(
                    np.linalg.norm(
                        np.subtract(self.config.camera_target, self.config.camera_position)
                    )
                ),
            )
        )
        camera.CreateClippingRangeAttr(self._Gf.Vec2f(0.01, 1000.0))

        up = self._Gf.Vec3d(0.0, 0.0, 1.0)
        xform = self._Gf.Matrix4d().SetLookAt(
            self._Gf.Point3d(*self.config.camera_position),
            self._Gf.Point3d(*self.config.camera_target),
            up,
        )
        xformable = self._UsdGeom.Xformable(camera)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(xform)
        return camera

    def _create_render_product(self, camera):
        return self.rep.create.render_product(
            str(camera.GetPath()), resolution=self.config.resolution
        )

    def _attach_writer(self, render_product):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        writer = self.rep.WriterRegistry.get("BasicWriter")
        if writer is None:
            raise RuntimeError("Replicator BasicWriter is unavailable in this Isaac Sim build.")

        writer.initialize(
            output_dir=str(self.output_dir),
            rgb=True,
            distance_to_camera=self.config.write_depth,
            instance_segmentation=self.config.write_instance_segmentation,
            camera_params=True,
        )
        writer.attach([render_product])
        self.writer = writer

    def _run_replicator(self):
        self.stage.SetTimeCodesPerSecond(self.config.frame_rate)
        if hasattr(self.timeline, "set_end_time"):
            duration = float(self.config.num_frames) / float(self.config.frame_rate)
            self.timeline.set_end_time(duration)
        if hasattr(self.timeline, "set_current_time"):
            self.timeline.set_current_time(0.0)
        if hasattr(self.timeline, "play"):
            self.timeline.play()

        self.rep.orchestrator.run(num_frames=self.config.num_frames)
        while self.rep.orchestrator.get_is_running():
            self.app.update()

        if hasattr(self.timeline, "stop"):
            self.timeline.stop()
        if self.writer is not None and hasattr(self.writer, "flush"):
            self.writer.flush()

    def _export_camera_params(self, camera) -> dict[str, Any]:
        width, height = self.config.resolution
        horiz_aperture = camera.GetHorizontalApertureAttr().Get()
        vert_aperture = camera.GetVerticalApertureAttr().Get()
        focal_length = camera.GetFocalLengthAttr().Get()

        fx = width * focal_length / horiz_aperture
        fy = height * focal_length / vert_aperture
        cx = width / 2.0
        cy = height / 2.0

        world_from_camera = np.array(
            self._UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(
                self._Usd.TimeCode.Default()
            )
        )
        camera_from_world = np.linalg.inv(world_from_camera)

        return {
            "intrinsic_matrix": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "extrinsic_matrix": camera_from_world.tolist(),
            "world_from_camera": world_from_camera.tolist(),
            "fov": self.config.camera_fov,
            "resolution": {"width": width, "height": height},
            "renderer": self.config.renderer,
            "usd_path": str(self.usd_path),
            "camera_prim": str(camera.GetPath()),
            "camera_position": self.config.camera_position,
            "camera_target": self.config.camera_target,
        }

    def _write_metadata(self, camera_params: dict[str, Any]):
        self.metadata_path = self.output_dir / "camera_metadata.json"
        with self.metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(camera_params, fp, indent=2)

    def run(self) -> dict[str, Any]:
        self._launch_app()
        self._import_isaac_modules()

        camera = self._configure_camera()
        render_product = self._create_render_product(camera)
        self._attach_writer(render_product)
        self._run_replicator()

        camera_params = self._export_camera_params(camera)
        self._write_metadata(camera_params)
        return camera_params

    def close(self):
        if self.app is not None:
            try:
                self.app.close()
            finally:
                self.app = None


__all__ = ["IsaacSimReplayConfig", "IsaacSimReplayRunner"]
