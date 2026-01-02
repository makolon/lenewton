#!/usr/bin/env python3
"""
Replay a recorded USD scene in Isaac Sim for high-quality rendering and camera capture.

Usage:
    uv run python examples/03_replay_in_isaacsim.py --usd examples/simulation.usd
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from lenewton.utils import (
    IsaacSimReplayConfig,
    IsaacSimReplayRunner,
)


def _vector_arg(values: Sequence[float]) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a USD capture in Isaac Sim and export camera data."
    )
    parser.add_argument(
        "--usd",
        type=str,
        default="examples/simulation.usd",
        help="Path to the recorded USD file (default: examples/simulation.usd)",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="RayTracedLighting",
        choices=["RayTracedLighting", "PathTracing"],
        help="Isaac Sim renderer to use (default: RayTracedLighting)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the Isaac Sim viewport",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=240,
        help="Number of frames to capture (default: 240)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Timeline frame rate for playback (default: 24.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/isaacsim_replay",
        help="Directory to save rendered data (default: outputs/isaacsim_replay)",
    )
    parser.add_argument(
        "--camera-prim",
        type=str,
        default="/World/ReplayCamera",
        help="Prim path for the replay camera (default: /World/ReplayCamera)",
    )
    parser.add_argument(
        "--camera-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, -1.2, 0.75),
        help="Camera position in world coordinates",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.25),
        help="Point for the camera to look at",
    )
    parser.add_argument(
        "--camera-fov",
        type=float,
        default=60.0,
        help="Camera field of view in degrees (default: 60.0)",
    )
    parser.add_argument(
        "--res-width",
        type=int,
        default=1280,
        help="Render width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--res-height",
        type=int,
        default=720,
        help="Render height in pixels (default: 720)",
    )
    parser.add_argument(
        "--write-depth",
        action="store_true",
        help="Also write per-pixel depth images",
    )
    parser.add_argument(
        "--write-instance-segmentation",
        action="store_true",
        help="Also write instance segmentation IDs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = IsaacSimReplayConfig(
        usd_path=args.usd,
        renderer=args.renderer,
        headless=args.headless,
        num_frames=args.frames,
        frame_rate=args.fps,
        output_dir=args.output_dir,
        camera_prim=args.camera_prim,
        camera_position=_vector_arg(args.camera_position),
        camera_target=_vector_arg(args.camera_target),
        camera_fov=args.camera_fov,
        resolution=(args.res_width, args.res_height),
        write_depth=args.write_depth,
        write_instance_segmentation=args.write_instance_segmentation,
    )

    with IsaacSimReplayRunner(config) as runner:
        camera_params = runner.run()
        metadata_path = runner.metadata_path

    print(f"Replay complete. RGB frames written under: {config.output_dir}")
    if config.write_depth:
        print("Depth: enabled")
    if config.write_instance_segmentation:
        print("Instance segmentation: enabled")
    if metadata_path is not None:
        print(f"Camera metadata saved to: {metadata_path}")
    if camera_params:
        print(f"Camera intrinsics: {camera_params['intrinsic_matrix']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
