#!/usr/bin/env python3
"""
Collect LeNewton demonstrations via socket teleoperation.

This example is now a thin wrapper around the reusable teleoperation utilities
in `lenewton.teleop`, keeping the example concise while the core logic lives
in the library.
"""

import argparse

from lenewton.teleop import DatasetCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect LeNewton demonstrations via socket teleoperation"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="BlockStack",
        help="LeNewton task name (default: BlockStack)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Socket host address (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=12345, help="Socket port (default: 12345)"
    )
    parser.add_argument(
        "--seed", type=int, default=12345, help="Random seed (default: 12345)"
    )
    parser.add_argument("--no-viewer", action="store_true", help="Disable viewer")
    parser.add_argument(
        "--record-hz",
        type=float,
        default=50.0,
        help="Dataset recording frequency in Hz (default: 50.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/lenewton",
        help="Output directgit ory for datasets (default: datasets/lenewton)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("LeNewton Socket Dataset Collector")
    print("=" * 50)
    print(f"Task: {args.task}")
    print(f"Socket: {args.host}:{args.port}")
    print(f"Viewer: {'Disabled' if args.no_viewer else 'Enabled'}")
    print(f"Recording frequency: {args.record_hz} Hz")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)

    collector = DatasetCollector(
        task_name=args.task,
        socket_host=args.host,
        socket_port=args.port,
        seed=args.seed,
        use_viewer=not args.no_viewer,
        record_frequency=args.record_hz,
    )

    print("\nStarting LeNewton dataset collection...")
    print("Make sure the SO100 leader socket sender is running!")
    print("\nKeyboard controls:")
    print("'a' key: Save current episode and recreate environment with new seed")
    print("'c' key: Discard current episode and recreate environment with new seed")
    print("Ctrl+C: Save current data and exit program")
    print("\nNote: Teleoperation continues even after task completion!\n")

    success = collector.run(output_dir=args.output_dir)

    if success:
        print("\nDataset collection completed successfully")
    else:
        print("\nDataset collection failed to start")


if __name__ == "__main__":
    main()
