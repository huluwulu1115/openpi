"""
Convert EurekaWorld Isaac-DROID simulated rollouts (collected with
`eurekaworld/scripts/rsl_rl/collect_openpi_dataset.py`) into a LeRobot dataset
that OpenPI can train on using `pi05_droid_finetune`.

This script must be run from an environment with `lerobot` installed (e.g. the
OpenPI uv environment).

Usage:
  # From the OpenPI repo root (openpi):
  uv run examples/droid/convert_eurekaworld_sim_dataset_to_lerobot.py \
    --data_dir /path/to/eurekaworld/datasets/sim_openpi_droid_lift/20260105_123456 \
    --repo_id your_hf_username/my_droid_dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = bgr[..., ::-1]
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EurekaWorld sim rollouts to LeRobot (DROID schema).")
    parser.add_argument("--data_dir", type=str, required=True, help="Collector output directory (contains meta.json).")
    parser.add_argument("--repo_id", type=str, required=True, help="Output LeRobot dataset repo_id.")
    parser.add_argument("--fps", type=int, default=None, help="Override dataset FPS (default: inferred from meta.json).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base directory where the LeRobot dataset will be written (default: <openpi repo>/lerobot_datasets).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    root_meta_path = data_dir / "meta.json"
    if not root_meta_path.exists():
        raise FileNotFoundError(f"Expected {root_meta_path} (did you pass the timestamp folder?)")

    root_meta = json.loads(root_meta_path.read_text())
    env_step_dt = float(root_meta.get("env_step_dt", 0.0))
    inferred_fps = int(round(1.0 / env_step_dt)) if env_step_dt > 0 else 15
    fps = int(args.fps) if args.fps is not None else inferred_fps

    episodes_root = data_dir / "episodes"
    episode_dirs = sorted([p for p in episodes_root.iterdir() if p.is_dir()])
    if not episode_dirs:
        raise ValueError(f"No episodes found under {episodes_root}")

    # Default output under the current OpenPI repo checkout:
    #   openpi/lerobot_datasets/<repo_id>/
    openpi_root = Path(__file__).resolve().parents[2]
    output_base = Path(args.output_dir) if args.output_dir is not None else (openpi_root / "lerobot_datasets")
    dataset_root = output_base / args.repo_id

    # Clean up any existing dataset in the output directory
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    # Create LeRobot dataset, define features to store (DROID naming conventions)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=fps,
        root=dataset_root,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    default_task = root_meta.get("prompt", "do something")
    image_ext = root_meta.get("image_format", "png")

    for ep_dir in tqdm(episode_dirs, desc="Converting episodes"):
        traj_path = ep_dir / "trajectory.npz"
        if not traj_path.exists():
            raise FileNotFoundError(traj_path)

        ep_meta = json.loads((ep_dir / "meta.json").read_text()) if (ep_dir / "meta.json").exists() else {}
        task = ep_meta.get("prompt", default_task)

        traj = np.load(traj_path)
        joint_position = traj["joint_position"].astype(np.float32)
        gripper_position = traj["gripper_position"].astype(np.float32)
        actions = traj["actions"].astype(np.float32)

        T = int(actions.shape[0])
        if joint_position.shape[0] != T or gripper_position.shape[0] != T:
            raise ValueError(
                f"Length mismatch in {traj_path}: {joint_position.shape=}, {gripper_position.shape=}, {actions.shape=}"
            )

        for t in range(T):
            ext1_path = ep_dir / "exterior_image_1_left" / f"{t:06d}.{image_ext}"
            ext2_path = ep_dir / "exterior_image_2_left" / f"{t:06d}.{image_ext}"
            wrist_path = ep_dir / "wrist_image_left" / f"{t:06d}.{image_ext}"

            dataset.add_frame(
                {
                    "exterior_image_1_left": _read_rgb(ext1_path),
                    "exterior_image_2_left": _read_rgb(ext2_path),
                    "wrist_image_left": _read_rgb(wrist_path),
                    "joint_position": joint_position[t],
                    "gripper_position": gripper_position[t],
                    "actions": actions[t],
                    # LeRobot standard: task string -> task_index internally
                    "task": task,
                }
            )
        dataset.save_episode()

    print(f"[DONE] Wrote LeRobot dataset: {args.repo_id}")
    print(f"       Location: {dataset_root}")
    print(f"       Episodes: {len(episode_dirs)}")
    print(f"       FPS: {fps}")


if __name__ == "__main__":
    main()


