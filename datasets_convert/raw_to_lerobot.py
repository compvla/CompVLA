"""Convert raw episodes of one task into a LeRobot v2.1 dataset.

Run from the repository root (config and data paths are relative to it):

    python -m datasets_convert.raw_to_lerobot --task Erase

Reads raw episodes from datasets/raw_data/{task}/ and writes the dataset to
datasets/lerobot/{task}/ (paths set in datasets_convert/raw_to_lerobot.yaml).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from utils.Lie import ForwardKinematics
from datasets_convert.utils import (
    create_dataset_dirs,
    load_episode_data, load_fr3_kinematics, compute_tcp_velocity, find_motion_range,
    postprocess_wrench, compute_stiffness_and_vt_disp,
    process_videos, compute_stats,
    validate_raw_episode, check_nan_inf, derive_gripper_cmd
)


if __name__ == '__main__':
    # ==========================================
    # Load Configuration
    # ==========================================

    # Load task info from tasks.json
    with open('datasets_convert/tasks.json', 'r', encoding='utf-8') as f:
        tasks_dict = json.load(f)
    task_list = list(tasks_dict.keys())
    task_to_idx = {task_name: task_info["index"] for task_name, task_info in tasks_dict.items()}
    task_to_description = {task_name: task_info["description"] for task_name, task_info in tasks_dict.items()}

    # Load state/action feature definitions
    with open('datasets_convert/features.yaml', 'r') as f:
        state_action_features = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=task_list, required=True)
    args = parser.parse_args()

    # Load conversion config
    with open('datasets_convert/raw_to_lerobot.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    task = args.task
    raw_dataset_path = cfg['raw_dataset_path']
    output_dataset_path = cfg['output_dataset_path']
    num_episodes = cfg['num_episodes']
    fps = cfg['fps']
    dt = 1.0 / fps

    # Raw camera name -> LeRobot video key
    camera_keys = {
        'cam0': 'observation.images.end_effector',
        'cam1': 'observation.images.front',
        'cam2': 'observation.images.front_right'
    }

    # ==========================================
    # Create Directories
    # ==========================================
    # Output layout: datasets/lerobot/{task}/data|videos|meta
    output_path = os.path.join(output_dataset_path, task)
    if os.path.exists(output_path):
        raise FileExistsError(f"Output folder '{output_path}' already exists. Remove it before converting again.")
    data_dir, video_dir, meta_dir = create_dataset_dirs(output_path)

    # ==========================================
    # Initialize Tracking
    # ==========================================
    # Per-episode lines of episodes.jsonl / episodes_stats.jsonl, filled as each episode is saved
    # and written to file after the loop
    episodes = []
    episodes_stats = []
    total_frames = 0
    task_idx = task_to_idx[task]
    task_description = task_to_description[task]
    skipped_episodes = []
    fr3_kinematics = load_fr3_kinematics()

    raw_task_path = os.path.join(raw_dataset_path, task)
    if not os.path.isdir(raw_task_path):
        raise FileNotFoundError(f"No raw data folder found for task '{task}' in '{raw_dataset_path}'.")

    episode_names = sorted(os.listdir(raw_task_path))
    if num_episodes > len(episode_names):
        raise ValueError(
            f"Requested {num_episodes} episodes but only {len(episode_names)} episodes found in {raw_task_path}"
        )

    # ==========================================
    # Main Processing Loop
    # ==========================================
    for raw_idx in tqdm(range(num_episodes), desc=f"Processing task: {task}"):
        episode_path = os.path.join(raw_task_path, episode_names[raw_idx])
        episode_label = f"{raw_idx} ({episode_names[raw_idx]})"

        is_valid, msg = validate_raw_episode(episode_path, camera_keys)
        if not is_valid:
            tqdm.write(f"Skipping episode {episode_label}: {msg}")
            skipped_episodes.append(episode_label)
            continue

        episode_data = load_episode_data(episode_path, fps)
        if episode_data is None:
            tqdm.write(f"Skipping episode {episode_label}: Failed to load episode data")
            skipped_episodes.append(episode_label)
            continue

        # ==========================================
        # Process Data
        # ==========================================
        timeline = episode_data['time'].values  # common timeline resampled at fps
        joints = episode_data[[f'joint_{i+1}' for i in range(7)]].values
        gripper = episode_data['gripper'].values  # normalized to [0, 1]
        wrench = episode_data[['Tx', 'Ty', 'Tz', 'Fx', 'Fy', 'Fz']].values
        tcp_pose = ForwardKinematics(joints, *fr3_kinematics)

        # NaN/Inf validation
        is_valid, msg = check_nan_inf({
            'joints': joints, 'gripper': gripper, 'wrench': wrench, 'tcp_pose': tcp_pose
        })
        if not is_valid:
            tqdm.write(f"Skipping episode {episode_label}: {msg}")
            skipped_episodes.append(episode_label)
            continue

        # Gripper
        gripper_state = gripper[:-1].reshape(-1, 1)
        gripper_cmd = derive_gripper_cmd(gripper).reshape(-1, 1)

        tcp_vel = compute_tcp_velocity(tcp_pose, dt)

        tcp_pose = tcp_pose[:-1]       # current-step poses, N frames
        tcp_wrench = postprocess_wrench(wrench[:-1], tcp_pose, fps)
        K, vt_disp = compute_stiffness_and_vt_disp(tcp_wrench)

        joint_action = joints[1:]
        timestamps = timeline[:-1]
        video_time = timeline[:-1]

        # Eliminate stationary frames (start and end only)
        first_motion_idx, last_motion_idx = find_motion_range(tcp_pose, gripper_state[:, 0])
        if first_motion_idx >= last_motion_idx:
            tqdm.write(f"Skipping episode {episode_label}: No motion detected")
            skipped_episodes.append(episode_label)
            continue

        motion_slice = slice(first_motion_idx, last_motion_idx)
        tcp_pose = tcp_pose[motion_slice]
        tcp_vel = tcp_vel[motion_slice]
        gripper_state = gripper_state[motion_slice]
        gripper_cmd = gripper_cmd[motion_slice]
        K = K[motion_slice]
        vt_disp = vt_disp[motion_slice]
        joint_action = joint_action[motion_slice]
        timestamps = timestamps[motion_slice]
        timestamps = timestamps - timestamps[0]
        video_time = video_time[motion_slice]

        # Post-trim NaN/Inf check (velocity/stiffness computations may introduce issues)
        is_valid, msg = check_nan_inf({
            'tcp_vel': tcp_vel, 'K': K, 'vt_disp': vt_disp
        })
        if not is_valid:
            tqdm.write(f"Skipping episode {episode_label}: {msg} after trimming")
            skipped_episodes.append(episode_label)
            continue

        # Index of the saved episode (skipped raw episodes are not counted)
        episode_idx = len(episodes)

        # ==========================================
        # Build State/Action Arrays
        # ==========================================
        state = np.concatenate([
            tcp_pose[:, :3, 0], tcp_pose[:, :3, 1], tcp_pose[:, :3, 2],
            tcp_pose[:, :3, 3], gripper_state
        ], axis=1)
        action = np.concatenate([tcp_vel, gripper_cmd, K, vt_disp, joint_action], axis=1)

        # ==========================================
        # Process Videos
        # ==========================================
        video_name = f'episode_{episode_idx:06d}.mp4'
        video_stats = process_videos(episode_path, video_time, camera_keys, video_dir, video_name, fps)

        # ==========================================
        # Compute Stats & Save
        # ==========================================
        stats = dict(video_stats)
        stats['action'] = compute_stats(action)
        stats['observation.state'] = compute_stats(state)
        stats['timestamp'] = compute_stats(timestamps)
        stats['frame_index'] = compute_stats(np.arange(len(state)))
        stats['episode_index'] = compute_stats(np.array([episode_idx] * len(state)))
        stats['index'] = compute_stats(np.arange(total_frames, total_frames + len(state)))
        stats['task_index'] = compute_stats(np.array([task_idx] * len(state)))

        parquet_table = pd.DataFrame({
            "observation.state": [row.tolist() for row in state],
            "action": [row.tolist() for row in action],
            "timestamp": timestamps.astype(float).tolist(),
            "episode_index": [episode_idx] * len(state),
            "frame_index": np.arange(len(state)).tolist(),
            "index": np.arange(total_frames, total_frames + len(state)).tolist(),
            "task_index": [task_idx] * len(state),
            "next.done": ([False] * (len(state) - 1)) + [True],
        })

        parquet_path = os.path.join(data_dir, f'episode_{episode_idx:06d}.parquet')
        parquet_table.to_parquet(parquet_path, index=False)

        episodes_stats.append({
            'episode_index': episode_idx,
            'stats': stats
        })
        episodes.append({
            'episode_index': episode_idx,
            'tasks': [task_description],
            'length': len(state)
        })
        total_frames += len(state)

    # ==========================================
    # Metadata Generation
    # ==========================================
    num_saved = len(episodes)

    tasks = []
    for name, description in task_to_description.items():
        tasks.append({
            'task_index': task_to_idx[name],
            'task': description
        })

    # Camera features
    cam_features = {}
    for cam_key in camera_keys.values():
        cam_features[cam_key] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": fps,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": False
            }
        }

    # 1. episodes_stats.jsonl
    with open(os.path.join(meta_dir, 'episodes_stats.jsonl'), 'w', encoding='utf-8') as f:
        for r in episodes_stats:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 2. episodes.jsonl
    with open(os.path.join(meta_dir, 'episodes.jsonl'), 'w', encoding='utf-8') as f:
        for r in episodes:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 3. tasks.jsonl
    with open(os.path.join(meta_dir, 'tasks.jsonl'), 'w', encoding='utf-8') as f:
        for r in tasks:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 4. info.json (dataset-level totals, needs all episodes processed)
    features = {
        "observation.state": state_action_features['observation.state'],
        "action": state_action_features['action'],
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    features.update(cam_features)

    info_data = {
        "codebase_version": "v2.1",
        "robot_type": "franka_research_3",
        "fps": fps,
        "total_episodes": num_saved,
        "total_frames": total_frames,
        "total_tasks": len(task_to_idx),
        "total_videos": num_saved * len(camera_keys),
        "total_chunks": 1,
        "chunks_size": 1000,
        "camera_keys": list(camera_keys.values()),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "splits": {"train": "0:" + str(num_saved)},
        "features": features
    }

    with open(os.path.join(meta_dir, 'info.json'), 'w') as f:
        json.dump(info_data, f, indent=2, ensure_ascii=False, sort_keys=True)

    print("\nConversion completed.")
    print(f"  Task: {task}")
    print(f"  Saved: {num_saved} / {num_episodes} episodes")
    if skipped_episodes:
        print(f"  Skipped: {', '.join(skipped_episodes)}")
    print(f"  Output: {output_path}")
