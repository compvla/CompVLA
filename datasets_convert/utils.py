import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from utils.Lie import LogSO3, invSO3, Adjoint, adjoint, invSE3


# ==========================================
# Robot / Sensor Constants
# ==========================================
MAX_GRIPPER_WIDTH = 0.08  # m, fully open

# Offsets along the tool z-axis (m)
TCP_OFFSET = 0.009  # 9mm, tcp length
FLANGE_UMITIP_OFFSET = 0.2766  # 276.6mm, flange to UMI tip
SENSOR_UMITIP_OFFSET = 0.2313  # 231.3mm, sensor to UMI tip
FLANGE_UMI_OFFSET = FLANGE_UMITIP_OFFSET - TCP_OFFSET  # 267.6mm
SENSOR_UMI_OFFSET = SENSOR_UMITIP_OFFSET - TCP_OFFSET  # 222.3mm
FLANGE_SENSOR_OFFSET = FLANGE_UMI_OFFSET - SENSOR_UMI_OFFSET  # 45.3mm

# Rotations about the z-axis (rad): flange -> F/T sensor, F/T sensor -> TCP
THETA_FLANGE_TO_SENSOR = -147.5 / 180 * np.pi
THETA_SENSOR_TO_TCP = 102.5 / 180 * np.pi
T_FLANGE_TO_SENSOR = np.array([
        [np.cos(THETA_FLANGE_TO_SENSOR), -np.sin(THETA_FLANGE_TO_SENSOR), 0, 0],
        [np.sin(THETA_FLANGE_TO_SENSOR),  np.cos(THETA_FLANGE_TO_SENSOR), 0, 0],
        [                             0,                               0, 1, FLANGE_SENSOR_OFFSET],
        [                             0,                               0, 0, 1]
])
T_SENSOR_TO_TCP = np.array([
        [np.cos(THETA_SENSOR_TO_TCP), -np.sin(THETA_SENSOR_TO_TCP), 0, 0],
        [np.sin(THETA_SENSOR_TO_TCP),  np.cos(THETA_SENSOR_TO_TCP), 0, 0],
        [                          0,                            0, 1, SENSOR_UMI_OFFSET],
        [                          0,                            0, 0, 1]
])

# Gripper mass properties
GRIPPER_MASS = 1.2966  # kg
GRIPPER_INERTIA_AT_COM_IN_FLANGE = np.array([
    [0.003403, 0.000145, -0.0000454],
    [0.000145, 0.004978, 0.0001296],
    [-0.0000454, 0.0001296, 0.0020217]
])  # kg m^2, about the COM, in flange-frame axes
GRIPPER_COM_IN_FLANGE = np.array([-0.00269, -0.002937, 0.08397, 1])  # m, homogeneous coordinates
GRIPPER_COM_IN_SENSOR = np.einsum('ij, j -> i', invSE3(T_FLANGE_TO_SENSOR), GRIPPER_COM_IN_FLANGE)[:3]
GRAVITY = np.array([0, 0, -9.81])  # m/s^2, world frame

MAX_FORCE_NORM = 50.0  # N
MAX_TORQUE_NORM = 10.0  # Nm

# A frame counts as moving once the TCP pose (rotation angle + translation) or the
# normalized gripper width differs from the reference frame by more than these.
MOTION_POSE_THRESHOLD = 1e-2
MOTION_GRIPPER_THRESHOLD = 1e-2

# Low-pass filter on the TCP-frame wrench. The sampling rate is the conversion fps;
# the cutoff must stay below its Nyquist (fps / 2).
WRENCH_LPF_CUTOFF = 0.8  # Hz
WRENCH_LPF_ORDER = 3

# Raw csv logs of an episode
RAW_CSV_FILES = {
    'wrench': 'ft_sensor_wrench.csv',
    'gripper': 'right_arm_gripper.csv',
    'joints': 'right_arm_joints.csv',
}


# ==========================================
# Raw Data Validation
# ==========================================

def validate_raw_episode(episode_path, cam_names):
    """Check that the raw csv logs and camera files of an episode exist. Returns (is_valid, message)."""
    for fname in RAW_CSV_FILES.values():
        fpath = os.path.join(episode_path, fname)
        if not os.path.isfile(fpath):
            return False, f"Missing file: {fname}"
        if os.path.getsize(fpath) == 0:
            return False, f"Empty file: {fname}"

    for cam in cam_names:
        video_file = os.path.join(episode_path, f"{cam}_rgb.avi")
        ts_file = os.path.join(episode_path, f"{cam}_rgb_timestamps.csv")
        if not os.path.isfile(video_file):
            return False, f"Missing video: {cam}_rgb.avi"
        if not os.path.isfile(ts_file):
            return False, f"Missing timestamps: {cam}_rgb_timestamps.csv"

    return True, "OK"


def check_nan_inf(arrays_dict):
    """Check {name: array} for NaN/Inf values. Returns (is_valid, message)."""
    for name, arr in arrays_dict.items():
        if np.isnan(arr).any():
            return False, f"NaN values found in {name}"
        if np.isinf(arr).any():
            return False, f"Inf values found in {name}"
    return True, "OK"


# ==========================================
# Episode Loading
# ==========================================

def resample_to_timeline(log, timeline, columns):
    """Linearly interpolate the given columns of a time-stamped log onto the timeline."""
    log = log.set_index('time')
    log = log[~log.index.duplicated(keep='first')]
    combined_index = log.index.union(timeline).sort_values()
    interpolated = log.reindex(combined_index)[columns].interpolate(method='index')
    resampled = interpolated.loc[timeline].reset_index()
    # The index keeps its 'time' name through the union here, but name the column explicitly in
    # case a pandas version drops it (reset_index would then call it 'index').
    resampled.rename(columns={'index': 'time'}, inplace=True)
    return resampled


def load_episode_data(episode_path, fps):
    """Load the raw csv logs of an episode and resample them onto a common timeline at fps.

    Returns a DataFrame with columns time, joint_1..7, gripper (normalized to [0, 1]) and
    Tx, Ty, Tz, Fx, Fy, Fz, or None if the logs cannot be read or do not overlap in time.
    """
    raw_logs = {}
    try:
        for name, fname in RAW_CSV_FILES.items():
            raw_logs[name] = pd.read_csv(os.path.join(episode_path, fname))
    except Exception as e:
        tqdm.write(f"Error reading files in {episode_path}: {e}")
        return None
    raw_logs['gripper']['gripper'] = raw_logs['gripper']['finger_joint1_position'] + raw_logs['gripper']['finger_joint2_position']

    # Common timeline across the episode's logs, resampled at fps
    t_start = max(log['time'].min() for log in raw_logs.values())
    t_end = min(log['time'].max() for log in raw_logs.values())
    timeline = np.arange(t_start, t_end, 1.0 / fps)
    if len(timeline) == 0:
        tqdm.write(f"No common time found for {episode_path}")
        return None

    # The joint_names column gives the order of values in positions, which may differ from the
    # canonical fr3_joint1..7 order across tasks or even rows, so joints are reordered by name.
    canonical_joints = [f'fr3_joint{i}' for i in range(1, 8)]
    joint_log = raw_logs['joints']
    pos_split = joint_log['positions'].str.split(',')
    name_split = joint_log['joint_names'].str.split(',')

    first_name_to_idx = {n: j for j, n in enumerate(name_split.iloc[0])}
    reorder_idx = [first_name_to_idx[c] for c in canonical_joints]  # canonical -> csv column index

    if (joint_log['joint_names'] == joint_log['joint_names'].iloc[0]).all():
        # Same order in every row: parse all positions at once, then reorder columns.
        # Each row holds 9 values: 7 arm joints + 2 finger joints.
        all_pos = pos_split.apply(lambda x: [float(v) for v in x[:9]])
        joint_positions = np.array(all_pos.tolist(), dtype=np.float64)[:, reorder_idx]
    else:
        joint_positions = np.zeros((len(joint_log), 7), dtype=np.float64)
        for i in range(len(joint_log)):
            pos = pos_split.iloc[i]
            name_to_idx = {n: j for j, n in enumerate(name_split.iloc[i])}
            for k, canon in enumerate(canonical_joints):
                idx = name_to_idx.get(canon)
                if idx is not None:
                    joint_positions[i, k] = float(pos[idx])

    joint_cols = [f'joint_{i+1}' for i in range(7)]
    parsed_joints = pd.DataFrame(joint_positions, columns=joint_cols)
    parsed_joints['time'] = joint_log['time'].values

    wrench_resampled = resample_to_timeline(raw_logs['wrench'], timeline, ['Tx', 'Ty', 'Tz', 'Fx', 'Fy', 'Fz'])
    gripper_resampled = resample_to_timeline(raw_logs['gripper'], timeline, ['gripper'])
    joints_resampled = resample_to_timeline(parsed_joints, timeline, joint_cols)

    episode_data = pd.merge(joints_resampled, gripper_resampled, on='time')
    episode_data = pd.merge(episode_data, wrench_resampled, on='time')

    # Normalize gripper width to [0, 1] (1.0 = MAX_GRIPPER_WIDTH), after interpolation
    episode_data['gripper'] = np.clip(episode_data['gripper'] / MAX_GRIPPER_WIDTH, 0, 1)

    return episode_data


# ==========================================
# Kinematics
# ==========================================

def load_fr3_kinematics():
    """Screw axes S (7 x 6) and home TCP pose M (4 x 4) of the FR3 arm with the UMI tool."""
    w = np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
            ])
    q = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.333],
                [0.0, 0.0, 0.0],
                [0.0825, 0.0, 0.649],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.033],
                [0.088, 0.0, 1.033]
            ])
    S = np.hstack([w, -np.cross(w, q)])
    sq = 1/np.sqrt(2)
    M_flange = np.array([
    [ sq,  sq,  0.0,  0.088],
    [ sq, -sq,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.926],
    [ 0.0,  0.0,  0.0,  1.0]])  # official fr3 flange pose (tcp orientation)

    offset = np.array([0.0, 0.0, FLANGE_UMI_OFFSET])
    M_tcp = M_flange.copy()
    M_tcp[:3, 3] += M_flange[:3, :3] @ offset

    return S, M_tcp


def compute_tcp_velocity(tcp_pose, dt):
    """Body-frame TCP twist [w, v] between consecutive poses: (N, 4, 4) -> (N-1, 6)."""
    tcp_vel = np.zeros((len(tcp_pose)-1, 6))
    tcp_vel[:, :3] = LogSO3(np.einsum('nij, njk -> nik', invSO3(tcp_pose[:-1, :3, :3]), tcp_pose[1:, :3, :3])) / dt
    tcp_vel[:, 3:] = (tcp_pose[1:, :3, 3] - tcp_pose[:-1, :3, 3]) / dt
    tcp_vel[:, 3:] = np.einsum('nij, nj -> ni', invSO3(tcp_pose[:-1, :3, :3]), tcp_vel[:, 3:])
    return tcp_vel


def find_motion_range(tcp_pose, gripper):
    """Frame range [first, last) left after dropping the stationary frames at the start and end.

    Walks in from each end until the TCP pose or the gripper width differs from that end's
    frame by more than MOTION_POSE_THRESHOLD / MOTION_GRIPPER_THRESHOLD.
    """
    first = 0
    last = len(tcp_pose)
    for i in range(len(tcp_pose)):
        pose_change = np.linalg.norm(LogSO3(invSO3(tcp_pose[0, :3, :3]) @ tcp_pose[i, :3, :3])) + \
            np.linalg.norm(tcp_pose[0, :3, 3] - tcp_pose[i, :3, 3])
        if pose_change > MOTION_POSE_THRESHOLD or abs(gripper[i] - gripper[0]) > MOTION_GRIPPER_THRESHOLD:
            break
        first += 1
    for i in range(len(tcp_pose)-1, -1, -1):
        pose_change = np.linalg.norm(LogSO3(invSO3(tcp_pose[-1, :3, :3]) @ tcp_pose[i, :3, :3])) + \
            np.linalg.norm(tcp_pose[-1, :3, 3] - tcp_pose[i, :3, 3])
        if pose_change > MOTION_POSE_THRESHOLD or abs(gripper[i] - gripper[-1]) > MOTION_GRIPPER_THRESHOLD:
            break
        last -= 1
    return first, last


# ==========================================
# Gripper Command
# ==========================================

# Gripper command thresholds, on the width normalized to [0, 1] (1.0 = MAX_GRIPPER_WIDTH).
# They are per frame, so they were tuned for fps = 10 and need retuning if fps changes.
GRIPPER_LATCH_STEP = 0.02     # 1.6 mm/frame: a real open/close, far above finger settling
GRIPPER_ONSET_STEP = 1e-3     # 0.08 mm/frame: faintest motion, only used to back-date an onset
GRIPPER_ONSET_MAX_FLAT = 2    # flat frames tolerated while walking back (10 Hz interpolation)


def derive_gripper_cmd(gripper):
    """Per-frame binary gripper command (1 = open, 0 = closed) from measured width.

    Two stages, because no single threshold does both jobs:

    1. A symmetric latch on GRIPPER_LATCH_STEP decides *whether* a transition happened.
       It ignores the sub-millimetre rebound the fingers make as a close settles --
       which otherwise reads as an "open" -- and being symmetric it can always latch
       back the other way instead of sticking for the rest of the episode.
    2. Each transition is then walked back to the first frame of that same motion
       using the much finer GRIPPER_ONSET_STEP, so the command still lands at the
       onset rather than once the motion is well underway.

    Args:
        gripper: (N,) width normalized to [0, 1].

    Returns:
        (N-1,) float array of 0.0/1.0, aligned with the next-step action fields.
    """
    width_step = np.diff(gripper)
    cmd = np.zeros_like(width_step)

    # Stage 1: robust latch -> trustworthy list of transitions
    current_cmd = 1.0 if gripper[0] > 0.5 else 0.0
    transitions = []  # (frame, direction): -1 = close, +1 = open
    for frame in range(len(width_step)):
        if width_step[frame] < -GRIPPER_LATCH_STEP and current_cmd != 0.0:
            current_cmd = 0.0
            transitions.append((frame, -1))
        elif width_step[frame] > GRIPPER_LATCH_STEP and current_cmd != 1.0:
            current_cmd = 1.0
            transitions.append((frame, +1))
        cmd[frame] = current_cmd

    # Stage 2: back-date each transition to the true onset of the motion
    earliest_onset = 0
    for transition_frame, direction in transitions:
        onset = transition_frame
        cursor = transition_frame
        flat_count = 0
        while cursor - 1 >= earliest_onset:
            prev_step = width_step[cursor - 1]
            moving = prev_step < -GRIPPER_ONSET_STEP if direction < 0 else prev_step > GRIPPER_ONSET_STEP
            if moving:
                cursor -= 1
                onset = cursor
                flat_count = 0
            elif abs(prev_step) <= GRIPPER_ONSET_STEP and flat_count < GRIPPER_ONSET_MAX_FLAT:
                cursor -= 1      # interpolation-flat frame; keep looking past it
                flat_count += 1
            else:
                break
        cmd[onset:transition_frame] = 0.0 if direction < 0 else 1.0
        earliest_onset = onset + 1

    return cmd


# ==========================================
# Wrench Processing
# ==========================================

def compensate_gravity(wrench, tcp_pose):
    """Remove the sensor bias (mean over the initial stationary frames) and the gripper's weight."""
    bias = []
    for i in range(len(tcp_pose)):
        if np.linalg.norm(LogSO3(invSO3(tcp_pose[0, :3, :3]) @ tcp_pose[i, :3, :3])) + \
            np.linalg.norm(tcp_pose[0, :3, 3] - tcp_pose[i, :3, 3]) > MOTION_POSE_THRESHOLD:
            break
        bias.append(wrench[i])
    bias = np.mean(bias, axis=0)

    R_sensor = np.einsum('nij, jk -> nik', tcp_pose[:, :3, :3], invSO3(T_SENSOR_TO_TCP[:3, :3]))
    gravity_in_sensor = np.einsum('nij, j -> ni', invSO3(R_sensor), GRAVITY)
    force_g = GRIPPER_MASS * gravity_in_sensor
    torque_g = np.cross(GRIPPER_COM_IN_SENSOR, force_g)

    wrench[:, :3] = wrench[:, :3] - torque_g + torque_g[0] - bias[:3]
    wrench[:, 3:] = wrench[:, 3:] - force_g + force_g[0] - bias[3:]
    return wrench


def differentiate(X, dt):
    if X.shape == (len(X), 4, 4):
        vel = np.zeros((len(X), 6))
        for i in range(1, len(X)-1):
            vel[i, :3] = LogSO3(np.einsum('ij, jk -> ik', invSO3(X[i, :3, :3]), X[i+1, :3, :3])) / dt
            vel[i, 3:] = (X[i+1, :3, 3] - X[i-1, :3, 3]) / (2*dt)  # central difference
        vel[0, :3] = LogSO3(np.einsum('ij, jk -> ik', invSO3(X[0, :3, :3]), X[1, :3, :3])) / dt
        vel[0, 3:] = (X[1, :3, 3] - X[0, :3, 3]) / dt
        vel[-1, :3] = LogSO3(np.einsum('ij, jk -> ik', invSO3(X[-2, :3, :3]), X[-1, :3, :3])) / dt
        vel[-1, 3:] = (X[-1, :3, 3] - X[-2, :3, 3]) / dt
    elif X.shape == (len(X), 6):
        vel = np.zeros_like(X)
        for i in range(1, len(X)-1):
            vel[i] = (X[i+1] - X[i-1]) / (2*dt)
        vel[0] = (X[1] - X[0]) / dt
        vel[-1] = (X[-1] - X[-2]) / dt
    else:
        raise ValueError("Input X must be of shape (n, 4, 4) or (n, 6)")
    return vel


def compensate_inertia(wrench, tcp_pose, dt):
    """Remove the gripper's inertial wrench, estimated from the sensor-frame motion."""
    # dt must match the rate wrench/tcp_pose are sampled at: it feeds two nested derivatives
    # below, so an off-by-k dt scales the whole compensation by k^2.
    T_s = np.einsum('nij, jk -> nik', tcp_pose, invSE3(T_SENSOR_TO_TCP))  # (n, 4, 4)
    V_s = differentiate(T_s, dt)  # (n, 6)
    V_s_dot = differentiate(V_s, dt)  # (n, 6)

    G_c = np.zeros((6, 6))
    G_c[0:3, 0:3] = GRIPPER_INERTIA_AT_COM_IN_FLANGE
    G_c[3:6, 3:6] = np.eye(3) * GRIPPER_MASS
    T_s2c = np.eye(4)
    T_s2c[:3, :3] = invSO3(T_FLANGE_TO_SENSOR[:3, :3])
    T_s2c[:3, 3] = GRIPPER_COM_IN_SENSOR
    G_s = np.einsum('ij, jk, kl -> il', Adjoint(invSE3(T_s2c)).T, G_c, Adjoint(invSE3(T_s2c)))  # (6, 6)

    inertial_wrench = np.einsum('ij, nj -> ni', G_s, V_s_dot) - np.einsum('nij, jk, nk -> ni', adjoint(V_s).transpose(0, 2, 1), G_s, V_s)
    wrench += inertial_wrench
    return wrench


def butterworth_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def postprocess_wrench(wrench, tcp_pose, fps):
    """Turn the raw F/T reading into the TCP-frame wrench used for stiffness and virtual target.

    gravity compensation -> inertia compensation -> sensor to tcp frame -> low-pass filter
    -> norm clipping -> sign inversion. fps is the rate wrench/tcp_pose were resampled to by
    load_episode_data, and is deliberately not defaulted: both the derivative step and the
    filter's Nyquist depend on it.
    """
    wrench = compensate_gravity(wrench.copy(), tcp_pose)  # the compensations modify wrench in place
    wrench = compensate_inertia(wrench, tcp_pose, dt=1.0 / fps)
    wrench_tcp = np.einsum('ij, nj -> ni', Adjoint(T_SENSOR_TO_TCP).T, wrench)
    for i in range(6):
        wrench_tcp[:, i] = butterworth_lowpass_filter(wrench_tcp[:, i], cutoff=WRENCH_LPF_CUTOFF, fs=fps, order=WRENCH_LPF_ORDER)
    # force/torque clipping
    torque_norm = np.linalg.norm(wrench_tcp[:, 0:3], axis=1)
    force_norm = np.linalg.norm(wrench_tcp[:, 3:6], axis=1)
    over_torque = torque_norm > MAX_TORQUE_NORM
    over_force = force_norm > MAX_FORCE_NORM
    wrench_tcp[over_torque, 0:3] = wrench_tcp[over_torque, 0:3] / torque_norm[over_torque, None] * MAX_TORQUE_NORM
    wrench_tcp[over_force, 3:6] = wrench_tcp[over_force, 3:6] / force_norm[over_force, None] * MAX_FORCE_NORM
    wrench_tcp = -wrench_tcp  # Invert for compliance control
    return wrench_tcp


# ==========================================
# Stiffness & Virtual Target
# ==========================================

# (K_max, K_min, f_min, f_max) per data type
K_PARAMS = {
    "force": (500, 200, 1, 15),
    "torque": (30, 10, 0.1, 2.5),
}


def compute_diagonal_stiffness(vec, data_type):
    """Diagonal stiffness (N x 3) from torque or force vectors (N x 3). Returns (K, f_min).

    The first axis (along the force/torque direction) decreases linearly from K_max to K_min as
    the magnitude rises from f_min to f_max; the other two axes stay at K_max.
    """
    assert data_type in ["force", "torque"], "type must be 'force' or 'torque'"

    K_max, K_min, f_min, f_max = K_PARAMS[data_type]

    magnitude = np.linalg.norm(vec, axis=1)
    mid = (magnitude >= f_min) & (magnitude <= f_max)
    high = magnitude > f_max
    K = np.ones((len(vec), 3)) * K_max
    K[mid, 0] = K_max - (K_max - K_min) * (magnitude[mid] - f_min) / (f_max - f_min)
    K[high, 0] = K_min
    return K, f_min


def compute_stiffness_and_vt_disp(wrench):
    """Stiffness K (N x 6) and virtual target displacement vt_disp (N x 6) from the TCP-frame wrench.

    Both are rotation first. vt_disp is the torque/force divided by the first-axis stiffness, and
    is set to zero where the torque/force magnitude is below f_min.
    """
    torque = wrench[:, 0:3]
    force = wrench[:, 3:6]
    K_R, t_min = compute_diagonal_stiffness(torque, data_type="torque")  # Nx3
    K_p, f_min = compute_diagonal_stiffness(force, data_type="force")  # Nx3
    K = np.concatenate([K_R, K_p], axis=1)  # Nx6

    vt_disp = np.zeros((len(wrench), 6))  # Nx6
    vt_disp[:, 0:3] = torque / K_R[:, 0][:, None]
    vt_disp[:, 3:6] = force / K_p[:, 0][:, None]

    # set to zero if force/torque is too small
    idx_small_torque = np.linalg.norm(torque, axis=1) < t_min
    idx_small_force = np.linalg.norm(force, axis=1) < f_min
    vt_disp[idx_small_torque, 0:3] = 0
    vt_disp[idx_small_force, 3:6] = 0
    return K, vt_disp


# ==========================================
# Video Processing
# ==========================================

def read_video(episode_path, timeline, cam_name):
    """Read a raw camera video and pick, for each time in the timeline, the nearest frame."""
    video_path = os.path.join(episode_path, f"{cam_name}_rgb.avi")
    cam_timestamps = pd.read_csv(os.path.join(episode_path, f"{cam_name}_rgb_timestamps.csv"))
    cam_time = (cam_timestamps['sec'] + cam_timestamps['nanosec']*1e-9).values
    frame_idx = np.abs(cam_time[None, :] - timeline[:, None]).argmin(axis=1)  # nearest frame per time

    raw_frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    if len(raw_frames) < len(cam_time):  # pad with black frames at the beginning
        tqdm.write(f"Warning: Video frames less than timestamps in {video_path}")
        for _ in range(len(cam_time) - len(raw_frames)):
            raw_frames.insert(0, np.zeros_like(raw_frames[0]))
    elif len(raw_frames) > len(cam_time):  # trim extra frames at the end
        tqdm.write(f"Warning: Video frames more than timestamps in {video_path}")
        raw_frames = raw_frames[:len(cam_time)]

    # raw_frames now has one frame per timestamp, so every index in frame_idx is valid
    return [raw_frames[idx] for idx in frame_idx]


def save_video(save_path, frames, fps):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)  # BGR format
    writer.release()


def process_videos(episode_path, timeline, camera_keys, video_dir, video_name, fps):
    video_stats = {}
    with ThreadPoolExecutor(max_workers=len(camera_keys)) as executor:
        futures = []
        for cam_name, video_key in camera_keys.items():
            futures.append(executor.submit(process_video, episode_path, timeline, cam_name, video_key, video_dir, video_name, fps))
        for future in futures:
            video_key, stats = future.result()
            video_stats[video_key] = stats
    return video_stats


def process_video(episode_path, timeline, cam_name, video_key, video_dir, video_name, fps):
    frames = read_video(episode_path, timeline, cam_name)
    os.makedirs(os.path.join(video_dir, video_key), exist_ok=True)
    save_video(os.path.join(video_dir, video_key, video_name), frames, fps=fps)
    stats = compute_stats(frames, "image")
    return video_key, stats


# ==========================================
# Dataset Output
# ==========================================

def create_dataset_dirs(save_path):
    """Create the data, videos and meta directories of a dataset. Returns their paths."""
    data_dir = os.path.join(save_path, 'data', 'chunk-000')
    video_dir = os.path.join(save_path, 'videos', 'chunk-000')
    meta_dir = os.path.join(save_path, 'meta')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    return data_dir, video_dir, meta_dir


def compute_stats(data, data_type="scalar"):
    """Per-episode min/max/mean/std/count in LeRobot format (images: per RGB channel, in [0, 1])."""
    stats = {}
    data = np.array(data)
    stats['count'] = [len(data)]
    if data_type == "scalar":
        stats['min'] = np.min(data, axis=0)
        stats['max'] = np.max(data, axis=0)
        stats['mean'] = np.mean(data, axis=0)
        stats['std'] = np.std(data, axis=0, ddof=0)
    elif data_type == "image":
        normalized_data = data / 255.0
        stats['min'] = np.min(normalized_data, axis=(0,1,2))[::-1].reshape(3, 1, 1)  # BGR to RGB
        stats['max'] = np.max(normalized_data, axis=(0,1,2))[::-1].reshape(3, 1, 1)
        stats['mean'] = np.mean(normalized_data, axis=(0,1,2))[::-1].reshape(3, 1, 1)
        stats['std'] = np.std(normalized_data, axis=(0,1,2), ddof=0)[::-1].reshape(3, 1, 1)
    else:
        raise ValueError("data_type must be 'scalar' or 'image'")

    # convert numpy types to native python types
    for key in stats.keys():
        if isinstance(stats[key], np.ndarray):
            stats[key] = stats[key].tolist()
        elif isinstance(stats[key], np.generic):
            stats[key] = stats[key].item()

    # make all stats values as list
    for key in stats.keys():
        if not isinstance(stats[key], list):
            stats[key] = [stats[key]]

    return stats
