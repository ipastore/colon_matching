import numpy as np
import cv2
import pycolmap
from scipy.spatial.transform import Rotation as R
import torch

def get_device_info(device):
    """Get device information similar to nvidia-smi."""
    if torch.cuda.is_available():
        device_info = {
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": torch.cuda.get_device_capability(device),
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_reserved": torch.cuda.memory_reserved(device),
            "memory_total": torch.cuda.get_device_properties(device).total_memory,
        }
    elif torch.backends.mps.is_available() and device == "mps":
        device_info = {
            "device_name": "MPS",
            "device_capability": None,
            "memory_allocated": None,
            "memory_reserved": None,
            "memory_total": None,
        }
    else:
        device_info = {
            "device_name": "CPU",
            "device_capability": None,
            "memory_allocated": None,
            "memory_reserved": None,
            "memory_total": None,
        }
    return device_info

def compute_AA(rot_errors_deg, trans_errors_m, thresholds_r, thresholds_t):
    """
    Compute Average Accuracy (AA) for a single scene given rotation & translation errors
    at multiple threshold pairs.

    Args:
        rot_errors_deg: [rot_err_1, rot_err_2, ...] for a single scene
        trans_errors_m: [trans_err_1, trans_err_2, ...] for a single scene
        thresholds_r: e.g. [1,2,3,...,10] (deg)
        thresholds_t: e.g. [0.2, 0.35, 0.6, 1, 2, 5]

    Returns:
        list: accuracies at each threshold pair for this scene
    """
    assert len(rot_errors_deg) == len(trans_errors_m), "Mismatch in array lengths"
    n_pairs = len(rot_errors_deg)
    if n_pairs == 0:
        return [0.0] * len(thresholds_r)

    accuracies = []
    for r_thresh, t_thresh in zip(thresholds_r, thresholds_t):
        n_accurate = 0
        for r_err, t_err in zip(rot_errors_deg, trans_errors_m):
            if (r_err < r_thresh) and (t_err < t_thresh):
                n_accurate += 1
        # fraction of pairs that pass this threshold
        acc = n_accurate / n_pairs
        accuracies.append(acc)
    
    return np.mean(accuracies)


# TODO: old. Replaced by the one with relative translation error
# def translation_error(t_colmap: np.ndarray, t_est: np.ndarray) -> float:
#     """
#     Compute translation error in meters between two translation vectors.
#     Both must be in the same coordinate scale. 
#     """
#     assert t_colmap.shape == (3,), "t_colmap must be shape (3,)"
#     assert t_est.shape == (3,),    "t_est must be shape (3,)"

#     # Scale the estimated translation to match the scale of the COLMAP translation
#     scale_factor = np.linalg.norm(t_colmap) / (1e-9 + np.linalg.norm(t_est))
#     t_est_scaled = t_est * scale_factor

#     # Compute error
#     err = np.linalg.norm(t_colmap - t_est_scaled)

#     return err

def translation_error(t_colmap: np.ndarray, t_est: np.ndarray, d: float) -> float:
    """
    Compute translation error in meters between two translation vectors.
    Both must be in the same coordinate scale. 
    """
    assert t_colmap.shape == (3,), "t_colmap must be shape (3,)"
    assert t_est.shape == (3,),    "t_est must be shape (3,)"

    # Compute error
    err = np.linalg.norm(t_colmap - t_est) / d

    return err

def rotation_error_deg(R_colmap: np.ndarray, R_est: np.ndarray) -> float:
    """
    Compute rotation error in degrees between two 3x3 rotation matrices.
    If R_est == R_colmap, result is ~0 degrees.
    """
    assert R_colmap.shape == (3, 3), "R_colmap must be 3x3"
    assert R_est.shape == (3, 3),    "R_est must be 3x3"

    # If colmap and est are identical, R_diff ~ identity => angle=0
    R_diff = R.from_matrix(R_colmap @ R_est.T)
    angle_rad = R_diff.magnitude()
    return np.degrees(angle_rad)

def adapt_mkpts_to_colmap(mkpts):
    """
    Convert keypoints to COLMAP-compatible format:
    - Ensure dtype is float64.
    - Ensure shape is (N, 2).
    - Apply (0.5, 0.5) shift to align with COLMAP's pixel convention.

    Args:
        mkpts (np.ndarray): Keypoints in image pixel coordinates (N, 2) with dtype float32.

    Returns:
        np.ndarray: Converted keypoints in COLMAP-compatible format (float64, (N, 2)).
    """
    if mkpts is None or len(mkpts) == 0:
        raise ValueError("mkpts is empty or None, cannot process.")

    # Convert to NumPy array if it's not already
    mkpts = np.asarray(mkpts, dtype=np.float64)

    # Ensure shape is (N, 2)
    if mkpts.ndim == 1 and len(mkpts) % 2 == 0:
        mkpts = mkpts.reshape(-1, 2)
    elif mkpts.ndim != 2 or mkpts.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2), but got {mkpts.shape}")

    # Apply COLMAP's (0.5, 0.5) pixel center convention
    mkpts_colmap = mkpts + 0.5
    return mkpts_colmap

def compute_relative_pose(cam_from_world1, cam_from_world2):
    """
    Compute relative pose (R, t) between two camera absolute poses.

    Args:
        cam_from_world1 (pycolmap.Rigid3d): Absolute pose of camera 1 (world -> cam1).
        cam_from_world2 (pycolmap.Rigid3d): Absolute pose of camera 2 (world -> cam2).

    Returns:
        R_rel (numpy.ndarray): 3x3 relative rotation matrix. (cam1 -> cam2)
        t_rel (numpy.ndarray): 3x1 relative translation vector. (cam1 -> cam2)
    """
    
    cam2_from_cam1 = cam_from_world1.inverse() * cam_from_world2 

    return cam2_from_cam1.rotation.matrix(), cam2_from_cam1.translation



def load_R_t(image_name, reconstruction):
    """ Load rotation matrix and translation vector for a given image from COLMAP reconstruction """
    image = reconstruction.find_image_with_name(image_name)

    # Convert COLMAP's quaternion + translation to a rotation matrix
    R = image.cam_from_world.rotation.matrix()
    t = image.cam_from_world.translation

    return R, t

import pycolmap
import numpy as np
from pathlib import Path

def compute_submap_diameter(sparse_model_dir):
    """
    Load a COLMAP Reconstruction from 'sparse_model_dir' and compute
    the diameter (largest spatial extent) based on the bounding box
    of its registered cameras / points3D.

    Returns:
        diameter (float): Euclidian distance between bounding-box corners.
    """
    # Load the reconstruction from the specified directory
    reconstruction = pycolmap.Reconstruction(sparse_model_dir)
    if len(reconstruction.images) == 0:
        raise ValueError(f"No registered images found in {sparse_model_dir}")

    # Compute bounding box (p0=0.0, p1=1.0 => full min/max box)
    bb_min, bb_max = reconstruction.compute_bounding_box(p0=0.0, p1=1.0)

    # The diameter is simply the distance between these two corners
    diameter = np.linalg.norm(bb_max - bb_min)
    return diameter


