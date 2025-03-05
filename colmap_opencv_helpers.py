import numpy as np
import cv2
import pycolmap
from scipy.spatial.transform import Rotation as R

def compute_mAA(rot_errors_deg, trans_errors_m, thresholds_r, thresholds_t):
    """
    Compute mean Average Accuracy (mAA) given rotation & translation errors
    at multiple threshold pairs.

    - rot_errors_deg: [rot_err_1, rot_err_2, ...]
    - trans_errors_m: [trans_err_1, trans_err_2, ...]
    - thresholds_r:   e.g. [1,2,3,...,10] (deg)
    - thresholds_t:   e.g. [0.2, 0.35, 0.6, 1, 2, 5] (meters)

    Return:
        float: average of the fraction of pairs that pass each threshold pair
    """
    assert len(rot_errors_deg) == len(trans_errors_m), "Mismatch in array lengths"
    n_pairs = len(rot_errors_deg)
    if n_pairs == 0:
        return 0.0

    accuracies = []
    for r_thresh, t_thresh in zip(thresholds_r, thresholds_t):
        n_accurate = 0
        for r_err, t_err in zip(rot_errors_deg, trans_errors_m):
            if (r_err < r_thresh) and (t_err < t_thresh):
                n_accurate += 1
        # fraction of pairs that pass this threshold
        acc = n_accurate / n_pairs
        accuracies.append(acc)

    # The final mAA is the average accuracy across threshold pairs
    return np.mean(accuracies)

def translation_error_m(t_colmap: np.ndarray, t_est: np.ndarray) -> float:
    """
    Compute translation error in meters between two translation vectors.
    Both must be in the same coordinate scale. 
    """
    assert t_colmap.shape == (3,), "t_colmap must be shape (3,)"
    assert t_est.shape == (3,),    "t_est must be shape (3,)"

    # Scale the estimated translation to match the scale of the COLMAP translation
    scale_factor = np.linalg.norm(t_colmap) / (1e-9 + np.linalg.norm(t_est))
    t_est_scaled = t_est * scale_factor

    # Compute error
    err = np.linalg.norm(t_colmap - t_est_scaled)

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

#TODO: Old. I am not using cv2, instead pycolmap that use all the transformations
def get_relative_pose(R0, t0, R1, t1):
    """ Compute the relative pose (R, t) from COLMAP. New reference frame is image1 """
    R_rel = R0.T @ R1  
    t_rel = R0.T @ (t1 - t0)  
    return R_rel, t_rel

#TODO: Old. I can use the image names insted of the image_id
def load_R_t_from_id(image_id, reconstruction):
    """ Load rotation matrix and translation vector for a given image from COLMAP reconstruction """
    image = reconstruction.images[image_id]

    # Convert COLMAP's quaternion + translation to a rotation matrix
    R = image.cam_from_world.rotation.matrix()
    t = image.cam_from_world.translation

    return R, t

def load_R_t(image_name, reconstruction):
    """ Load rotation matrix and translation vector for a given image from COLMAP reconstruction """
    image = reconstruction.find_image_with_name(image_name)

    # Convert COLMAP's quaternion + translation to a rotation matrix
    R = image.cam_from_world.rotation.matrix()
    t = image.cam_from_world.translation

    return R, t

#TODO: Old. I am not using cv2, instead pycolmap
def estimate_relative_pose(mkpts0, mkpts1, K):
    """
    Estimate relative pose (R, t) from matched keypoints using the essential matrix.
    :param mkpts0: Matched keypoints in image 1 (Nx2 array)
    :param mkpts1: Matched keypoints in image 2 (Nx2 array)
    :param K: Camera intrinsic matrix
    :return: Estimated rotation (R) and translation (t)
    """
    E, _ = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R_rel, t_rel, _ = cv2.recoverPose(E, mkpts0, mkpts1, K)
    return R_rel, t_rel

#TODO: Old. I am not using cv2, instead pycolmap that handles intrinsics and distortions internally
def load_camera_intrinsics(image_id, reconstruction):
    """ Load camera intrinsics from COLMAP reconstruction """
    image = reconstruction.images[image_id]
    camera = reconstruction.cameras[image.camera_id]
    K = np.array([[camera.params[0], 0, camera.params[2]],
                  [0, camera.params[1], camera.params[3]],
                  [0, 0, 1]])
    return K

#TODO: Old. is there a built-in function for this? find_image_with_name(self: pycolmap.Reconstruction, name: str)→ pycolmap.Image
def load_image_name_image_id_dict(reconstruction):
    """ Get image ID from image name """
    image_ids = {image.name: image_id for image_id, image in reconstruction.images.items()}
    return image_ids
