import numpy as np
import cv2
import pycolmap
from scipy.spatial.transform import Rotation as R
import torch
from match_image_pairs import match_image_pairs
from my_logging import debug_log
import gc
from memory_profiler import profile

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

# #TODO colon_matching: option1 OLD
# def translation_error_realative_to_t_colmap(t_colmap: np.ndarray, t_est: np.ndarray) -> float:
#     """
#     Compute translation normalized relative error between two translation vectors.

#     """
#     assert t_colmap.shape == (3,), "t_colmap must be shape (3,)"
#     assert t_est.shape == (3,),    "t_est must be shape (3,)"

#     # Scale the estimated translation to match the scale of the COLMAP translation
#     scale_factor = np.linalg.norm(t_colmap) / np.linalg.norm(t_est)
#     t_est_scaled = t_est * scale_factor

#     # Compute the normalized error
#     err = np.linalg.norm(t_colmap - t_est_scaled) / np.linalg.norm(t_colmap)

#     return err

# #TODO colon_matching: option2 NOT USED
# def translation_error_unit(t_colmap: np.ndarray, t_est: np.ndarray) -> float:
#     """
#     Mide la diferencia entre las direcciones de t_colmap y t_est,
#     ambos normalizados a vectores unitarios. Devuelve la norma
#     de la diferencia, es decir, en [0, 2].
#     """
#     assert t_colmap.shape == (3,), "t_colmap must be shape (3,)"
#     assert t_est.shape == (3,),    "t_est must be shape (3,)"

#     norm_colmap = np.linalg.norm(t_colmap) 
#     norm_est    = np.linalg.norm(t_est)   

#     t_colmap_unit = t_colmap / norm_colmap
#     t_est_unit    = t_est    / norm_est

#     # Error = || (unit_colmap - unit_est) ||
#     # Va de 0 (idénticos) a 2 (opuestos en dirección)
#     err = np.linalg.norm(t_colmap_unit - t_est_unit)
#     return err

#TODO colon_matching: option3
def translation_error_direction_deg(t_colmap: np.ndarray, t_est: np.ndarray) -> float:

    assert t_colmap.shape == (3,), "t_colmap must be shape (3,)"
    assert t_est.shape == (3,),    "t_est must be shape (3,)"
    
    # Normalize the translation vectors
    t_colmap_unit = t_colmap / np.linalg.norm(t_colmap)
    t_est_unit    = t_est    / np.linalg.norm(t_est)

    # Compute the angle between the two normalized vectors
    cos_angle = np.dot(t_colmap_unit, t_est_unit)

    # Clip the value to avoid numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Compute the angle in radians
    err = np.arccos(cos_angle)  

    return np.degrees(err)  

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


def load_covisibility_graph(file_path):
    """
    Loads a covisibility graph from a text file.
    The covisibility graph is represented as a dictionary where each key is a reference image
    and its value is a list of covisible images. The input file should have each line formatted
    as follows:
        reference_image covisible_image_1 covisible_image_2 ...
    If a reference image has no covisible images, the line should only contain the reference image.
    Args:
        file_path (str): The path to the text file containing the covisibility graph.
    Returns:
        dict: A dictionary representing the covisibility graph. Keys are reference image names
              (str), and values are lists of covisible image names (list of str).
    """

    covisibility_graph = {}

    with open(file_path, 'r') as f:
        for line in f: 
            line = line.strip().split()
            if not line:
                continue
    
            reference_image = line[0]
            covisible_images = line[1:] if len(line) > 1 else []
            covisibility_graph[reference_image] = covisible_images
    
    return covisibility_graph



def compute_parallax(P, C1, C2):
    """
    Compute the parallax angle (in radians) between two cameras and a 3D point.

    Args:
        P (np.ndarray): 3D point (shape: [3])
        C1 (np.ndarray): Center of camera 1 (shape: [3])
        C2 (np.ndarray): Center of camera 2 (shape: [3])

    Returns:
        float: Parallax angle in radians
    """
    v1 = (P - C1)
    v1 /= np.linalg.norm(v1)

    v2 = (P - C2)
    v2 /= np.linalg.norm(v2)

    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    theta = np.arccos(cos_theta)

    return np.degrees(theta)

def filter_3D_points(reconstruction, min_track_len, max_reproj_error):
    valid_points = set()
    for p3d_id, p3d in reconstruction.points3D.items():
        if p3d.track.length() >= min_track_len and p3d.error <= max_reproj_error:
            valid_points.add(p3d_id)
    return valid_points
    
def filter_image_pairs(
    images,
    reconstruction,
    covisibility_graph,
    min_parallax=0.0,
    covisibility_threshold=0.0,
    min_track_len=3,
    max_reproj_error=2.0,
    min_shared_points=15,
    logger=None
):
    """
    Filter image pairs based on sequential order and covisibility criteria.
    
    Args:
        images (list): List of image paths or image names
        reconstruction (pycolmap.Reconstruction): COLMAP reconstruction object
        covisibility_graph (dict): Covisibility graph from load_covisibility_graph()
        covisibility_threshold (float): Minimum 3D covisibility score to include a pair
        min_track_len (int): Minimum track length for 3D covisibility computation
        max_reproj_error (float): Maximum reprojection error for 3D covisibility computation
        min_shared_points (int): Minimum number of 3D points shared between images
        logger: Optional logger object for info and warnings
        
    Returns:
        list: List of dicts, each with keys: img0, img1, shared_points, covis_score, median_parallax
    """

    final_pairs = []
    valid_points = filter_3D_points(reconstruction, min_track_len, max_reproj_error)
    
    i = 0
    # Greedy forward matching of pairs
    while i < len(images):
        img0 = images[i]
        img0_name = img0.name if hasattr(img0, 'name') else img0
        match_found = False

        for j in range(i + 1, len(images)):
            img1 = images[j]
            img1_name = img1.name if hasattr(img1, 'name') else img1

            # Filter1: Check if img0 and img1 are neighbors in the covisibility graph
            if img0_name not in covisibility_graph :
                debug_log(logger, 'filter_image_pairs', f"{img0_name} not found in covisibility graph.")
                continue
            
            if img1_name not in covisibility_graph[img0_name]:
                debug_log(logger, 'filter_image_pairs', f"Pair ({img0_name}, {img1_name}) not neighbors in covisibility graph. Skipping.")
                continue
            
            # Get the 'Image' object for each image
            imageA = reconstruction.find_image_with_name(img0_name)
            imageB = reconstruction.find_image_with_name(img1_name)
            
            if imageA is None or imageB is None:
                debug_log(logger, 'filter_image_pairs', f"Could not find images in reconstruction. Skipping pair.")
                continue
            
            # Filter 2: Shared 3D points
            pointsA = {pt.point3D_id for pt in imageA.points2D if pt.has_point3D()}
            pointsB = {pt.point3D_id for pt in imageB.points2D if pt.has_point3D()}
            
            # Intersect with the valid points (filtered)
            pointsA_valid = pointsA.intersection(valid_points)
            pointsB_valid = pointsB.intersection(valid_points)
                
            shared_points = pointsA_valid.intersection(pointsB_valid)
            n_shared = len(shared_points)
            
            if n_shared < min_shared_points:
                debug_log(logger, 'filter_image_pairs', f"Pair ({img0_name}, {img1_name}) has only {n_shared} shared points < {min_shared_points}. Skipping.")
                continue
            
            # Filter 3: Covisibility score
            min_count = min(len(pointsA_valid), len(pointsB_valid))
            covis_score = n_shared / float(min_count)
            
            if covis_score < covisibility_threshold:
                debug_log(logger, 'filter_image_pairs', f"Pair ({img0_name}, {img1_name}) has low covisibility score {covis_score:.3f} < {covisibility_threshold}. Skipping.")
                continue

            # Filter 4: Check parallax angle        
            camera_center_A = imageA.cam_from_world.inverse().translation
            camera_center_B = imageB.cam_from_world.inverse().translation

            parallax_angles = [
                compute_parallax(reconstruction.points3D[pid].xyz, camera_center_A, camera_center_B)
                for pid in shared_points
            ]

            if len(parallax_angles) == 0:
                debug_log(logger, 'filter_image_pairs', f"Pair ({img0_name}, {img1_name}) has no parallax angles. WARNING.")
                continue

            median_parallax = np.median(parallax_angles)
            if median_parallax < min_parallax:
                debug_log(logger, 'filter_image_pairs',
                            f"Pair ({img0_name}, {img1_name}) rejected: parallax {(median_parallax):.2f}° < {min_parallax:.2f}°")
                continue
        
            
            final_pairs.append({
                "img0": img0,
                "img1": img1,
                "shared_points": n_shared,
                "covis_score": covis_score,
                "median_parallax": median_parallax,
            })
            match_found = True
            i = j #Jump forward in the sequence
            debug_log(logger, 'filter_image_pairs', f"Pair ({img0_name}, {img1_name}) passed all filters.")
            break  # Greedy: take only the first valid match ahead

        if not match_found:
            i+=1
            debug_log(logger, 'filter_image_pairs', f"No match found for {img0_name}. Skipping to next image.")

    debug_log(logger, 'filter_image_pairs', f"{len(images)} initial images")
    debug_log(logger, 'filter_image_pairs', f"{len(final_pairs)} pairs passed all filters")
    
    return final_pairs

def get_relative_pose_from_colmap(image0, image1):
    """Get relative pose between two images from a COLMAP reconstruction"""

    # Get the absolute poses of the images
    absolute_pose0 = image0.cam_from_world
    absolute_pose1 = image1.cam_from_world

    # Compute relative pose with pycolmap
    R01_colmap, t01_colmap = compute_relative_pose(absolute_pose0, absolute_pose1)

    return R01_colmap, t01_colmap

def get_relative_pose_from_matcher(img0_path, img1_path, camera0, camera1,
                                    output_submap_dir, model_name,
                                    matcher, logger=None, resize=None, masking=False,
                                    plot_kpts=False, min_matches_for_pose=8):
    result_matcher = match_image_pairs(
        img0_path, img1_path,
        output_submap_dir, model_name,
        matcher,
        logger=logger,
        resize=resize,
        masking=masking,
        plot_kpts=plot_kpts
    )

    # Extract sub-step times from the result_matcher
    extractor_time = result_matcher["timings"].get("extractor_time", 0.0)
    filter_time    = result_matcher["timings"].get("filter_time", 0.0)
    match_time     = result_matcher["timings"].get("matcher_time", 0.0)

    # Instead of skipping, add a high penalty error for pairs with too few matches
    if result_matcher is None or len(result_matcher['matched_kpts0']) < min_matches_for_pose:
        logger.warning(f"Not enough matches found or result_matcher is None for {img0_path.stem} and {img1_path.stem}. Adding penalty error values.")
        return None, None, None, None, None, None
    
    mkpts0 = result_matcher['matched_kpts0']
    mkpts1 = result_matcher['matched_kpts1']

    # Convert mkpts to COLMAP coordinates
    corrected_mkpts0 = adapt_mkpts_to_colmap(mkpts0)
    corrected_mkpts1 = adapt_mkpts_to_colmap(mkpts1)

    # Compute E matrix
    #TODO colon: choose RANSAC options
    # estimation_options = pycolmap.RANSACOptions()
    result_colmap = pycolmap.estimate_essential_matrix(corrected_mkpts0, corrected_mkpts1, camera0, camera1)

    # Similarly for essential matrix estimation failure, add penalty instead of skipping
    if result_colmap is None:
        logger.warning(f"Essential matrix estimation failed for {img0_path.stem} and {img1_path.stem}.")
        del result_colmap, corrected_mkpts0, corrected_mkpts1, mkpts0, mkpts1, matcher
        gc.collect()
        return None, None, None, None, None, None


    # Extract R and t from the essential matrix
    R01_est = result_colmap['cam2_from_cam1'].rotation.matrix()
    t01_est = result_colmap['cam2_from_cam1'].translation

    del result_colmap, corrected_mkpts0, corrected_mkpts1, mkpts0, mkpts1, matcher
    gc.collect()
    return result_matcher, R01_est, t01_est, extractor_time, filter_time, match_time