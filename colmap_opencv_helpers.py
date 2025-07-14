import numpy as np
import cv2
import pycolmap
from scipy.spatial.transform import Rotation as R
import torch
from match_image_pairs import match_image_pairs
from my_logging import debug_log
import gc
from memory_profiler import profile
import time
from matching.viz import plot_matches
from pathlib import Path
import json
import random
import os
import warnings

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

def compute_mAA(rot_errors_deg, trans_errors_m, thresholds_r, thresholds_t):
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

def compute_AA(rot_err, trans_err, thresholds_r, thresholds_t):
    """
    Compute Average Accuracy (AA) for a single pair of rotation & translation errors
    at multiple threshold pairs.

    Args:
        rot_err: rotation error in degrees
        trans_err: translation error in degrees
        thresholds_r: list of rotation thresholds in degrees
        thresholds_t: list of translation thresholds in meters

    Returns:
        float: Average Accuracy (AA) score
    """

    assert len(thresholds_r) == len(thresholds_t), "Threshold lists must have the same length"

    for r_thresh, t_thresh in zip(thresholds_r, thresholds_t):
        accuracies = []
        if rot_err < r_thresh and trans_err < t_thresh:
            accuracies.append(1.0)
        else:
            accuracies.append(0.0)

    return np.mean(accuracies)

# #TODO colon_matching: option1 OLD
def translation_error_relative_to_colmap(t_colmap: np.ndarray, t_est: np.ndarray) -> float:
    """
    Compute translation normalized relative error between two translation vectors.

    """
    assert t_colmap.shape == (3,), "t_colmap must be shape (3,)"
    assert t_est.shape == (3,),    "t_est must be shape (3,)"

    # Scale the estimated translation to match the scale of the COLMAP translation
    scale_factor = np.linalg.norm(t_colmap) / np.linalg.norm(t_est)
    t_est_scaled = t_est * scale_factor

    # Compute the normalized error
    err = np.linalg.norm(t_colmap - t_est_scaled)
    return err

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

    if np.array_equal(t_est, np.zeros(3)):
        t_est_unit = np.zeros(3)
    else:
        t_est_unit = t_est / np.linalg.norm(t_est)

    # Normalize the translation vectors
    t_colmap_unit = t_colmap / np.linalg.norm(t_colmap)

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

def compute_relative_pose(T_w_c0, T_w_c1):
    """
    Compute relative pose (R, t) between two camera absolute poses.

    Args:
        T_w_c0 (pycolmap.Rigid3d): Absolute pose of camera 0 (world -> cam0).
        T_w_c1 (pycolmap.Rigid3d): Absolute pose of camera 1 (world -> cam1).

    Returns:
        R_01 (numpy.ndarray): 3x3 relative rotation matrix. (cam0 -> cam1)
        t_01 (numpy.ndarray): 3x1 relative translation vector. (cam0 -> cam1)
    """
    
    T_c0_c1 = T_w_c0.inverse() * T_w_c1 

    return T_c0_c1.rotation.matrix(), T_c0_c1.translation



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


def filter_image_pairs_greedy_sequential(
    images,
    reconstruction,
    covisibility_graph,
    min_parallax=0.0,
    covisibility_threshold=0.0,
    min_track_len=3,
    max_reproj_error=2.0,
    min_shared_points=15,
    min_distance_bw_frames=0,
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

    #TODO colon: check if it is necessary to filter the 3D points. To by pass, set parameters to 0
    valid_points = filter_3D_points(reconstruction, min_track_len, max_reproj_error)
    
    i = 0
    # Greedy forward matching of pairs
    while i < len(images):
        img0 = images[i]
        img0_name = get_img_name(img0)
        match_found = False


        for j in range(i + 1, len(images)):
            img1 = images[j]
            img1_name = get_img_name(img1)
            
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

            # Get frame distance
            distance_bw_frames = get_frame_distance(img0_name, img1_name)
            if distance_bw_frames < min_distance_bw_frames:
                debug_log(logger, 'filter_image_pairs', f"Pair ({img0_name}, {img1_name})rejected:  has distance {distance_bw_frames} < {min_distance_bw_frames}.")
                continue
        
            # Report the reprojection error for the pair with the original 3D points
            shared_points_pre_filtering = pointsA.intersection(pointsB)
            
            # Get reprojection errors for the shared points in both images
            reprojection_errors = get_pair_reprojection_errors(
                reconstruction, imageA, imageB, shared_points_pre_filtering
            )
            
            final_pairs.append({
                "img0": img0,
                "img1": img1,
                "shared_points": n_shared,
                "covis_score": covis_score,
                "median_parallax": median_parallax,
                "reprojection_errors": reprojection_errors,
                "distance_bw_frames": distance_bw_frames,
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

def filter_image_pairs_exhaustive(
    images,
    reconstruction,
    covisibility_graph,
    min_parallax=0.0,
    covisibility_threshold=0.0,
    min_track_len=3,
    max_reproj_error=2.0,
    min_shared_points=15,
    min_distance_bw_frames=0,
    logger=None
):
    """
    Filter image pairs using exhaustive matching (compare all possible pairs).
    
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

    # Filter 3D points based on track length and reprojection error
    valid_points = filter_3D_points(reconstruction, min_track_len, max_reproj_error)

    # Exhaustive matching: check all possible pairs
    for i in range(len(images)):
        img0 = images[i]
        img0_name = get_img_name(img0)

        # Get all possible pairs with images after this one
        for j in range(i + 1, len(images)):
            img1 = images[j]
            img1_name = get_img_name(img1)
            
            # Filter 1: Check if img0 and img1 are neighbors in the covisibility graph
            if img0_name not in covisibility_graph:
                debug_log(logger, 'filter_image_pairs', f"{img0_name} not found in covisibility graph.")
                continue
            
            if img1_name not in covisibility_graph[img0_name]:
                debug_log(logger, 'filter_image_pairs', f"Pair ({img0_name}, {img1_name}) not neighbors in covisibility graph. Skipping.")
                continue
            
            # Get the 'Image' objects
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

            # Get frame distance
            distance_bw_frames = get_frame_distance(img0_name, img1_name)
            if distance_bw_frames < min_distance_bw_frames:
                debug_log(logger, 'filter_image_pairs', f"Pair ({img0_name}, {img1_name}) has distance {distance_bw_frames} < {min_distance_bw_frames}. Skipping.")
                continue

            # Report the reprojection error for the pair with the original 3D points
            shared_points_pre_filtering = pointsA.intersection(pointsB)
            
            # Get reprojection errors for the shared points in both images
            reprojection_errors = get_pair_reprojection_errors(
                reconstruction, imageA, imageB, shared_points_pre_filtering
            )

            # This pair passed all filters, add to final pairs
            final_pairs.append({
                "img0": img0,
                "img1": img1,
                "shared_points": n_shared,
                "covis_score": covis_score,
                "median_parallax": median_parallax,
                "reprojection_errors": reprojection_errors,
                "distance_bw_frames": distance_bw_frames,
            })
            debug_log(logger, 'filter_image_pairs', f"Pair ({img0_name}, {img1_name}) passed all filters.")

    debug_log(logger, 'filter_image_pairs', f"{len(images)} initial images")
    debug_log(logger, 'filter_image_pairs', f"{len(final_pairs)} pairs passed all filters")
    
    return final_pairs

def filter_image_pairs(
    images,
    reconstruction,
    covisibility_graph,
    cache_file=None,
    force_recompute=False,
    min_parallax=0.0,
    covisibility_threshold=0.0,
    min_track_len=3,
    max_reproj_error=2.0,
    min_shared_points=15,
    min_distance_bw_frames=0,
    pair_images_strategy='exhaustive', # 'exhaustive' or 'greedy_sequential'
    random_subset=True,
    random_subset_size=50,
    logger=None
):
    """Filter image pairs with caching capability"""
    
    # Check if cache exists and should be used
    if cache_file and os.path.exists(cache_file) and not force_recompute:
        debug_log(logger, 'filter_image_pairs', f"Loading filtered pairs from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            
        # Convert cached data back to the right format
        final_pairs = []
        for pair_data in cached_data:
            # Convert paths back to Path objects
            pair_data["img0"] = Path(pair_data["img0"])
            pair_data["img1"] = Path(pair_data["img1"])
            pair_data["shared_points"] = int(pair_data["shared_points"])
            pair_data["covis_score"] = float(pair_data["covis_score"])
            pair_data["median_parallax"] = float(pair_data["median_parallax"])
            pair_data["reprojection_errors"] = pair_data["reprojection_errors"] 
            pair_data["distance_bw_frames"] = float(pair_data["distance_bw_frames"])
            final_pairs.append(pair_data)

    elif pair_images_strategy == 'exhaustive':
        # Use exhaustive matching to filter pairs
        debug_log(logger, 'filter_image_pairs', "Using exhaustive matching strategy")
        final_pairs = filter_image_pairs_exhaustive(
            images, reconstruction, covisibility_graph,
            min_parallax=min_parallax,
            covisibility_threshold=covisibility_threshold,
            min_track_len=min_track_len,
            max_reproj_error=max_reproj_error,
            min_shared_points=min_shared_points,
            min_distance_bw_frames=min_distance_bw_frames,
            logger=logger
        )
    elif pair_images_strategy == 'greedy_sequential':
        # Use greedy sequential matching to filter pairs
        debug_log(logger, 'filter_image_pairs', "Using greedy sequential matching strategy")
        final_pairs = filter_image_pairs_greedy_sequential(
            images, reconstruction, covisibility_graph,
            min_parallax=min_parallax,
            covisibility_threshold=covisibility_threshold,
            min_track_len=min_track_len,
            max_reproj_error=max_reproj_error,
            min_shared_points=min_shared_points,
            min_distance_bw_frames=min_distance_bw_frames,
            logger=logger
        )
    else:
        raise ValueError(f"Unknown pair_images_strategy: {pair_images_strategy}. Use 'exhaustive' or 'greedy_sequential'.")
    
    # Save to cache if specified
    if cache_file and not os.path.exists(cache_file):
        # Convert to serializable format
        serializable_pairs = []
        for pair in final_pairs:
            serializable_pair = pair.copy()
            serializable_pair["img0"] = str(pair["img0"])
            serializable_pair["img1"] = str(pair["img1"])
            serializable_pair["shared_points"] = int(pair["shared_points"])
            serializable_pair["covis_score"] = float(pair["covis_score"])
            serializable_pair["median_parallax"] = float(pair["median_parallax"])
            serializable_pair["reprojection_errors"] = serializable_pair["reprojection_errors"]
            serializable_pair["distance_bw_frames"] = float(pair["distance_bw_frames"])
            serializable_pairs.append(serializable_pair)
            
        debug_log(logger, 'filter_image_pairs', f"Saving {len(final_pairs)} pairs to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(serializable_pairs, f, indent=2)

    
    if random_subset and len(final_pairs) >= random_subset_size:
        # Randomly select a subset of pairs if specified
        debug_log(logger, 'filter_image_pairs', f"Selecting random subset of {random_subset_size} pairs from {len(final_pairs)} total pairs")
        final_pairs = random.sample(final_pairs, random_subset_size)
    elif random_subset and len(final_pairs) < random_subset_size:
        # If not enough pairs, use all available pairs
        logger.warning(f"Random subset not applied. Using all {len(final_pairs)} pairs.")
    else:
        debug_log(logger, 'filter_image_pairs', f"Using all {len(final_pairs)} pairs without random selection")
    debug_log(logger, 'filter_image_pairs', f"Final pairs count: {len(final_pairs)}")
    
    return final_pairs

def get_relative_pose_from_colmap(image0, image1):
    """Get relative pose between two images from a COLMAP reconstruction"""

    # Get the absolute poses of the images
    T_w_c0 = image0.cam_from_world.inverse()
    T_w_c1 = image1.cam_from_world.inverse()

    # Compute relative pose with pycolmap
    R01_colmap, t01_colmap = compute_relative_pose(T_w_c0, T_w_c1)

    return R01_colmap, t01_colmap

def get_relative_pose_from_matcher(img0_path, img1_path, camera0, camera1,
                                    output_submap_dir, model_name,
                                    matcher, logger=None, resize=None, masking=False,
                                    plot_kpts=False, min_matches_for_pose=5,
                                    min_inliers_for_pose=5):
    
    #Match images and get the result
    result_matcher, img0, img1, masked_img0, masked_img1 = match_image_pairs(
        img0_path, img1_path,
        output_submap_dir, model_name,
        matcher,
        logger=logger,
        resize=resize,
        masking=masking,
        plot_kpts=plot_kpts
    )

    # Instead of skipping, add a high penalty error for pairs with too few matches
    if result_matcher is None or len(result_matcher['matched_kpts0']) < min_matches_for_pose:
        logger.warning(f"Not enough matches found for {img0_path.stem} and {img1_path.stem}. Adding penalty error values.")
        return None, None, None, None, None, None

    # Extract sub-step times from the result_matcher
    extractor_time = result_matcher["timings"].get("extractor_time", 0.0)
    filter_time    = result_matcher["timings"].get("filter_time", 0.0)
    match_time     = result_matcher["timings"].get("matcher_time", 0.0)
    
    mkpts0 = result_matcher['matched_kpts0']
    mkpts1 = result_matcher['matched_kpts1']

    # Convert mkpts to COLMAP coordinates
    corrected_mkpts0 = adapt_mkpts_to_colmap(mkpts0)
    corrected_mkpts1 = adapt_mkpts_to_colmap(mkpts1)

    # Compute E matrix
    #TODO colon: choose RANSAC options
    # estimation_options = pycolmap.RANSACOptions()
    result_colmap = pycolmap.estimate_essential_matrix(corrected_mkpts0, corrected_mkpts1, camera0, camera1)

    # Similarly for essential matrix estimation failure, add penalty instead of skipping.
    if result_colmap is None:
        logger.warning(f"Could not compute E matrix with {img0_path.stem} and {img1_path.stem}. Adding penalty error values.")
        del result_colmap, corrected_mkpts0, corrected_mkpts1, mkpts0, mkpts1, matcher
        gc.collect()
        return None, None, None, None, None, None

    # Extract R and t from the essential matrix
    R01_est = result_colmap['cam2_from_cam1'].inverse().rotation.matrix()
    t01_est = result_colmap['cam2_from_cam1'].inverse().translation
    
    # Load the inliers from COLMAP
    inlier_mask = result_colmap['inlier_mask']
    result_matcher['inlier_kpts0'] = result_matcher['matched_kpts0'][inlier_mask]
    result_matcher['inlier_kpts1'] = result_matcher['matched_kpts1'][inlier_mask]
    result_matcher['num_inliers'] = len(result_matcher['inlier_kpts0'])

    if result_matcher['num_inliers'] < min_inliers_for_pose:
        logger.warning(f"Not enough inliers for {img0_path.stem} and {img1_path.stem}. Adding penalty error values.")
        del result_colmap, corrected_mkpts0, corrected_mkpts1, mkpts0, mkpts1, matcher
        gc.collect()
        return None, None, None, None, None, None

    if masking:
        start_plotting = time.perf_counter()
        plot_path = output_submap_dir / f'{img0_path.stem}_{img1_path.stem}_{model_name}.png'
        plot_matches(masked_img0, masked_img1, result_matcher, show_all_kpts=plot_kpts, save_path=plot_path)
        end_plotting = time.perf_counter()
        debug_log(logger, 'match_image_pairs', f'Plotting matches took {end_plotting - start_plotting:.3f} seconds')
        debug_log(logger, 'match_image_pairs', f'Saved plot to {plot_path}')
        
    else:
        start_plotting = time.perf_counter()
        plot_path = output_submap_dir / f'{img0_path.stem}_{img1_path.stem}_{model_name}.png'
        plot_matches(img0, img1, result_matcher, show_all_kpts=plot_kpts, save_path=plot_path)
        end_plotting = time.perf_counter()
        debug_log(logger, 'match_image_pairs', f'Plotting matches took {end_plotting - start_plotting:.3f} seconds')
        debug_log(logger, 'match_image_pairs', f'Saved plot to {plot_path}')
            

    del result_colmap, corrected_mkpts0, corrected_mkpts1, mkpts0, mkpts1, matcher, img0, img1, masked_img0, masked_img1
    gc.collect()
    return result_matcher, R01_est, t01_est, extractor_time, filter_time, match_time


#TODO colon: old, should be replaced by adding key, values to the pair_data dict
def save_result_matcher_npz(save_dir: Path,
                            model_name: str,
                            result_matcher: dict,
                            image0_name: str,
                            image1_name: str,
                            R_est: np.ndarray = None,
                            t_est: np.ndarray = None,
                            parallax: float = None,
                            inliers: int = None):
    """
    Save result_matcher and metadata as .npz in subfolder 'npz'.
    """
    npz_dir = save_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    save_path = npz_dir / f"{Path(image0_name).stem}_{Path(image1_name).stem}_{model_name}.npz"

    np.savez_compressed(
        save_path,
        matched_kpts0=result_matcher.get('matched_kpts0'),
        matched_kpts1=result_matcher.get('matched_kpts1'),
        all_kpts0=result_matcher.get('all_kpts0'),
        all_kpts1=result_matcher.get('all_kpts1'),
        scores=result_matcher.get('scores'),
        inlier_kpts0=result_matcher.get('inlier_kpts0'),
        inlier_kpts1=result_matcher.get('inlier_kpts1'),
        image0=image0_name,
        image1=image1_name,
        R_est=R_est,
        t_est=t_est,
        parallax=parallax
    )


def get_img_name(img_path):
    """
    Get the image name considering the parent folder structure, where
    the parent folder is either "exterior" or "interior" (like Graham Hall dataset).
    Other labels could be considered for other datasets.
    
    Args:
        img_path: Path object of the image
        
    Returns:
        str: Formatted image name
    """
    uproot_folder = img_path.parent.name
    
    # Check if the parent folder is "exterior" or "interior"
    if uproot_folder in ["exterior", "interior"]:
        # Add the parent folder of the image to the image name
        return f"{uproot_folder}/{img_path.name}"
    else:
        return img_path.name
    
def get_frame_distance(img0_name, img1_name):
    """
    Extract the frame number from the image names and compute the distance between them.
    Args:
        img0_name (str): Name of the first image
        img1_name (str): Name of the second image
    Returns:
        int: Distance between the two frames
    """
    # Extract the frame number from the image names
    frame0 = int(img0_name.split(".")[0][3:])   # Assuming format like out8627.png
    frame1 = int(img1_name.split(".")[0][3:])   # Assuming format like out8627.png
    # Compute the distance between the two frames
    
    return abs (frame0 - frame1)

def set_all_seeds(seed):
    """Set all seeds for reproducible results."""

    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for any libraries that check it
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"All seeds set to {seed} for reproducibility")

def select_random_subset_of_pairs(pairs, random_subset_size):

    indices = np.random.choice(len(pairs), random_subset_size, replace=False)
    random_pairs = [pairs[i] for i in indices]
    
    return random_pairs

def get_pair_reprojection_errors(reconstruction, imageA, imageB, shared_points):
    """
    Get the reprojection errors of 3D points shared between two images.
    """
    reprojection_errors = []
    
    # Create lookup dictionaries to avoid scanning tracks for each point
    point2D_A_lookup = {}
    point2D_B_lookup = {}
    
    # Pre-populate lookups
    for point3D_id in shared_points:
        point3D = reconstruction.points3D[point3D_id]
        for track_el in point3D.track.elements:
            if track_el.image_id == imageA.image_id:
                point2D_A_lookup[point3D_id] = imageA.points2D[track_el.point2D_idx]
            elif track_el.image_id == imageB.image_id:
                point2D_B_lookup[point3D_id] = imageB.points2D[track_el.point2D_idx]
    
    # Get cameras
    cameraA = reconstruction.cameras[imageA.camera_id]
    cameraB = reconstruction.cameras[imageB.camera_id]
    
    # Process all points
    for point3D_id in shared_points:
        if point3D_id in point2D_A_lookup and point3D_id in point2D_B_lookup:
            point3D = reconstruction.points3D[point3D_id]
            point2D_A = point2D_A_lookup[point3D_id]
            point2D_B = point2D_B_lookup[point3D_id]
            
            # Project and calculate errors
            point3D_camA = imageA.cam_from_world * point3D.xyz
            point3D_camB = imageB.cam_from_world * point3D.xyz

            # Normalize homogeneous coordinates
            proj_A = cameraA.img_from_cam(point3D_camA)
            proj_B = cameraB.img_from_cam(point3D_camB)
            
            error_A = np.linalg.norm(proj_A - point2D_A.xy)
            error_B = np.linalg.norm(proj_B - point2D_B.xy)
            
            rms_error = np.sqrt((error_A**2 + error_B**2) / 2)
            reprojection_errors.append(rms_error)
    
    return reprojection_errors