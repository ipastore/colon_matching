import pycolmap
from matching import get_matcher
from specular_mask import *
# from process_image_pairs import *
from pathlib import Path
from colmap_opencv_helpers import *
# from config_error_measurement import *
from my_logging import setup_logging, debug_log
import numpy as np
import itertools
from matching.utils import get_default_device
import time
import json
import csv
import os
import sys
import traceback
from datetime import datetime
import torch
from reporting_helpers import finalize_current_submap, create_sequence_report, save_submap_report, progress_bar
import gc
import psutil
import tracemalloc
import yaml

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


############################# CONFIG #############################
# Load configuration from YAML file
config = load_config()

device = get_default_device()
device_info = get_device_info(device)
############################# Logging ############################# 
DEBUG = config['debug']['enabled']  # Global debug flag from config
activated_debug_flags = set(config['debug']['activated_flags'])
############################# CHOOSE MODELS #############################
models = config['models']
############################# RESIZE #############################
resize = config['resize']
############################# MASK #############################
masking = config['masking']
############################# Key Points #############################
plot_kpts = config['plot_kpts']
############################# Matcher Hyperparameters #############################
matcher_kwargs = config['matcher_kwargs']
############################# RANSAC #############################
ransac_kwargs = config['ransac_kwargs']
############################# Seq #############################
seq = config['seq']
seq_dir = Path(f'data/{seq}')
############################# Covisibility #############################|
covisibility_threshold = config['covisibility_threshold']
min_shared_points = config['min_shared_points']
min_track_len = config['min_track_len']
max_reproj_error = config['max_reproj_error']
############################ Parallax #############################
min_parallax = config['min_parallax']
############################# Min Pairs for submap #############################
min_pairs_for_submap = config['min_pairs_for_submap']
############################# Min Matches for pose estimation #############################
min_matches_for_pose = config['min_matches_for_pose']
min_inliers_for_pose = config['min_inliers_for_pose']
############################# Thresholds #############################
thresholds_r = np.array(config['thresholds_r'])
thresholds_t = np.array(config['thresholds_t'])
############################# Pair Images Strategy #############################
pair_images_strategy = config['pair_images_strategy']
random_subset = config['random_subset']
random_subset_size = config['random_subset_size']
############################## Min distance between frames #############################
min_distance_bw_frames = config['min_distance_bw_frames']

def main_loop():
    
    # Set seeds for reproducibility
    seed = 42
    set_all_seeds(seed)

    tracemalloc.start()

    # Start logger
    logger = setup_logging(DEBUG, activated_debug_flags)

    # Get all submaps of sequence get the names of the submaps
    submaps = [int(f.stem) for f in Path(f'data/{seq}/sparse').iterdir() if f.is_dir()]
    # Sort in ascending order
    submaps.sort()
    # Convert to string
    submaps = [str(submap) for submap in submaps]

    # Create timestamp for report filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Iterate over all the submaps in the sequence
    for submap in submaps:
    # for submap in [submaps[6], submaps[9]]:               ######################### USING JUST SOME SUBMAPS ########################

        
        logger.info(f"Starting submap: {submap}")
        # Define the paths for the submap model and the source of images of the mode
        sparse_model_dir = seq_dir / 'sparse' / submap
        
        # Load the COLMAP reconstruction for the submap
        reconstruction = pycolmap.Reconstruction(str(sparse_model_dir))

        # Get p3d error for submap pre filtering[]. Used for reporting purposes
        submap_p3d_errors_pre_filter = [p3d.error for p3d in reconstruction.points3D.values()]

        #Get image pairs for the submap
        images = list(Path(f'data/{seq}/submap_images/{submap}').glob('*.png')) 
        

        # Load previously generated covisibility graph (generate_cameras_model.py)                                       
        covisibility_path = sparse_model_dir / "camerasModel.txt"
        covisibility_graph = load_covisibility_graph(covisibility_path)
        # Sort images to ensure they're in sequential order
        images.sort(key=lambda x: x.name)

        # Define a cache file path based on your parameters
        cache_dir = Path(f"cache/{submap}/{pair_images_strategy}")

        cache_filename = f"pairs_parallax{min_parallax}_frames{min_distance_bw_frames}.json"
        cache_file = seq_dir / cache_dir / cache_filename
        # Ensure the cache directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Use the cached version
        pairs = filter_image_pairs(
            images, reconstruction, covisibility_graph, 
            cache_file=cache_file,
            force_recompute=False,  # Set to True if you want to recompute regardless of cache
            min_parallax=min_parallax,
            covisibility_threshold=covisibility_threshold, 
            min_track_len=min_track_len,
            max_reproj_error=max_reproj_error, 
            min_shared_points=min_shared_points, 
            min_distance_bw_frames=min_distance_bw_frames,
            pair_images_strategy=pair_images_strategy,
            random_subset=random_subset,
            random_subset_size=random_subset_size, 
            logger=logger
        )

        # Compute total pair of images in the submap with the filtered pairs
        total_images_submap = len(pairs)

        if len(pairs) <  min_pairs_for_submap:
            logger.info(f"Skipping submap {submap} because there are no image pairs to process")
            continue

        for model_name in models:
            try:
                logger.info(f"Processing model: {model_name}")

                # Initialize the matcher
                matcher = get_matcher(model_name, device=device, **matcher_kwargs, **ransac_kwargs)
                # TODO: for Roma and maybe others, there is no matcher.extractor or matcher.conf. I have fixed it for SIFT-NN but dk if I should for all the others
                if hasattr(matcher, 'extractor') and matcher.extractor is not None and hasattr(matcher.extractor, 'conf'):
                    debug_log(logger, "error_measurement", f"Extractor conf: {matcher.extractor.conf}")
                if hasattr(matcher, 'matcher') and matcher.matcher is not None and hasattr(matcher.matcher, 'conf'):
                    debug_log(logger, "error_measurement", f"Matcher conf: {matcher.matcher.conf}")
                   
                output_report_dir = Path(f'output/error_measurement/{seq}/{model_name}/{timestamp}')
                output_report_dir.mkdir(parents=True, exist_ok=True)

                output_submap_dir = output_report_dir / submap
                output_submap_dir.mkdir(parents=True, exist_ok=True)

                # Storage for submap-level metrics
                pair_metrics = []
                submap_rot_errs = []
                submap_trans_errs = []
                registered_images = []

                # Measure submap pipeline total time
                submap_start_time = time.perf_counter()

                for pair_data in pairs:
                    
                    img0_path = pair_data["img0"]
                    img1_path = pair_data["img1"]
                    covis_score = pair_data["covis_score"]
                    parallax = pair_data["median_parallax"]
                    reprojection_errors = pair_data["reprojection_errors"]
                    
                    #FIX colon: need to add the progress when penalizing with low inliers, len(pair_metrics) doesnt grow
                    # Create a terminal progress bar
                    progress_bar(logger, len(pair_metrics), len(pairs), img0_path, img1_path)

                    # Start timer for pair processing
                    pair_start_time = time.perf_counter()

                    # Get images names
                    img0_name = get_img_name(img0_path)
                    img1_name = get_img_name(img1_path)

                    # Get image objects
                    image0 = reconstruction.find_image_with_name(img0_name)
                    image1 = reconstruction.find_image_with_name(img1_name)

                    # Get camera objects
                    camera0 = reconstruction.camera(image0.camera_id)
                    camera1 = reconstruction.camera(image1.camera_id)

                    # Get relative pose from colmap
                    R01_colmap, t01_colmap = get_relative_pose_from_colmap(image0, image1)
                    
                    # Get relative pose from matcher
                    result_matcher, R01_est, t01_est, extractor_time, filter_time, match_time = get_relative_pose_from_matcher(
                        img0_path, img1_path, camera0, camera1, output_submap_dir, model_name, matcher, logger=logger, resize=resize,
                          masking=masking, plot_kpts=plot_kpts, min_matches_for_pose=min_matches_for_pose, min_inliers_for_pose=min_inliers_for_pose
                    )

                    registered = True
                    # Penalize with identity if no matches or inliers
                    if result_matcher is None or R01_est is None or t01_est is None:
                        R01_est = np.identity(3)
                        t01_est = np.zeros(3)
                        registered = False

                    # rotation error
                    rot_err = rotation_error_deg(R01_colmap, R01_est)
                    # translation error relative to the diameter of the submap
                    trans_err_deg = translation_error_direction_deg(t01_colmap, t01_est)
                    pair_end_time = time.perf_counter()

                    if registered:
                        # Add pair of images to registered images set
                        registered_images.append((img0_name, img1_name))

                    debug_log(logger, "error_measurement",f"{img0_name}_{img1_name} trans_error_deg: {trans_err_deg:.3f} degrees")
                    debug_log(logger, "error_measurement",f"{img0_name}_{img1_name} rot_error: {rot_err:.3f} degrees" )
                    debug_log(logger, "error_measurement",f"parallax: {parallax:.3f} degrees" )
                    debug_log(logger, "error_measurement",f"inliers: {result_matcher['num_inliers']}" if result_matcher is not None and 'num_inliers' in result_matcher else "N/A")
                    debug_log(logger, "error_measurement",f"matches: {len(result_matcher['matched_kpts0'])}" if result_matcher is not None and 'matched_kpts0' in result_matcher else "N/A")

                    # Append errors to submap lists, for reporting purposes
                    submap_rot_errs.append(rot_err)
                    submap_trans_errs.append(trans_err_deg)

                    # Build a dictionary for this pair
                    pair_info = {
                        "image0": img0_name,
                        "image1": img1_name,
                        "mkpts": len(result_matcher['matched_kpts0']) if result_matcher is not None and 'matched_kpts0' in result_matcher else 0,
                        "inliers": result_matcher['num_inliers'] if result_matcher is not None and 'num_inliers' in result_matcher else 0,
                        "kpts0": len(result_matcher['all_kpts0']) if result_matcher is not None and 'all_kpts0' in result_matcher else 0,
                        "kpts1": len(result_matcher['all_kpts1']) if result_matcher is not None and 'all_kpts1' in result_matcher else 0,
                        "extractor_time": extractor_time if extractor_time is not None else 0.0,
                        "filter_time": filter_time if filter_time is not None else 0.0,
                        "matcher_time": match_time if match_time is not None else 0.0,
                        "total_pair_time": pair_end_time - pair_start_time,
                        "rot_error":  rot_err,
                        "trans_error_deg":    trans_err_deg,
                        "t_colmap_norm": np.linalg.norm(t01_colmap),
                        "t_est_norm": np.linalg.norm(t01_est),
                        "covis_score": covis_score,
                        "parallax": parallax,
                        "R01_est": R01_est,
                        "t01_est": t01_est,
                        "R01_colmap": R01_colmap,
                        "t01_colmap": t01_colmap,
                        "mean_reprojection_error": np.mean(reprojection_errors),   
                        "std_reprojection_error": np.std(reprojection_errors),            
                        "median_reprojection_error": np.median(reprojection_errors),
                        "n_3d_colmap_points": len(reprojection_errors),    
                        "distance_bw_frames": pair_data["distance_bw_frames"],
                    }

                    pair_metrics.append(pair_info)

                    del result_matcher, R01_est, t01_est, R01_colmap, t01_colmap, image0, image1, camera0, camera1, pair_info
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                # Finalize submap
                submap_data = finalize_current_submap(
                    submap, submap_rot_errs, submap_trans_errs, pair_metrics, 
                    submap_start_time, thresholds_r, thresholds_t,
                    registered_images, total_images_submap, logger
                )
                
                # Metadata
                metadata = {
                    "device_info": device_info,
                    "resize": resize,
                    "masking": masking,
                    "matcher_params": matcher_kwargs,
                    "thresholds_r": thresholds_r.tolist(),
                    "thresholds_t": thresholds_t.tolist(),
                    "extractor_config": matcher.extractor.conf if hasattr(matcher, 'extractor') and hasattr(matcher.extractor, 'conf') else "not available",
                    "matcher_config": matcher.matcher.conf if hasattr(matcher, 'matcher') and hasattr(matcher.matcher, 'conf') else "not available",
                    "sequence": seq,
                    "submap": submap,
                    "min_parallax": min_parallax,
                    "min_distance_bw_frames": min_distance_bw_frames,
                    "pair_images_strategy": pair_images_strategy,
                    "random_subset": random_subset,
                    "random_subset_size": random_subset_size,
                    "mean_submap_p3d_error_pre_filter": np.mean(submap_p3d_errors_pre_filter),
                    "median_submap_p3d_error_pre_filter": np.median(submap_p3d_errors_pre_filter),
                    "model_name": model_name,
                    "timestamp": timestamp,
                    "report_status": "complete",
                    
                }

                save_submap_report(
                    submap_data, seq, submap, model_name, timestamp,
                    metadata, logger=logger
                )
                        
                del submap_rot_errs, submap_trans_errs, pair_metrics, registered_images
                del reconstruction, covisibility_graph, images, pairs
                del submap_data

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except KeyboardInterrupt:
                # Finish and save current submap if interrupted
                logger.warning("Processing interrupted by user (Ctrl+C)")
                
                # Try to finalize current submap if we're in the middle of one
                current_submap = submaps[submaps.index(submap)] if 'submap' in locals() else None

                if current_submap and 'submap_start_time' in locals():
                    logger.info(f"Finalizing data for current submap {current_submap} before exiting")
                    submap_data = finalize_current_submap(
                        submap, submap_rot_errs, submap_trans_errs, pair_metrics, 
                        submap_start_time, thresholds_r, thresholds_t,
                        registered_images, total_images_submap, logger
                    )
                    
                    # TODO colon: add metadata to the submap report
                    # Metadata
                    metadata = {
                        "device_info": device_info,
                        "resize": resize,
                        "masking": masking,
                        "matcher_params": matcher_kwargs,
                        "thresholds_r": thresholds_r.tolist(),
                        "thresholds_t": thresholds_t.tolist(),
                        "extractor_config": matcher.extractor.conf if hasattr(matcher, 'extractor') and hasattr(matcher.extractor, 'conf') else "not available",
                        "matcher_config": matcher.matcher.conf if hasattr(matcher, 'matcher') and hasattr(matcher.matcher, 'conf') else "not available",
                        "sequence": seq,
                        "submap": submap,
                        # "submap_p3d_errors_pre_filter": submap_p3d_errors_pre_filter,
                        "mean_submap_p3d_error_pre_filter": np.mean(submap_p3d_errors_pre_filter),
                        "median_submap_p3d_error_pre_filter": np.median(submap_p3d_errors_pre_filter),
                        "model_name": model_name,
                        "timestamp": timestamp,
                        "report_status": "interrupted",
                    }

                    save_submap_report(
                        submap_data, seq, submap, model_name, timestamp,
                        metadata, logger=logger
                    )

                    sys.exit(1)
        

if __name__ == "__main__":
    main_loop()
