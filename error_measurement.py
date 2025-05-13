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



############################# CONFIG #############################
device = get_default_device()
device_info = get_device_info(device)
############################# Logging ############################# 
DEBUG = True  # Global debug flag. Set to False to disable extra debug logging.
# activated_debug_flags = {"match_image_pairs", "error_measurement"} 
# "filter_image_pairs", "filter_image_feats_with_mask", "easy_medium_hard", "filter_image_feats_with_mask"
#  "filter_feat_dict_with_mask", "base_matcher_forward", "Roma_forward", "Roma_forward_symmetric", "TinyRoma_forward"
# activated_debug_flags = {"ALL"}
activated_debug_flags = {"filter_image_pairs","error_measurement"}
############################# CHOOSE MODELS #############################
# model_name = 'sift-nn'
# model_name = 'gim-lg'
# model_name = 'tiny-roma'
# model_name = 'sift-lg'
# model_name = 'superpoint-lg'
# model_name = 'roma'
models = ["sift-nn", "superpoint-lg"]
############################# CHOOSE IMAGE DIRECTORY #############################
image_dir = Path(f'data')
############################# RESIZE #############################
# resize = 512
resize = None
############################# MASK #############################
# specular_mask = False
masking = True
############################# Key Points #############################
plot_kpts = True

############################# Matcher Hyperparameters #############################
# Parameters for SuperPoint extractor and LightGlue matcher
matcher_kwargs = {
############################### SuperPoint #############################
    'detection_threshold': 0.005,  # SuperPoint keypoint detection threshold
    'max_superpoint_keypoints': 1024,     # Maximum number of keypoints to detect
############################### LightGlue (with superpoint, check if using with another detector, should modify the matcher class) #############################
    'filter_threshold': 0.1 ,       # LightGlue matching threshold
    "depth_confidence": -1,        # early stopping, disable with -1 / 0.95
    "width_confidence": -1,          # point pruning, disable with -1 / 0.99
############################### SIFT #############################
    'lowe_thresh': 0.85,          # Lowe's ratio test threshold
    'max_sift_keypoints': 2048,    # Maximum number of keypoints to detect
    'contrast_threshold': 0.02,   # SIFT contrast threshold
    'edge_threshold': 15,         # SIFT edge threshold
    'n_octave_layers': 4,         # SIFT number of octave layers
############################### KNN (SIFT) #############################
    # TODO colon: if we want to upgrade these parameter, some logic in the matcher should be changed. Right now it's hardcoded to 2
    'k_neighbors': 2             # Number of nearest neighbors to consider for matching
}
############################# RANSAC #############################
#RANSAC it´s not used in the current implementation (we could bypass it)
ransac_kwargs = {
    'ransac_reproj_thresh': 0.0,
    'ransac_conf': 0.0,
    'ransac_iters': 0
}
############################# Seq #############################
seq = "seq_001"
# seq = "graham-hall_toy"
# seq = "seq_001_toy"
seq_dir = Path(f'data/{seq}')
############################# Subsample #############################
subsample = 1 # for subsampling the img_train list
min_images_afer_subsampling = 2 # for skipping submaps with too few images after subsampling
############################# Covisibility #############################|
covisibility_threshold = 0.0 # for filtering image pairs (easy, medium, hard)
min_shared_points = 15 # for filtering image pairs
min_track_len = 3 # for filtering image pairs
max_reproj_error = 2.0 # for filtering image pairs
############################ Parallax #############################
min_parallax = 1 # for robust estimation of relative pose (degrees)
############################# Min Pairs for submap #############################
min_pairs_for_submap = 10 # for skipping submaps with too few pairs
############################# Min Matches for pose estimation #############################
min_matches_for_pose = 5 # for skipping pairs with too few matches
############################# Thresholds #############################
#TODO: colon: add a relative threshold to t_colmap for the translation error?
thresholds_r = np.array([1,3,5,10,20]) 
thresholds_t = np.array([5,10,15,20,30])
############################# Thresholds #############################
# pair_images_strategy = "greedy_sequential" # for filtering image pairs
pair_images_strategy = "exhaustive" # for filtering image pairs


def main_loop():

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
    for model_name in models:
        logger.info(f"Processing model: {model_name}")

        # Initialize the matcher
        matcher = get_matcher(model_name, device=device, **matcher_kwargs, **ransac_kwargs)
        debug_log(logger, "error_measurement", f"Extractor conf: {matcher.extractor.conf}")
        debug_log(logger, "error_measurement", f"Matcher conf: {matcher.matcher.conf}")
    
        try:
            # Iterate over all the submaps in the sequence
            for submap in [submaps[6], submaps[9]]:               ######################### USING JUST SOME SUBMAPS ########################
            # for submap in submaps:
                logger.info(f"Starting submap: {submap}")
                # Define the paths for the submap model and the source of images of the mode
                sparse_model_dir = seq_dir / 'sparse' / submap
                
                # Load the COLMAP reconstruction for the submap
                reconstruction = pycolmap.Reconstruction(sparse_model_dir)

                #Get image pairs for the submap
                images = list(Path(f'data/{seq}/sub_maps_images/{submap}').glob('*.png')) + \
                        list(Path(f'data/{seq}/img_train/exterior').glob('*.jpg')) + \
                        list(Path(f'data/{seq}/img_train/exterior').glob('*.JPG')) + \
                        list(Path(f'data/{seq}/img_train/interior').glob('*.jpg')) + \
                        list(Path(f'data/{seq}/img_train/interior').glob('*.JPG')) 
                                        
                                        
                # Subsample the images list to avoid memory issues
                images = images[::subsample]
                debug_log(logger, "error_measurement", f"{submap} subsampled by {subsample} has {len(images)} images")
                
                if len(images) < min_images_afer_subsampling:
                    logger.info(f"Skipping submap {submap} because it has less than {min_images_afer_subsampling} images after subsampling")
                    continue
                
                covisibility_path = sparse_model_dir / "camerasModel.txt"
                covisibility_graph = load_covisibility_graph(covisibility_path)
                # Sort images to ensure they're in sequential order
                images.sort(key=lambda x: x.name)

                if pair_images_strategy == "exhaustive":
                
                    # All with all, passing the filters (covisibility_graph, min_parallax)
                    pairs = filter_image_pairs(images, reconstruction, covisibility_graph, min_parallax=min_parallax ,covisibility_threshold = covisibility_threshold, min_track_len=min_track_len,
                                            max_reproj_error=max_reproj_error, min_shared_points=min_shared_points, logger=logger)

                elif pair_images_strategy == "greedy_sequential":
                    
                    # Take pair of images that first sastisfy the filters (covisibility_graph, min_parallax)
                    pairs = filter_image_pairs_greedy_sequential(images, reconstruction, covisibility_graph, min_parallax=min_parallax ,covisibility_threshold = covisibility_threshold, min_track_len=min_track_len,
                                max_reproj_error=max_reproj_error, min_shared_points=min_shared_points, logger=logger)
                
                else:
                    raise ValueError(f"Unknown pair_images_strategy: {pair_images_strategy}")
                
                # Compute total images in the submap with the filtered pairs
                total_images_submap = len(set(itertools.chain(*pairs)))

                if len(pairs) <  min_pairs_for_submap:
                    logger.info(f"Skipping submap {submap} because there are no image pairs to process")
                    continue

                output_report_dir = Path(f'output/error_measurement/{seq}/{model_name}/{timestamp}')
                output_report_dir.mkdir(parents=True, exist_ok=True)

                output_submap_dir = output_report_dir / submap
                output_submap_dir.mkdir(parents=True, exist_ok=True)

                # Storage for submap-level metrics
                pair_metrics = []
                submap_rot_errs = []
                submap_trans_errs = []
                registered_images = set()

                # Measure submap pipeline total time
                submap_start_time = time.perf_counter()

                for pair_data in pairs:
                    
                    img0_path = pair_data["img0"]
                    img1_path = pair_data["img1"]
                    covis_score = pair_data["covis_score"]
                    parallax = pair_data["median_parallax"]
                    
                    # Create a terminal progress bar
                    progress_bar(logger, len(pair_metrics), len(pairs), img0_path, img1_path)

                    pair_start_time = time.perf_counter()
                    # Get images names


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
                    
                    ##### START get_relative_pose_from_matcher #######
                    # Get relative pose froom matcher
                    result_matcher, R01_est, t01_est, extractor_time, filter_time, match_time = get_relative_pose_from_matcher(
                        img0_path, img1_path, camera0, camera1, output_submap_dir, model_name, matcher, logger=logger, resize=resize, masking=masking, plot_kpts=plot_kpts, min_matches_for_pose=min_matches_for_pose
                    )

                    if result_matcher is None:
                        continue

                    # rotation error
                    rot_err = rotation_error_deg(R01_colmap, R01_est)
                    # translation error relative to the diameter of the submap
                    trans_err_deg = translation_error_direction_deg(t01_colmap, t01_est)
                    trans_err_rel = translation_error_relative_to_colmap(t01_colmap, t01_est)
                    pair_end_time = time.perf_counter()
                    
                    debug_log(logger, "error_measurement",f"{img0_path.name.rsplit('.', 1)[0]}_{img1_path.name.rsplit('.', 1)[0]} trans_err_rel: {trans_err_rel:.3f} ")
                    debug_log(logger, "error_measurement",f"{img0_path.name.rsplit('.', 1)[0]}_{img1_path.name.rsplit('.', 1)[0]} trans_error_deg: {trans_err_deg:.3f} degrees" )
                    debug_log(logger, "error_measurement",f"{img0_path.name.rsplit('.', 1)[0]}_{img1_path.name.rsplit('.', 1)[0]} rot_error: {rot_err:.3f} degrees" )
                    debug_log(logger, "error_measurement",f"parallax: {parallax:.3f} degrees" )
                    debug_log(logger, "error_measurement",f"inliers: {result_matcher['num_inliers']}")
                    debug_log(logger, "error_measurement",f"matches: {len(result_matcher['matched_kpts0'])}")

                    #TODO colon: make another submap_errs_rel or whatever if I want to use both metrics.
                    submap_rot_errs.append(rot_err)
                    submap_trans_errs.append(trans_err_deg)

                    # Build a dictionary for this pair
                    pair_info = {
                        "image0": img0_path.name,
                        "image1": img1_path.name,
                        "mkpts": len(result_matcher['matched_kpts0']),
                        "inliers": result_matcher['num_inliers'],
                        "kpts0": len(result_matcher['all_kpts0']),
                        "kpts1": len(result_matcher['all_kpts1']),
                        "extractor_time": extractor_time,
                        "filter_time": filter_time,
                        "matcher_time": match_time,
                        "total_pair_time": pair_end_time - pair_start_time,
                        "rot_error":  rot_err,
                        "trans_error_deg":    trans_err_deg,
                        "trans_error_rel": trans_err_rel,
                        "t_colmap_norm": np.linalg.norm(t01_colmap),
                        "covis_score": covis_score,
                        "parallax": parallax
                    }
                    pair_metrics.append(pair_info)

                    # Add images to registered images set
                    registered_images.add(img0_path.name)
                    registered_images.add(img1_path.name)
                    
                    # Save the result of the matcher in hard disk
                    save_result_matcher_npz(
                        output_submap_dir,
                        model_name,
                        result_matcher,
                        img0_name,
                        img1_name,
                        R01_est,
                        t01_est, 
                        parallax,
                    )
                                    
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
                
                # Guarda el JSON del submap individualmente
                submap_output_dir = Path(f'output/error_measurement/{seq}/{model_name}/{timestamp}/submaps')
                submap_output_dir.mkdir(parents=True, exist_ok=True)
                save_submap_report(
                    submap_data, seq, submap, model_name, 
                    submap_output_dir, logger, reason="complete"
                )
                        
                del submap_rot_errs, submap_trans_errs, pair_metrics, registered_images
                del reconstruction, covisibility_graph, images, pairs
                del submap_data

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            # Al final, genera el reporte de secuencia completa
            sequence_output_dir = Path(f'output/error_measurement/{seq}/{model_name}/{timestamp}')
            submaps_dir = sequence_output_dir / 'submaps'
            
            # Metadata para el reporte de secuencia
            metadata = {
                "device_info": device_info,
                "resize": resize,
                "masking": masking,
                "matcher_params": matcher_kwargs,
                "subsampling": subsample,
                "thresholds_r": thresholds_r.tolist(),
                "thresholds_t": thresholds_t.tolist(),
                "extractor_config": matcher.extractor.conf,
                "matcher_config": matcher.matcher.conf
            }
            
            # Crea el reporte final de secuencia
            create_sequence_report(seq, model_name, submaps_dir, sequence_output_dir, logger, metadata)
            
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            debug_log(logger, "memory_management", "[Top 10 memory consuming lines]")
            for stat in top_stats[:10]:
                debug_log(logger, "memory_management", stat)

        except KeyboardInterrupt:
            # Código similar pero también guarda el submap actual de forma individual
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
                
                # Guarda el JSON del submap individualmente
                submap_output_dir = Path(f'output/error_measurement/{seq}/{model_name}/{timestamp}/submaps')
                submap_output_dir.mkdir(parents=True, exist_ok=True)
                save_submap_report(
                    submap_data, seq, submap, model_name, 
                    submap_output_dir, logger, reason="interrupted"
                )

                # Al final, genera el reporte de secuencia completa
                sequence_output_dir = Path(f'output/error_measurement/{seq}/{model_name}/{timestamp}')
                submaps_dir = sequence_output_dir / 'submaps'
                
                # Metadata para el reporte de secuencia
                metadata = {
                    "device_info": device_info,
                    "resize": resize,
                    "masking": masking,
                    "matcher_params": matcher_kwargs,
                    "subsampling": subsample,
                    "thresholds_r": thresholds_r.tolist(),
                    "thresholds_t": thresholds_t.tolist(),
                    "extractor_config": matcher.extractor.conf,
                    "matcher_config": matcher.matcher.conf,
                }
                
                # Crea el reporte final de secuencia
                create_sequence_report(seq, model_name, submaps_dir, sequence_output_dir, logger, metadata)

                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')

                debug_log(logger, "memory_management", "[Top 10 memory consuming lines]")
                for stat in top_stats[:10]:
                    debug_log(logger, "memory_management", stat)
                
            sys.exit(1)
        

if __name__ == "__main__":
    main_loop()
