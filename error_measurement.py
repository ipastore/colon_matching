import pycolmap
from matching import get_matcher
from specular_mask import *
from process_image_pairs import *
from pathlib import Path
from colmap_opencv_helpers import *
# from config_error_measurement import *
from my_logging import setup_logging
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
from reporting_helpers import save_report, finalize_current_submap

############################# CONFIG #############################
device = get_default_device()
device_info = get_device_info(device)
############################# Logging ############################# 
DEBUG = True  # Global debug flag. Set to False to disable extra debug logging.
############################# CHOOSE MODELS #############################
# model_name = 'sift-nn'
# model_name = 'gim-lg'
# model_name = 'tiny-roma'
# model_name = 'sift-lg'
model_name = 'superpoint-lg'
# model_name = 'roma'
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
############################# Sequence and Submap #############################
seq = "seq_001"
############################# Matcher Hyperparameters #############################
# Parameters for SuperPoint extractor and LightGlue matcher
matcher_kwargs = {
    'detection_threshold': 0.005,  # SuperPoint keypoint detection threshold
    'max_num_keypoints': 1024,     # Maximum number of keypoints to detect
    'filter_threshold': 0.1 ,       # LightGlue matching threshold
    "depth_confidence": -1,        # early stopping, disable with -1 / 0.95
    "width_confidence": -1          # point pruning, disable with -1 / 0.99
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
seq_dir = Path(f'data/{seq}')
############################# Subsample #############################
subsample = 20
############################# Thresholds #############################
thresholds_r = np.linspace(0.1, 5, 10)
thresholds_t = np.linspace(0.01, 0.1, 10)

# Start logger
logger = setup_logging(DEBUG)
matcher = get_matcher(model_name, device=device, **matcher_kwargs, **ransac_kwargs)
logger.debug(f"Extractor conf: {matcher.extractor.conf}")
logger.debug(f"Matcher conf: {matcher.matcher.conf}")

# Get all submaps of sequence get the names of the submaps
submaps = [f.stem for f in Path(f'data/{seq}/sparse').iterdir() if f.is_dir()]
# Sort in ascending order
submaps.sort()

# Prepare overall data structure to store results
final_report = {
    "device_info": device_info,
    "model_name": model_name,
    "resize": resize,
    "masking": masking,
    "matcher_params": matcher_kwargs,
    "sequence_name": seq,
    "subsampling": subsample,
    "thresholds_r": thresholds_r.tolist(),
    "thresholds_t": thresholds_t.tolist(),
    "extractor_config": matcher.extractor.conf,       # or str(matcher.extractor.conf)
    "matcher_config": matcher.matcher.conf,           # or str(matcher.matcher.conf)
    "submaps": {},
    "sequence_averages": {},
    "sequence_mAA": 0.0,
    "processing_complete": False,  # Flag to indicate if processing completed normally
}

# We'll keep track of times/metrics across all submaps to compute a global average
all_pairs_extractor_times   = []
all_pairs_filter_times      = []
all_pairs_matcher_times     = []
all_pairs_total_times       = []
all_rot_errs                = []
all_trans_errs              = []
all_submap_aas              = []

# Create timestamp for report filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

try:
    # Iterate over all the submaps in the sequence
    for submap in submaps:
        logger.info(f"Starting submap: {submap}")
        # Define the paths for the submap model and the source of images of the mode
        sparse_model_dir = seq_dir / 'sparse' / submap
        
        # Load the COLMAP reconstruction for the submap
        reconstruction = pycolmap.Reconstruction(sparse_model_dir)
        d = compute_submap_diameter(sparse_model_dir)
        logger.debug(f"{submap} diameter: {d:.3f}")

        #Get image pairs for the submap
        images = list(Path(f'data/{seq}/sub_maps_images/{submap}').glob('*.png'))
        # Subsample the images list to avoid memory issues
        images = images[::subsample]
        logger.debug(f"{submap} subsampled by {subsample} has {len(images)} images")
        
        if len(images) < 5:
            logger.info(f"Skipping submap {submap} because it has less than 5 images after subsampling")
            continue

        # Get all the pairs for the subsampled images  
        pairs = list(itertools.combinations(images, 2))
        output_report_dir = Path(f'output/error_measurement/{seq}/{model_name}/{timestamp}')
        output_report_dir.mkdir(parents=True, exist_ok=True)

        output_submap_dir = output_report_dir / submap
        output_submap_dir.mkdir(parents=True, exist_ok=True)

        # Storage for submap-level metrics
        pair_metrics = []
        submap_rot_errs = []
        submap_trans_errs = []
        
        # Measure submap pipeline total time
        submap_start_time = time.perf_counter()

        for img0_path, img1_path in pairs:
            logger.info(f"Starting pair: {img0_path.stem} and {img1_path.stem}")
            pair_start_time = time.perf_counter()

            try:
                image0_name_extension = img0_path.name
                image1_name_extension = img1_path.name

                # Get the image objects from the reconstruction
                image0 = reconstruction.find_image_with_name(image0_name_extension)
                image1 = reconstruction.find_image_with_name(image1_name_extension)

                # Get the absolute poses of the images
                absolute_pose0 = image0.cam_from_world
                absolute_pose1 = image1.cam_from_world

                # Compute relative pose with pycolmap
                R01_colmap, t01_colmap = compute_relative_pose(absolute_pose0, absolute_pose1)

                # Process the image pair with the matcher
                result = process_image_pairs(
                    img0_path, img1_path,
                    output_submap_dir, model_name,
                    matcher,
                    logger=logger,
                    resize=resize,
                    masking=masking,
                    plot_kpts=plot_kpts
                )

                if len(result['matched_kpts0']) < 5:
                    logger.warning(f"Not enough matches found for {img0_path.stem} and {img1_path.stem}")
                    continue

                # Extract sub-step times from the result
                extractor_time = result["timings"].get("extractor_time", 0.0)
                filter_time      = result["timings"].get("filter_time", 0.0)
                match_time     = result["timings"].get("matcher_time", 0.0)

                mkpts0 = result['matched_kpts0']
                mkpts1 = result['matched_kpts1']

                # Get camera objects
                camera0 = reconstruction.camera(image0.camera_id)
                camera1 = reconstruction.camera(image1.camera_id)

                # Convert mkpts to COLMAP coordinates
                corrected_mkpts0 = adapt_mkpts_to_colmap(mkpts0)
                corrected_mkpts1 = adapt_mkpts_to_colmap(mkpts1)

                # Compute E matrix
                #TODO colon: choose RANSAC options
                # estimation_options = pycolmap.RANSACOptions()
                result_colmap = pycolmap.estimate_essential_matrix(corrected_mkpts0, corrected_mkpts1, camera0, camera1)

                # Extract R and t from the essential matrix
                R01_est = result_colmap['cam2_from_cam1'].rotation.matrix()
                t01_est = result_colmap['cam2_from_cam1'].translation

                # rotation error
                rot_err = rotation_error_deg(R01_colmap, R01_est)
                # translation error relative to the diameter of the submap
                trans_err = translation_error(t01_colmap, t01_est, d)

                logger.debug(f"{image0_name_extension.rsplit('.', 1)[0]}_{image1_name_extension.rsplit('.', 1)[0]} trans_error: {trans_err:.3f}")
                logger.debug(f"{image0_name_extension.rsplit('.', 1)[0]}_{image1_name_extension.rsplit('.', 1)[0]} rot_error: {rot_err:.3f}")
                
                submap_rot_errs.append(rot_err)
                submap_trans_errs.append(trans_err)

                pair_end_time = time.perf_counter() 

                # Build a dictionary for this pair
                pair_info = {
                    "image0": image0_name_extension,
                    "image1": image1_name_extension,
                    "mkpts0": len(mkpts0),
                    "mkpts1": len(mkpts1),
                    "kpts0": len(result['all_kpts0']),
                    "kpts1": len(result['all_kpts1']),
                    "extractor_time": extractor_time,
                    "filter_time": filter_time,
                    "matcher_time": match_time,
                    "total_pair_time": pair_end_time - pair_start_time,
                    "rot_error_deg":  rot_err,
                    "trans_error":    trans_err
                }
                pair_metrics.append(pair_info)

                # Push to global arrays to do sequence-level stats
                all_pairs_extractor_times.append(extractor_time)
                all_pairs_filter_times.append(filter_time)
                all_pairs_matcher_times.append(match_time)
                all_pairs_total_times.append(pair_info["total_pair_time"])
                all_rot_errs.append(rot_err)
                all_trans_errs.append(trans_err)
                
            except Exception as e:
                logger.error(f"Error processing pair {img0_path.name} and {img1_path.name}: {e}")
                logger.error(traceback.format_exc())
                # Stop and finalize current submap
                logger.warning(f"Stopping processing due to error. Will save data collected so far.")
                finalize_current_submap(submap, submap_rot_errs, submap_trans_errs, pair_metrics, 
                                        submap_start_time, thresholds_r, thresholds_t, all_submap_aas,
                                        all_pairs_extractor_times, all_pairs_filter_times, all_pairs_matcher_times,
                                        all_pairs_total_times, all_rot_errs, final_report, logger)
                
                save_report(final_report, seq, model_name, logger, output_report_dir, output_submap_dir,reason=f"error_in_pair_{img0_path.stem}_{img1_path.stem}")
                sys.exit(1)

        # After all pairs for this submap, compute submap-level stats
        if submap_rot_errs:  # Only if we have successful pairs
            # Compute submap-level RMSE for rotation error
            submap_rmse_rot = np.sqrt(np.mean(np.array(submap_rot_errs) ** 2))

            # Compute the average accuracy for this submap
            aa = compute_AA(submap_rot_errs, submap_trans_errs, thresholds_r, thresholds_t)
            all_submap_aas.append(aa)
            logger.debug(f"{submap} AA: {aa:.3f}")

            # Store submap info
            final_report["submaps"][submap] = {
                "pairs": pair_metrics,
                "average_extractor_time": float(np.mean([p["extractor_time"] for p in pair_metrics])),
                "average_filter_time": float(np.mean([p["filter_time"] for p in pair_metrics])),
                "average_matcher_time": float(np.mean([p["matcher_time"] for p in pair_metrics])),
                "average_total_time": float(np.mean([p["total_pair_time"] for p in pair_metrics])),
                "submap_total_time": time.perf_counter() - submap_start_time,
                "submap_rmse_rotation_deg": submap_rmse_rot,
                "submap_AA": aa,
                "N_pairs": len(pair_metrics)
            }

            logger.debug(f"{submap} total_time: {final_report['submaps'][submap]['submap_total_time']:.3f}s, AA: {aa:.3f}")
        else:
            logger.warning(f"No successful pairs processed for submap {submap}, skipping submap stats")

    # After all submaps, compute final mAA
    if len(all_submap_aas) > 0:
        mAA = float(np.mean(all_submap_aas))
    else:
        mAA = 0.0

    # Compute averages only if we have valid data
    if all_pairs_extractor_times:
        final_report["sequence_averages"]["extractor_time"] = float(np.mean(all_pairs_extractor_times))
    if all_pairs_filter_times:
        final_report["sequence_averages"]["filter_time"] = float(np.mean(all_pairs_filter_times))
    if all_pairs_matcher_times:
        final_report["sequence_averages"]["matcher_time"] = float(np.mean(all_pairs_matcher_times)) 
    if all_pairs_total_times:
        final_report["sequence_averages"]["total_pair_time"] = float(np.mean(all_pairs_total_times))

    # Example RMSE across entire sequence
    if len(all_rot_errs) > 0:
        seq_rmse_rot = float(np.sqrt(np.mean(np.array(all_rot_errs) ** 2)))
    else:
        seq_rmse_rot = 0.0

    final_report["sequence_averages"]["rmse_rotation_deg"] = seq_rmse_rot
    final_report["sequence_mAA"] = mAA
    final_report["processing_complete"] = True  # Mark processing as complete
    
    # Save final report at the end of normal execution
    save_report(final_report, seq, model_name, logger, output_report_dir, output_submap_dir,reason="complete")

except KeyboardInterrupt:
    logger.warning("Processing interrupted by user (Ctrl+C)")
    
    # Try to finalize current submap if we're in the middle of one
    current_submap = submaps[submaps.index(submap)] if 'submap' in locals() else None
    if current_submap and 'submap_start_time' in locals():
        logger.info(f"Finalizing data for current submap {current_submap} before exiting")
        finalize_current_submap(submap, submap_rot_errs, submap_trans_errs, pair_metrics, 
                                submap_start_time, thresholds_r, thresholds_t, all_submap_aas,
                                all_pairs_extractor_times, all_pairs_filter_times, all_pairs_matcher_times,
                                all_pairs_total_times, all_rot_errs, final_report, logger)    
    # Save the report with the data collected so far
    save_report(final_report, seq, model_name, logger, output_report_dir, output_submap_dir,reason="interrupted")
    sys.exit(1)
    
except Exception as e:
    logger.error(f"Critical error in main processing loop: {e}")
    logger.error(traceback.format_exc())
    
    # Try to finalize current submap if we're in the middle of one
    current_submap = submaps[submaps.index(submap)] if 'submap' in locals() else None
    if current_submap and 'submap_start_time' in locals():
        logger.info(f"Finalizing data for current submap {current_submap} before exiting")
        finalize_current_submap(submap, submap_rot_errs, submap_trans_errs, pair_metrics, 
                                submap_start_time, thresholds_r, thresholds_t, all_submap_aas,
                                all_pairs_extractor_times, all_pairs_filter_times, all_pairs_matcher_times,
                                all_pairs_total_times, all_rot_errs, final_report, logger)       
    # Save the report with the data collected so far
    save_report(final_report,seq, model_name, logger, output_report_dir, output_submap_dir,reason="error")
    sys.exit(1)  # Exit with error code instead of re-raising