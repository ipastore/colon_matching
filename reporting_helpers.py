import json
import csv
from types import SimpleNamespace
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from colmap_opencv_helpers import compute_AA

# Custom serialization function to handle numpy arrays and SimpleNamespace objects
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, SimpleNamespace):
        return obj.__dict__
    return obj

def save_submap_report(submap_data, seq, submap_name, model_name, output_dir, logger, metadata = None, reason = "normal"):
    """Save report data for a single submap to a JSON file"""
    # Create timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filename with sequence, submap name and timestamp
    json_filename = output_dir / f"submap_{seq}_{submap_name}_{timestamp}.json"
    
    # Add metadata to the submap report
    report = {
        "sequence": seq,
        "submap": submap_name,
        "model_name": model_name,
        "timestamp": timestamp,
        "report_status": reason,
        "data": submap_data
    }

    #Update report with additional metadata if provided
    if metadata:
        report.update(metadata)
    
    # Save as JSON
    with open(json_filename, 'w') as json_file:
        json.dump(report, json_file, default=convert_to_serializable, indent=4)
    
    logger.info(f"Submap report saved as JSON to: {json_filename}")
    return json_filename

def finalize_current_submap(submap, submap_rot_errs, submap_trans_errs, pair_metrics,
                           submap_start_time, thresholds_r, thresholds_t, 
                           registered_images, total_images_submap, logger):
    """Compute statistics for the current submap only - no global accumulation"""
    if not submap_rot_errs:  # If no successful pairs were processed for this submap
        logger.warning(f"No successful pairs processed for submap {submap}, cannot compute stats")
        return {}
    
    submap_end_time = time.perf_counter()
    submap_total_time = submap_end_time - submap_start_time
    
    # Compute submap-level RMSE for rotation error
    submap_rmse_rot = np.sqrt(np.mean(np.array(submap_rot_errs) ** 2))

    # Compute the average accuracy for this submap
    aa = compute_AA(submap_rot_errs, submap_trans_errs, thresholds_r, thresholds_t)

    # Calculate Nimg for the submap. Nimg is the number of PAIR of images that were registered over ALL PAIRS of images.
    Nimg_submap = len(registered_images)
    Nimg_percentage = (Nimg_submap / total_images_submap) * 100 if total_images_submap > 0 else 0
    
    # Create submap data dictionary
    submap_data = {
        "pairs": pair_metrics,
        "average_extractor_time": float(np.mean([p["extractor_time"] for p in pair_metrics])) if pair_metrics else 0,
        "average_filter_time": float(np.mean([p["filter_time"] for p in pair_metrics])) if pair_metrics else 0,
        "average_matcher_time": float(np.mean([p["matcher_time"] for p in pair_metrics])) if pair_metrics else 0,
        "average_total_time": float(np.mean([p["total_pair_time"] for p in pair_metrics])) if pair_metrics else 0,
        "submap_total_time": submap_total_time,
        "submap_rmse_rotation_deg": submap_rmse_rot,
        "submap_AA": aa,
        "Total_img": total_images_submap,
        "Nimg_percentage": Nimg_percentage,
    }
    
    return submap_data

#TODO colon: convert this function to be used in a separate script for creating sequence reports.
def create_sequence_report(seq, model_name, submaps_dir, output_dir, logger, metadata=None):
    """
    Create a sequence report by reading individual submap JSONs.
    This function can be called separately after all submaps are processed.
    """
    # Create the output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find all submap JSON files
    submap_files = list(submaps_dir.glob(f"submap_{seq}_*.json"))
    
    if not submap_files:
        logger.warning(f"No submap files found in {submaps_dir}")
        return None
    
    # Create a sequence report structure
    sequence_report = {
        "sequence_name": seq,
        "model_name": model_name,
        "timestamp": timestamp,
        "submaps": {},
        "sequence_averages": {},
        "sequence_mAA": 0.0,
        "processing_complete": True
    }
    
    # Add any additional metadata
    if metadata:
        sequence_report.update(metadata)
    
    # Variables for accumulating sequence stats
    all_submap_aas = []
    all_extractor_times = []
    all_filter_times = []
    all_matcher_times = []
    all_total_times = []
    all_rot_errs = []
    all_registered_images = []
    
    # Load each submap and include its data
    for submap_file in submap_files:
        with open(submap_file, 'r') as f:
            submap_data = json.load(f)
            
        submap_name = submap_data.get("submap")
        submap_content = submap_data.get("data", {})
        
        # Add to sequence report
        sequence_report["submaps"][submap_name] = submap_content
        
        # Accumulate stats for sequence averages
        all_submap_aas.append(submap_content.get("submap_AA", 0))
        all_extractor_times.append(submap_content.get("average_extractor_time", 0))
        all_filter_times.append(submap_content.get("average_filter_time", 0))
        all_matcher_times.append(submap_content.get("average_matcher_time", 0))
        all_total_times.append(submap_content.get("average_total_time", 0))
        all_registered_images.append(submap_content.get("Nimg_percentage", 0))
        
        # Accumulate rotation errors from all pairs
        for pair in submap_content.get("pairs", []):
            all_rot_errs.append(pair.get("rot_error_deg", 0))
    
    # Compute sequence-level statistics
    if all_submap_aas:
        sequence_report["sequence_mAA"] = float(np.mean(all_submap_aas))
    
    sequence_report["sequence_averages"] = {
        "extractor_time": float(np.mean(all_extractor_times)) if all_extractor_times else 0,
        "filter_time": float(np.mean(all_filter_times)) if all_filter_times else 0,
        "matcher_time": float(np.mean(all_matcher_times)) if all_matcher_times else 0,
        "total_pair_time": float(np.mean(all_total_times)) if all_total_times else 0,
        "avg_Nimg_percentage": float(np.mean(all_registered_images)) if all_registered_images else 0,
        "rmse_rotation_deg": float(np.sqrt(np.mean(np.array(all_rot_errs) ** 2))) if all_rot_errs else 0,
    }
    
    # Save the sequence report
    sequence_file = output_dir / f"sequence_{seq}_{timestamp}.json"
    with open(sequence_file, 'w') as f:
        json.dump(sequence_report, f, default=convert_to_serializable, indent=4)
        
    logger.info(f"Sequence report saved to {sequence_file}")
    return sequence_file

def progress_bar(logger,progress, total, img0_path, img1_path):
    percent = progress / total
    bar_length = 30
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    logger.info(f'[{bar}] {int(percent*100)}% ({progress}/{total}) - {img0_path.stem} & {img1_path.stem}')
    return