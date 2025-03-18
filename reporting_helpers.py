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

# # Function to save the report in case of early stopping
# def save_report(final_report, seq, model_name, logger, 
#                 output_report_dir, output_submap_dir, reason="normal"):
#     """Save the final report with current data even if processing was interrupted"""
#     # Create timestamp for unique filenames
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # Add reason for report generation
#     final_report["report_status"] = reason

#     # Save as JSON
#     json_filename = output_report_dir / f"seq_full_report_{timestamp}_{reason}.json"
#     with open(json_filename, 'w') as json_file:
#         json.dump(final_report, json_file, default=convert_to_serializable, indent=4)
#     logger.info(f"Final report saved as JSON to: {json_filename}")

#     # Save summary metrics as CSV
#     csv_filename = output_report_dir / f"seq_summary_report{timestamp}_{reason}.csv"
#     with open(csv_filename, 'w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
        
#         # Write header
#         csv_writer.writerow(['Sequence', 'Model', 'mAA', 'RMSE_Rot_deg',  
#                             'Avg_Extractor_Time', 'Avg_Filter_Time', 'Avg_Matcher_Time', 
#                             'Avg_Total_Time', 'avg_Nimg_percentage','Status'])
        
#         # Write summary row
#         csv_writer.writerow([
#             seq,
#             model_name,
#             final_report["sequence_mAA"],
#             final_report["sequence_averages"].get("rmse_rotation_deg", 0),
#             final_report["sequence_averages"].get("extractor_time", 0),
#             final_report["sequence_averages"].get("filter_time", 0),
#             final_report["sequence_averages"].get("matcher_time", 0),
#             final_report["sequence_averages"].get("total_pair_time", 0),
#             final_report["sequence_averages"].get("avg_Nimg_percentage", 0),
#             reason
#         ])
    
#     # Only save per-submap metrics if we have submaps data
#     if final_report["submaps"]:
#         last_submap = list(final_report["submaps"].keys())[-1]

#         # Save per-submap metrics as CSV
#         submap_csv_filename = output_submap_dir / f"submap_report_{timestamp}_{reason}.csv"

#         with open(submap_csv_filename, 'w', newline='') as csv_file:
#             csv_writer = csv.writer(csv_file)
            
#             # Write header
#             csv_writer.writerow(['Sequence', 'Model', 'Submap', 'AA', 'RMSE_Rot_deg', 
#                                 'Avg_Extractor_Time', 'Avg_Filter_Time', 'Avg_Matcher_Time', 
#                                 'Avg_Total_Time', 'Total_img', 'N_img'])
            
#             # Write a row for each submap
#             for submap_name, submap_data in final_report["submaps"].items():
#                 csv_writer.writerow([
#                     seq,
#                     model_name,
#                     submap_name,
#                     submap_data.get("submap_AA", 0),
#                     submap_data.get("submap_rmse_rotation_deg", 0),
#                     submap_data.get("average_extractor_time", 0),
#                     submap_data.get("average_filter_time", 0),
#                     submap_data.get("average_matcher_time", 0),
#                     submap_data.get("average_total_time", 0),
#                     submap_data.get("Total_img", 0),
#                     submap_data.get("Nimg_percentage", 0)
#                 ])
        
#         logger.info(f"Summary metrics saved as CSV to: {csv_filename}")
#         logger.info(f"Per-submap metrics saved as CSV to: {submap_csv_filename}")

#     logger.info(f"======= FINAL REPORT ({reason}) for sequence {seq} =======")
#     return json_filename, csv_filename

# # Function to compute and save current submap statistics before exiting
# def finalize_current_submap(submap, submap_rot_errs, submap_trans_errs, pair_metrics,
#                              submap_start_time, thresholds_r, thresholds_t, all_submap_aas,
#                              all_pairs_extractor_times, all_pairs_filter_times, all_pairs_matcher_times,
#                              all_pairs_total_times, all_rot_errs, final_report,  registered_images, 
#                              total_images_submap, all_registered_images, logger):
#     """Compute and save statistics for the current submap before exiting due to error or interrupt"""
#     if not submap_rot_errs:  # If no successful pairs were processed for this submap
#         logger.warning(f"No successful pairs processed for submap {submap}, cannot compute stats")
#         return
    
#     submap_end_time = time.perf_counter()
#     submap_total_time = submap_end_time - submap_start_time
    
#     # Compute submap-level RMSE for rotation error
#     submap_rmse_rot = np.sqrt(np.mean(np.array(submap_rot_errs) ** 2))

#     # Compute the average accuracy for this submap
#     aa = compute_AA(submap_rot_errs, submap_trans_errs, thresholds_r, thresholds_t)

#     #TODO colon_matching: skip all_xxx variables
#     all_submap_aas.append(aa)


#     # Calculate Nimg for the submap
#     Nimg_submap = len(registered_images)
#     Nimg_percentage = (Nimg_submap / total_images_submap) * 100
    
#     #TODO colon_matching: skip all_xxx variables
#     all_registered_images.append(Nimg_percentage)

#     # Store submap info
#     final_report["submaps"][submap] = {
#         "pairs": pair_metrics,
#         "average_extractor_time": float(np.mean([p["extractor_time"] for p in pair_metrics])) if pair_metrics else 0,
#         "average_filter_time": float(np.mean([p["filter_time"] for p in pair_metrics])) if pair_metrics else 0,
#         "average_matcher_time": float(np.mean([p["matcher_time"] for p in pair_metrics])) if pair_metrics else 0,
#         "average_total_time": float(np.mean([p["total_pair_time"] for p in pair_metrics])) if pair_metrics else 0,
#         "submap_total_time": submap_total_time,
#         "submap_rmse_rotation_deg": submap_rmse_rot,
#         "submap_AA": aa,
#         "Total_img": total_images_submap,
#         "Nimg_percentage": Nimg_percentage,
#     }

#     #TODO colon_matching: skip all_xxx variables
#     # Append to global arrays
#     all_pairs_extractor_times.extend([p["extractor_time"] for p in pair_metrics])
#     all_pairs_filter_times.extend([p["filter_time"] for p in pair_metrics])
#     all_pairs_matcher_times.extend([p["matcher_time"] for p in pair_metrics])
#     all_pairs_total_times.extend([p["total_pair_time"] for p in pair_metrics])
#     all_rot_errs.extend(submap_rot_errs)
    
#     #TODO colon_matching: skip all_xxx variables
#     # Calculate sequence statistics with current data
#     if len(all_submap_aas) > 0:
#         final_report["sequence_mAA"] = float(np.mean(all_submap_aas))
    
#     #TODO colon_matching: skip all_xxx variables
#     if all_pairs_extractor_times:
#         final_report["sequence_averages"]["extractor_time"] = float(np.mean(all_pairs_extractor_times))
#     if all_pairs_filter_times:
#         final_report["sequence_averages"]["filter_time"] = float(np.mean(all_pairs_filter_times))
#     if all_pairs_matcher_times:
#         final_report["sequence_averages"]["matcher_time"] = float(np.mean(all_pairs_matcher_times)) 
#     if all_pairs_total_times:
#         final_report["sequence_averages"]["total_pair_time"] = float(np.mean(all_pairs_total_times))

#     # Example RMSE across entire sequence
#     if len(all_rot_errs) > 0:
#         seq_rmse_rot = float(np.sqrt(np.mean(np.array(all_rot_errs) ** 2)))
#         final_report["sequence_averages"]["rmse_rotation_deg"] = seq_rmse_rot

# #TODO: take all the skipped functions in the finalize current_submap and put them here or in a related function
# def compute_final_sequence_stats(final_report, all_submap_aas, all_pairs_extractor_times, all_pairs_filter_times, all_pairs_matcher_times, all_pairs_total_times, all_rot_errs, all_registered_images):
#     """Compute final sequence-level statistics and update the final report."""
#     if len(all_submap_aas) > 0:
#         mAA = float(np.mean(all_submap_aas))
#     else:
#         mAA = 0.0

#     # Compute averages only if we have valid data
#     if all_pairs_extractor_times:
#         final_report["sequence_averages"]["extractor_time"] = float(np.mean(all_pairs_extractor_times))
#     if all_pairs_filter_times:
#         final_report["sequence_averages"]["filter_time"] = float(np.mean(all_pairs_filter_times))
#     if all_pairs_matcher_times:
#         final_report["sequence_averages"]["matcher_time"] = float(np.mean(all_pairs_matcher_times)) 
#     if all_pairs_total_times:
#         final_report["sequence_averages"]["total_pair_time"] = float(np.mean(all_pairs_total_times))

#     # Example RMSE across entire sequence
#     if len(all_rot_errs) > 0:
#         seq_rmse_rot = float(np.sqrt(np.mean(np.array(all_rot_errs) ** 2)))
#     else:
#         seq_rmse_rot = 0.0

#     final_report["sequence_averages"]["rmse_rotation_deg"] = seq_rmse_rot
#     final_report["sequence_mAA"] = mAA

#         # Calculate average Nimg percentage across all submaps
#     if all_registered_images:
#         avg_Nimg_percentage = float(np.mean(all_registered_images))
#     else:
#         avg_Nimg_percentage = 0.0
#     final_report["sequence_averages"]["avg_Nimg_percentage"] = avg_Nimg_percentage
#     final_report["processing_complete"] = True  # Mark processing as complete



def save_submap_report(submap_data, seq, submap_name, model_name, output_dir, logger, reason="normal"):
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

    # Calculate Nimg for the submap
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