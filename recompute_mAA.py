#This script should load a submap JSON file, extract the relevant data, and compute the mAA for each submap, given another thresholds

import json
import os
from pathlib import Path
from datetime import datetime
import argparse
import logging
import numpy as np
from colmap_opencv_helpers import compute_mAA
from reporting_helpers import save_submap_report
from types import SimpleNamespace
from typing import List, Dict, Any

############################### CONFIG ###############################
#Define new thresholds
thresholds_r = [0.5, 1.0, 2.0, 5.0]
thresholds_t = [0.5, 1.0, 2.0, 5.0]
# Define the path to root dir with all the submaps
submaps_dir = Path("./output/error_measurement/seq_001/superpoint-lg/20250526_124928/submaps")

# Define the output directory (one level up)
output_dir = submaps_dir.parent

def main():

    # Get all the submap JSON files
    submap_files = list(submaps_dir.glob("*.json"))
    
    if not submap_files:
        print("No submap JSON files found in the specified directory.")
        return
  
    # Process each submap file
    for submap_file in submap_files:
        
        submap_rot_errors = []
        submap_trans_errors = []
        
        with open(submap_file, 'r') as f:
            data = json.load(f)

        # Extract relevant data from the JSON file
        pair_metrics = data["submap_data"]["pair_metrics"]

        # Append all submap errors
        for pair in pair_metrics:
            submap_rot_errors.append(pair["rot_error"])
            submap_trans_errors.append(pair["trans_error_deg"])
        
        # Old mAA
        old_mAA = data["submap_data"]["results"]["submap_mAA"]

        # Compute the new mAA for the submap
        new_mAA = compute_mAA(submap_rot_errors, submap_trans_errors, thresholds_r, thresholds_t)

        # Update new_mAA in the submap data, results. Also update thresholds in metadata
        data["submap_data"]["results"]["submap_mAA"] = new_mAA
        data["metadata"]["thresholds_r"] = thresholds_r
        data["metadata"]["thresholds_t"] = thresholds_t
        data["metadata"]["report_status"] = "recomputed"
        
        # New timestamp to avoid overwriting
        new_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # # Save updated report
        save_submap_report(data["submap_data"], data["metadata"]["sequence"],data["metadata"]["submap"],
                           data["metadata"]["model_name"], new_timestamp, data["metadata"], submaps_dir)

        print(f"Old mAA: {old_mAA}")
        print(f"Updated mAA for {submap_file.name}: {new_mAA}")

    return 

if __name__ == "__main__":
    main()