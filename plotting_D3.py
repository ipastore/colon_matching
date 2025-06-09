import json
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np

# En tu caso, ajusta la ruta según corresponda
timestamp_name = '20250608_185951'
output_dir = Path(f'./output/plot_images/D3/{timestamp_name}')
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1) Load JSON files from submaps directory
# ------------------------------------------------------------------
submaps_dir = Path('./output/error_measurement/seq_001/superpoint-lg/20250608_185951_parallax4_distance5/submaps')
# Find all submap JSON files
submap_files = list(submaps_dir.glob(f"submap_*.json"))

# Make submaps_dict from submap_files (JSON)
submaps_dict = {}
for submap_file in submap_files:
    with open(submap_file, 'r') as f:
        submap_report = json.load(f)
    
        submap_name = submap_report.get("metadata", {}).get("submap")

        # Add to submaps_dict
        submaps_dict[submap_name] = {
            "metadata": submap_report.get("metadata", []),
            "submap_data": submap_report.get("submap_data", []),
        }

    # ------------------------------------------------------------------
    # 2) Flatten the data into a list of dictionaries
    # ------------------------------------------------------------------
    records = []

    for submap_id, submap_report in submaps_dict.items():
        pairs = submap_report.get("submap_data", []).get("pair_metrics", [])
        
        # Podemos recuperar info global de submap
        total_img = submap_report.get("submap_data").get("results", None).get("Total_img",None)
        nimg_percentage = submap_report.get("submap_data").get("results", None).get("Nimg_percentage",None)
        mAA = submap_report.get("submap_data").get("results", None).get("submap_mAA", None)
        mean_submap_p3d_error_pre_filter = submap_report.get("metadata").get("mean_submap_p3d_error_pre_filter", None)
        median_submap_p3d_error_pre_filter = submap_report.get("metadata").get("median_submap_p3d_error_pre_filter", None)
        submap_p3d_errors_pre_filter = submap_report.get("metadata").get("submap_p3d_errors_pre_filter", None)
        model_name = submap_report.get("metadata").get("model_name", None)
        sequence = submap_report.get("metadata").get("sequence", None)

        for pair in pairs:
            mkpts = pair.get("mkpts", None)

            row = {
                'submap_id': submap_id,
                'model_name': model_name,
                'sequence': sequence,
                'image0': pair["image0"],
                'image1': pair["image1"],
                'rot_error': pair["rot_error"],
                'trans_error_deg': pair["trans_error_deg"],
                'trans_error_rel': pair["trans_error_rel"],
                't_colmap_norm': pair ["t_colmap_norm"],
                'mkpts': mkpts,
                'inliers': pair["inliers"],
                'Total_img': total_img,
                'mAA': mAA,
                'Nimg_percentage': nimg_percentage,
                'parallax': pair["parallax"],
                "covis_score": pair["covis_score"],
                "mean_reprojection_error": pair["mean_reprojection_error"],
                "median_reprojection_error": pair["median_reprojection_error"],
                "std_reprojection_error": pair["std_reprojection_error"],
                "mean_submap_p3d_error_pre_filter": mean_submap_p3d_error_pre_filter,
                "median_submap_p3d_error_pre_filter": median_submap_p3d_error_pre_filter,
                "distance_bw_frames": pair["distance_bw_frames"],
                "n_3d_colmap_points": pair["n_3d_colmap_points"] if "n_3d_colmap_points" in pair else None,
                # "submap_p3d_errors_pre_filter": submap_p3d_errors_pre_filter,

            }
            records.append(row)


# ------------------------------------------------------------------
# 3)Load HTML template
# ------------------------------------------------------------------


# Read the HTML template
with open('plotting_D3_template.html', 'r') as f:
    html_template = f.read()

# ------------------------------------------------------------------
# 4) Save final HTML
# ------------------------------------------------------------------

import os

# Replace placeholder
html_final = html_template.replace("{{DATA_JSON}}", json.dumps(records))
output_path = output_dir / f"D3_{timestamp_name}.html"

# Save
with open(output_path, 'w') as f:
    f.write(html_final)

print(f"Dashboard saved to {output_path}")
