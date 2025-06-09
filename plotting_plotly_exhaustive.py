import json
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np

# En tu caso, ajusta la ruta según corresponda
timestamp_name = '20250603_155724'
output_dir = Path(f'./output/plot_images/plotly/{timestamp_name}')
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1) Carga del JSON
# ------------------------------------------------------------------
submaps_dir = Path('./output/error_measurement/seq_001/superpoint-lg/20250603_155724/submaps')

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
# 2) Construimos un DataFrame con TODA la info necesaria en cada fila
# ------------------------------------------------------------------
rows = []
for submap_id, submap_report in submaps_dict.items():
    pairs = submap_report.get("submap_data", []).get("pair_metrics", [])
    
    # Podemos recuperar info global de submap
    total_img = submap_report.get("submap_data").get("results", None).get("Total_img",None)
    nimg_percentage = submap_report.get("submap_data").get("results", None).get("Nimg_percentage",None)
    mean_submap_p3d_error_pre_filter = submap_report.get("metadata").get("mean_submap_p3d_error_pre_filter", None)
    submap_p3d_errors_pre_filter = submap_report.get("metadata").get("submap_p3d_errors_pre_filter", None)

    for pair in pairs:


        mkpts = pair.get("mkpts", None)

        row = {
            'submap_id': submap_id,
            'image0': pair["image0"],
            'image1': pair["image1"],
            'rot_error': pair["rot_error"],
            'trans_error_deg': pair["trans_error_deg"],
            'trans_error_rel': pair["trans_error_rel"],
            't_colmap_norm': pair ["t_colmap_norm"],
            'mkpts': mkpts,
            'inliers': pair["inliers"],
            'Total_img': total_img,
            'Nimg_percentage': nimg_percentage,
            'parallax': pair["parallax"],
            "covis_score": pair["covis_score"],
            "mean_reprojection_error": pair["mean_reprojection_error"],
            "median_reprojection_error": pair["median_reprojection_error"],
            "std_reprojection_error": pair["std_reprojection_error"],
            "mean_submap_p3d_error_pre_filter": mean_submap_p3d_error_pre_filter,
            "submap_p3d_errors_pre_filter": submap_p3d_errors_pre_filter,
        }
        rows.append(row)

df = pd.DataFrame(rows)
print(df.head())

# # ------------------------------------------------------------------
# #Boxplot de error de rotación (incluyendo outliers y hover extra)
# # ------------------------------------------------------------------
# fig = px.box(
#     df,
#     x="submap_id",
#     y="rot_error",
#     points="all",  # "all" para mostrar todos los puntos
#     hover_data=["image0", "image1", "mkpts","inliers", "Total_img", "Nimg_percentage", "parallax", "covis_score", "trans_error_deg", "trans_error_rel", "t_colmap_norm",
#                 "mean_reprojection_error", "median_reprojection_error", "std_reprojection_error", "mean_submap_p3d_error_pre_filter", "median_submap_p3d_error_pre_filter"],
# )


# fig.update_layout(
#     title="Distribution of Rotation Error by Submap",
#     yaxis_title="Rot. Error (deg)"
# )

# rotation_file = output_dir / "rot_error_boxplot.html"
# fig.write_html(rotation_file)
# print(f"Saved rotation boxplot to {rotation_file}")



# # ------------------------------------------------------------------
# # Correlation plot: Inliers vs. Rotation Error
# # ------------------------------------------------------------------
# fig = px.scatter(
#     df,
#     x="inliers",
#     y="rot_error",
#     color="submap_id",
#     hover_data=["image0", "image1", "mkpts", "Total_img", "Nimg_percentage", "parallax", "covis_score", "trans_error_deg", "trans_error_rel", "t_colmap_norm",
#                  "mean_reprojection_error", "median_reprojection_error", "std_reprojection_error", "mean_submap_p3d_error_pre_filter", "median_submap_p3d_error_pre_filter"]
# )

# fig.update_layout(
#     title="Correlation between Number of Inliers and Rotation Error",
#     xaxis_title="Number of Inliers",
#     yaxis_title="Rotation Error (deg)"
# )

# inliers_rot_file = output_dir / "inliers_rot_error_correlation.html"
# fig.write_html(inliers_rot_file)
# print(f"Saved inliers-rotation error correlation plot to {inliers_rot_file}")


# # ------------------------------------------------------------------
# # Box plot: P3d Error Pre-Filter
# # ------------------------------------------------------------------

# fig = px.box(
#     df,
#     x="submap_id",
#     y="submap_p3d_errors_pre_filter",
#     points="all",  # "all" para mostrar todos los puntos
# #     hover_data=["image0", "image1", "mkpts", "inliers", "Total_img", "Nimg_percentage", "parallax", "covis_score", "trans_error_deg", "trans_error_rel", "t_colmap_norm",
# #                  "mean_reprojection_error", "median_reprojection_error", "std_reprojection_error", "mean_submap_p3d_error_pre_filter", "median_submap_p3d_error_pre_filter"],
#  )

# fig.update_layout(
#     title="Distribution of P3D Errors Pre-Filter by Submap",
#     yaxis_title="P3D Errors Pre-Filter"
# )
# p3d_error_file = output_dir / "p3d_errors_pre_filter_boxplot.html"
# fig.write_html(p3d_error_file)
# print(f"Saved P3D errors pre-filter boxplot to {p3d_error_file}")


# Optimize box plot for speed with large datasets
fig = px.box(
    df,
    x="submap_id",
    y="submap_p3d_errors_pre_filter",
    points="outliers",  # Only show outliers instead of all points
    # hover_data limited to essential information
    hover_data=["mean_submap_p3d_error_pre_filter"]
)

# Add quartile information explicitly in the layout
fig.update_layout(
    title="Distribution of P3D Errors Pre-Filter by Submap",
    yaxis_title="P3D Errors Pre-Filter",
    boxmode="group",
    # Limit the y-axis range if there are extreme outliers
    # yaxis_range=[0, df["submap_p3d_errors_pre_filter"].quantile(0.99)]
)

# Use a more efficient renderer
fig.update_layout(template="plotly")

# Set to save with lower resolution for faster rendering
p3d_error_file = output_dir / "p3d_errors_pre_filter_boxplot.html"
fig.write_html(p3d_error_file, include_plotlyjs='cdn')  # Use CDN for plotly.js
print(f"Saved P3D errors pre-filter boxplot to {p3d_error_file}")