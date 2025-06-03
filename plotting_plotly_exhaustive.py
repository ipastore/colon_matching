import json
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np

# En tu caso, ajusta la ruta según corresponda
timestamp_name = '20250602_230230'
output_dir = Path(f'./output/plot_images/plotly/{timestamp_name}')
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1) Carga del JSON
# ------------------------------------------------------------------
submaps_dir = Path('./output/error_measurement/seq_001_toy/superpoint-lg/20250602_230230/submaps')

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
    median_submap_p3d_error_pre_filter = submap_report.get("metadata").get("median_submap_p3d_error_pre_filter", None)

    for pair in pairs:
        # if pair["parallax"] < 3 or pair["inliers"] < 50 or pair["t_colmap_norm"] < 0.5:

        #     continue

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
            "median_submap_p3d_error_pre_filter": median_submap_p3d_error_pre_filter,
        }
        rows.append(row)

df = pd.DataFrame(rows)
print(df.head())

# ------------------------------------------------------------------
# 3) Boxplot de error de rotación (incluyendo outliers y hover extra)
# ------------------------------------------------------------------
fig = px.box(
    df,
    x="submap_id",
    y="rot_error",
    points="all",  # "all" para mostrar todos los puntos
    hover_data=["image0", "image1", "mkpts","inliers", "Total_img", "Nimg_percentage", "parallax", "covis_score", "trans_error_deg", "trans_error_rel", "t_colmap_norm",
                "mean_reprojection_error", "median_reprojection_error", "std_reprojection_error", "mean_submap_p3d_error_pre_filter", "median_submap_p3d_error_pre_filter"],
)


fig.update_layout(
    title="Distribution of Rotation Error by Submap",
    yaxis_title="Rot. Error (deg)"
)

rotation_file = output_dir / "rot_error_boxplot.html"
fig.write_html(rotation_file)
print(f"Saved rotation boxplot to {rotation_file}")

# ------------------------------------------------------------------
# 4) Boxplot de error de traslación
# ------------------------------------------------------------------
fig = px.box(
    df,
    x="submap_id",
    y="trans_error_deg",
    points="all",
    hover_data=["image0", "image1", "mkpts", "inliers", "Total_img", "Nimg_percentage", "parallax", "covis_score", "rot_error", "trans_error_rel", "t_colmap_norm",
                 "mean_reprojection_error", "median_reprojection_error", "std_reprojection_error", "mean_submap_p3d_error_pre_filter", "median_submap_p3d_error_pre_filter"],
)
# Si quieres ampliar el rango (p.ej. hasta 5), haz:
# fig.update_yaxes(range=[0, 5])

fig.update_layout(
    title="Distribution of error of Translation (degrees between translation vectors) by Submap",
    yaxis_title="Trans. Error (deg)"
)

translation_file = output_dir / "trans_error_deg_boxplot.html"
fig.write_html(translation_file)
print(f"Saved translation boxplot to {translation_file}")

# ------------------------------------------------------------------
# 5) Histograma de rotaciones
#    Ajustar nbins y rango
# ------------------------------------------------------------------
fig = px.histogram(
    df,
    x="rot_error",
    color="submap_id",
    marginal="box",
    hover_data=["image0", "image1", "mkpts","inliers","Total_img", "Nimg_percentage", "parallax", "covis_score", "trans_error_deg", "trans_error_rel", "t_colmap_norm",
                 "mean_reprojection_error", "median_reprojection_error", "std_reprojection_error", "mean_submap_p3d_error_pre_filter", "median_submap_p3d_error_pre_filter"],
    nbins=1800,         # Aumentar bins
    # range_x=[0, 180]  # Si quieres ver hasta 180°
)
fig.update_layout(
    title="Distribution of Rotation Error by Submap",
    xaxis_title="Rot. Error (deg)"
)

rotation_hist_file = output_dir / "rot_error_histogram.html"
fig.write_html(rotation_hist_file)
print(f"Saved rotation histogram to {rotation_hist_file}")

# ------------------------------------------------------------------
# 6) Histograma de traslaciones
# ------------------------------------------------------------------
fig = px.histogram(
    df,
    x="trans_error_deg",
    color="submap_id",
    marginal="box",
    hover_data=["image0", "image1", "mkpts", "inliers","Total_img", "Nimg_percentage", "parallax", "covis_score", "rot_error", "trans_error_rel", "t_colmap_norm", 
                 "mean_reprojection_error", "median_reprojection_error", "std_reprojection_error", "mean_submap_p3d_error_pre_filter", "median_submap_p3d_error_pre_filter"],
    nbins=1800,         # Ajusta según el rango y lo que desees
    # range_x=[0, 5]    # Si quieres llegar hasta 5
)
fig.update_layout(
    title="Distribution of Translation Error (degrees between translation vectors) by Submap",
    xaxis_title="Trans. Error"
)

translation_hist_file = output_dir / "trans_error_deg_histogram.html"
fig.write_html(translation_hist_file)
print(f"Saved translation histogram to {translation_hist_file}")

# ------------------------------------------------------------------
# 7) Correlation plot: Parallax vs. Translation Error
# ------------------------------------------------------------------
fig = px.scatter(
    df,
    x="parallax",
    y="trans_error_deg",
    color="submap_id",
    hover_data=["image0", "image1", "mkpts","inliers", "Total_img", "Nimg_percentage", "covis_score", "rot_error", "trans_error_rel", "t_colmap_norm",
                 "mean_reprojection_error", "median_reprojection_error", "std_reprojection_error", "mean_submap_p3d_error_pre_filter", "median_submap_p3d_error_pre_filter"]
)

fig.update_layout(
    title="Correlation between Parallax and Translation Error",
    xaxis_title="Parallax",
    yaxis_title="Translation Error (deg)"
)

# Add ticks at intervals of 1 instead of 5
fig.update_xaxes(dtick=1)  # This sets tick intervals to 1

parallax_trans_file = output_dir / "parallax_trans_correlation.html"
fig.write_html(parallax_trans_file)
print(f"Saved parallax-translation correlation plot to {parallax_trans_file}")

# ------------------------------------------------------------------
# 8) Correlation plot: Parallax vs. Rotation Error
# ------------------------------------------------------------------
fig = px.scatter(
    df,
    x="parallax",
    y="rot_error",
    color="submap_id",
    hover_data=["image0", "image1", "mkpts","inliers", "Total_img", "Nimg_percentage", "covis_score", "trans_error_deg", "trans_error_rel", "t_colmap_norm",
                 "mean_reprojection_error", "median_reprojection_error", "std_reprojection_error", "mean_submap_p3d_error_pre_filter", "median_submap_p3d_error_pre_filter"]
)

fig.update_layout(
    title="Correlation between Parallax and Rotation Error",
    xaxis_title="Parallax",
    yaxis_title="Rotation Error (deg)"
)

# Add ticks at intervals of 1 instead of 5
fig.update_xaxes(dtick=1)  # This sets tick intervals to 1

parallax_rot_file = output_dir / "parallax_rot_correlation.html"
fig.write_html(parallax_rot_file)
print(f"Saved parallax-rotation correlation plot to {parallax_rot_file}")


# ------------------------------------------------------------------
# 12) Correlation plot: Inliers vs. Translation Error (degrees)
# ------------------------------------------------------------------
fig = px.scatter(
    df,
    x="inliers",
    y="trans_error_deg",
    color="submap_id",
    hover_data=["image0", "image1", "mkpts", "Total_img", "Nimg_percentage", "parallax", "covis_score", "rot_error", "trans_error_rel", "t_colmap_norm",
                 "mean_reprojection_error", "median_reprojection_error", "std_reprojection_error", "mean_submap_p3d_error_pre_filter", "median_submap_p3d_error_pre_filter"]
)

fig.update_layout(
    title="Correlation between Number of Inliers and Translation Error",
    xaxis_title="Number of Inliers",
    yaxis_title="Translation Error (deg)"
)

inliers_trans_file = output_dir / "inliers_trans_deg_correlation.html"
fig.write_html(inliers_trans_file)
print(f"Saved inliers-translation error correlation plot to {inliers_trans_file}")


# ------------------------------------------------------------------
# 14) Correlation plot: Inliers vs. Rotation Error
# ------------------------------------------------------------------
fig = px.scatter(
    df,
    x="inliers",
    y="rot_error",
    color="submap_id",
    hover_data=["image0", "image1", "mkpts", "Total_img", "Nimg_percentage", "parallax", "covis_score", "trans_error_deg", "trans_error_rel", "t_colmap_norm",
                 "mean_reprojection_error", "median_reprojection_error", "std_reprojection_error", "mean_submap_p3d_error_pre_filter", "median_submap_p3d_error_pre_filter"]
)

fig.update_layout(
    title="Correlation between Number of Inliers and Rotation Error",
    xaxis_title="Number of Inliers",
    yaxis_title="Rotation Error (deg)"
)

inliers_rot_file = output_dir / "inliers_rot_error_correlation.html"
fig.write_html(inliers_rot_file)
print(f"Saved inliers-rotation error correlation plot to {inliers_rot_file}")

import plotly.express as px
import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# Your DataFrame 'df' already loaded
# ---------------------------------------------------

# Define thresholds for min_inliers and min_parallax
min_inliers_values = list(range(0, int(df['inliers'].max()) + 50, 10))   # Every 50
min_parallax_values = list(range(0, int(df['parallax'].max()) + 1, 1))  # Every 5 degrees

# Generate all combinations of (min_inliers, min_parallax)
filter_combinations = []
for inl in min_inliers_values:
    for par in min_parallax_values:
        filter_combinations.append((inl, par))

# Labels for dropdown
labels = [f"≥ {inl} inliers & ≥ {par}° parallax" for inl, par in filter_combinations]

# Precompute filtered datasets
filtered_data = []
for min_inl, min_par in filter_combinations:
    filtered_df = df[(df['inliers'] >= min_inl) & (df['parallax'] >= min_par)]
    filtered_data.append(filtered_df)

# Initialize the figure with the first filter
fig = px.box(
    filtered_data[0],
    x="submap_id",
    y="rot_error",
    points="all",
    hover_data=[
        "image0", "image1", "mkpts", "inliers", "Total_img", "Nimg_percentage",
        "parallax", "covis_score", "trans_error_deg", "trans_error_rel", "t_colmap_norm",
        "mean_reprojection_error", "median_reprojection_error", "std_reprojection_error",
        "mean_submap_p3d_error_pre_filter", "median_submap_p3d_error_pre_filter"
    ]
)

# Create a single dropdown with all (min_inliers, min_parallax) combinations
dropdown_buttons = [
    {
        "label": label,
        "method": "update",
        "args": [
            {
                "x": [filtered_data[i]["submap_id"]],
                "y": [filtered_data[i]["rot_error"]],
            },
            {
                "title": f"Rotation Error - {label}",
                "yaxis": {"title": "Rotation Error (deg)"},
                "xaxis": {"title": "Submap ID"},
            }
        ],
    }
    for i, label in enumerate(labels)
]

fig.update_layout(
    title="Rotation Error - ≥ 0 inliers & ≥ 0° parallax",
    yaxis_title="Rotation Error (deg)",
    xaxis_title="Submap ID",
    updatemenus=[
        {
            "buttons": dropdown_buttons,
            "direction": "down",
            "showactive": True,
            "x": 1.2,
            "y": 1.2,
            "xanchor": "left",
            "yanchor": "top",
        }
    ]
)

# Save to HTML
rotation_file = Path('./output/error_measurement/seq_001_toy/superpoint-lg/20250602_230230/') / "rot_error_boxplot_mininliers_minparallax_filter.html"
fig.write_html(rotation_file)
print(f"Saved dynamic rotation boxplot with Min-Inliers and Min-Parallax filters to {rotation_file}")
