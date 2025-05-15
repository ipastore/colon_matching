import json
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np

# En tu caso, ajusta la ruta según corresponda
timestamp_name = '20250514_082802'
output_dir = Path(f'./output/plot_images/plotly/{timestamp_name}')
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1) Carga del JSON
# ------------------------------------------------------------------
submaps_dir = Path('./output/error_measurement/seq_001/superpoint-lg/20250514_082802/submaps')

# Find all submap JSON files
submap_files = list(submaps_dir.glob(f"submap_*.json"))

# Make submaps_dict from submap_files (JSON)
submaps_dict = {}
for submap_file in submap_files:
    with open(submap_file, 'r') as f:
        submap_data = json.load(f)
    
        submap_name = submap_data.get("submap")
        submap_content = submap_data.get("data", {})

        # Add to submaps_dict
        submaps_dict[submap_name] = submap_content

# ------------------------------------------------------------------
# 2) Construimos un DataFrame con TODA la info necesaria en cada fila
# ------------------------------------------------------------------
rows = []
for submap_id, submap_data in submaps_dict.items():
    pairs = submap_data.get("pairs", [])
    
    # Podemos recuperar info global de submap
    total_img = submap_data.get("Total_img", None)
    nimg_percentage = submap_data.get("Nimg_percentage", None)
    
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
            "covis_score": pair["covis_score"]
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
    hover_data=["image0", "image1", "mkpts","inliers", "Total_img", "Nimg_percentage", "parallax", "covis_score", "trans_error_deg", "trans_error_rel"],
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
    hover_data=["image0", "image1", "mkpts", "inliers", "Total_img", "Nimg_percentage", "parallax", "covis_score", "rot_error", "trans_error_rel", "t_colmap_norm"],
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
    hover_data=["image0", "image1", "mkpts","inliers","Total_img", "Nimg_percentage", "parallax", "covis_score", "trans_error_deg", "trans_error_rel", "t_colmap_norm"],
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
    hover_data=["image0", "image1", "mkpts", "inliers","Total_img", "Nimg_percentage", "parallax", "covis_score", "rot_error", "trans_error_rel", "t_colmap_norm"],
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
    hover_data=["image0", "image1", "mkpts","inliers", "Total_img", "Nimg_percentage", "covis_score", "rot_error", "trans_error_rel", "t_colmap_norm"]
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
    hover_data=["image0", "image1", "mkpts","inliers", "Total_img", "Nimg_percentage", "covis_score", "trans_error_deg", "trans_error_rel", "t_colmap_norm"]
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

# # ------------------------------------------------------------------
# # 9) Boxplot of relative translation error
# # ------------------------------------------------------------------
# fig = px.box(
#     df,
#     x="submap_id",
#     y="trans_error_rel",
#     points="all",
#     hover_data=["image0", "image1", "mkpts", "inliers", "Total_img", "Nimg_percentage", "parallax", "covis_score", "rot_error", "trans_error_deg", "t_colmap_norm"],
# )

# fig.update_layout(
#     title="Distribution of Relative Translation Error by Submap",
#     yaxis_title="Relative Translation Error"
# )

# trans_rel_file = output_dir / "trans_error_rel_boxplot.html"
# fig.write_html(trans_rel_file)
# print(f"Saved relative translation error boxplot to {trans_rel_file}")

# # ------------------------------------------------------------------
# # 10) Histogram of relative translation error
# # ------------------------------------------------------------------
# fig = px.histogram(
#     df,
#     x="trans_error_rel",
#     color="submap_id",
#     marginal="box",
#     hover_data=["image0", "image1", "mkpts", "inliers", "Total_img", "Nimg_percentage", "parallax", "covis_score", "rot_error", "trans_error_deg", "t_colmap_norm"],
#     nbins=1800,
# )

# fig.update_layout(
#     title="Distribution of Relative Translation Error by Submap",
#     xaxis_title="Relative Translation Error"
# )

# trans_rel_hist_file = output_dir / "trans_error_rel_histogram.html"
# fig.write_html(trans_rel_hist_file)
# print(f"Saved relative translation error histogram to {trans_rel_hist_file}")

# # ------------------------------------------------------------------
# # 11) Correlation plot: Parallax vs. Relative Translation Error
# # ------------------------------------------------------------------
# fig = px.scatter(
#     df,
#     x="parallax",
#     y="trans_error_rel",
#     color="submap_id",
#     hover_data=["image0", "image1", "mkpts", "inliers", "Total_img", "Nimg_percentage", "covis_score", "rot_error", "trans_error_deg", "t_colmap_norm"]
# )

# fig.update_layout(
#     title="Correlation between Parallax and Relative Translation Error",
#     xaxis_title="Parallax",
#     yaxis_title="Relative Translation Error"
# )

# # Add ticks at intervals of 1 instead of 5
# fig.update_xaxes(dtick=1)  # This sets tick intervals to 1

# parallax_trans_rel_file = output_dir / "parallax_trans_rel_correlation.html"
# fig.write_html(parallax_trans_rel_file)
# print(f"Saved parallax-relative translation correlation plot to {parallax_trans_rel_file}")

# ------------------------------------------------------------------
# 12) Correlation plot: Inliers vs. Translation Error (degrees)
# ------------------------------------------------------------------
fig = px.scatter(
    df,
    x="inliers",
    y="trans_error_deg",
    color="submap_id",
    hover_data=["image0", "image1", "mkpts", "Total_img", "Nimg_percentage", "parallax", "covis_score", "rot_error", "trans_error_rel", "t_colmap_norm"]
)

fig.update_layout(
    title="Correlation between Number of Inliers and Translation Error",
    xaxis_title="Number of Inliers",
    yaxis_title="Translation Error (deg)"
)

inliers_trans_file = output_dir / "inliers_trans_deg_correlation.html"
fig.write_html(inliers_trans_file)
print(f"Saved inliers-translation error correlation plot to {inliers_trans_file}")

# # ------------------------------------------------------------------
# # 13) Correlation plot: Inliers vs. Relative Translation Error
# # ------------------------------------------------------------------
# fig = px.scatter(
#     df,
#     x="inliers",
#     y="trans_error_rel",
#     color="submap_id",
#     hover_data=["image0", "image1", "mkpts", "Total_img", "Nimg_percentage", "parallax", "covis_score", "rot_error", "trans_error_deg", "t_colmap_norm"]
# )

# fig.update_layout(
#     title="Correlation between Number of Inliers and Relative Translation Error",
#     xaxis_title="Number of Inliers",
#     yaxis_title="Relative Translation Error"
# )

# inliers_trans_rel_file = output_dir / "inliers_trans_rel_correlation.html"
# fig.write_html(inliers_trans_rel_file)
# print(f"Saved inliers-relative translation error correlation plot to {inliers_trans_rel_file}")

# ------------------------------------------------------------------
# 14) Correlation plot: Inliers vs. Rotation Error
# ------------------------------------------------------------------
fig = px.scatter(
    df,
    x="inliers",
    y="rot_error",
    color="submap_id",
    hover_data=["image0", "image1", "mkpts", "Total_img", "Nimg_percentage", "parallax", "covis_score", "trans_error_deg", "trans_error_rel", "t_colmap_norm"]
)

fig.update_layout(
    title="Correlation between Number of Inliers and Rotation Error",
    xaxis_title="Number of Inliers",
    yaxis_title="Rotation Error (deg)"
)

inliers_rot_file = output_dir / "inliers_rot_error_correlation.html"
fig.write_html(inliers_rot_file)
print(f"Saved inliers-rotation error correlation plot to {inliers_rot_file}")

# # ------------------------------------------------------------------
# # 15) Correlation plot: Relative Translation Error vs. Translation Error (degrees)
# # ------------------------------------------------------------------
# fig = px.scatter(
#     df,
#     x="trans_error_rel",
#     y="trans_error_deg",
#     color="submap_id",
#     hover_data=["image0", "image1", "mkpts", "inliers", "Total_img", "Nimg_percentage", "parallax", "covis_score", "rot_error", "t_colmap_norm"]
# )

# fig.update_layout(
#     title="Correlation between Relative Translation Error and Translation Error (degrees)",
#     xaxis_title="Relative Translation Error",
#     yaxis_title="Translation Error (deg)"
# )

# trans_rel_deg_file = output_dir / "trans_error_rel_deg_correlation.html"
# fig.write_html(trans_rel_deg_file)
# print(f"Saved relative vs. angular translation error correlation plot to {trans_rel_deg_file}")
