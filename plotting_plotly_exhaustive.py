import json
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np

# En tu caso, ajusta la ruta según corresponda
timestamp_name = '250407_165322_submap_9_exhaustive'
output_dir = Path(f'./output/plot_images/plotly/{timestamp_name}')
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1) Carga del JSON
# ------------------------------------------------------------------
json_path = './output/error_measurement/seq_001/superpoint-lg/20250407_165322_submap_9_exhaustive/sequence_seq_001_20250407_222934.json'
with open(json_path, 'r') as f:
    data = json.load(f)

submaps_dict = data.get("submaps", {})

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
            'trans_error': pair["trans_error"],
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
    hover_data=["image0", "image1", "mkpts","inliers", "Total_img", "Nimg_percentage", "parallax", "covis_score", "trans_error"],
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
    y="trans_error",
    points="all",
    hover_data=["image0", "image1", "mkpts", "inliers", "Total_img", "Nimg_percentage", "parallax", "covis_score", "rot_error"],
)
# Si quieres ampliar el rango (p.ej. hasta 5), haz:
# fig.update_yaxes(range=[0, 5])

fig.update_layout(
    title="Distribution of error of Translation (degrees between translation vectors) by Submap",
    yaxis_title="Trans. Error (deg)"
)

translation_file = output_dir / "trans_error_boxplot.html"
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
    hover_data=["image0", "image1", "mkpts","inliers","Total_img", "Nimg_percentage", "parallax", "covis_score", "trans_error"],
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
    x="trans_error",
    color="submap_id",
    marginal="box",
    hover_data=["image0", "image1", "mkpts", "inliers","Total_img", "Nimg_percentage", "parallax", "covis_score", "rot_error"],
    nbins=1800,         # Ajusta según el rango y lo que desees
    # range_x=[0, 5]    # Si quieres llegar hasta 5
)
fig.update_layout(
    title="Distribution of Translation Error (degrees between translation vectors) by Submap",
    xaxis_title="Trans. Error"
)

translation_hist_file = output_dir / "trans_error_histogram.html"
fig.write_html(translation_hist_file)
print(f"Saved translation histogram to {translation_hist_file}")

# ------------------------------------------------------------------
# 7) Correlation plot: Parallax vs. Translation Error
# ------------------------------------------------------------------
fig = px.scatter(
    df,
    x="parallax",
    y="trans_error",
    color="submap_id",
    hover_data=["image0", "image1", "mkpts","inliers", "Total_img", "Nimg_percentage", "covis_score", "rot_error"]
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
    hover_data=["image0", "image1", "mkpts","inliers", "Total_img", "Nimg_percentage", "covis_score", "trans_error"]
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
