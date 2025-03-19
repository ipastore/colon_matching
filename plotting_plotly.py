import json
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np

# En tu caso, ajusta la ruta según corresponda
timestamp_name = '20250319_192554'
output_dir = Path(f'./output/plot_images/plotly/{timestamp_name}')
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1) Carga del JSON
# ------------------------------------------------------------------
json_path = './output/error_measurement/seq_001/superpoint-lg/20250319_192554/sequence_seq_001_20250319_222139.json'
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
        # En tu JSON original usabas 'mkpts0'/'mkpts1' en lugar de 'mkpts0'/'mkpts1'
        # Ajusta aquí si tu JSON no tiene esas llaves exactas.
        mkpts0 = pair.get("mkpts0", None)
        mkpts1 = pair.get("mkpts1", None)

        row = {
            'submap_id': submap_id,
            'image0': pair["image0"],
            'image1': pair["image1"],
            'rot_error_deg': pair["rot_error_deg"],
            'trans_error': pair["trans_error"],
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
            'Total_img': total_img,
            'Nimg_percentage': nimg_percentage
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
    y="rot_error_deg",
    points="all",  # "all" para mostrar todos los puntos
    hover_data=["image0","image1","mkpts0","mkpts1","Total_img","Nimg_percentage"]
)

# Si tienes rotaciones de hasta 180°, podrías forzar el rango:
# fig.update_yaxes(range=[0, 180])  # Descomenta si quieres limitar a 180° máximo

fig.update_layout(
    title="Distribución de Error de Rotación por Submap (Interactivo)",
    yaxis_title="Rot. Error (deg)"
)

rotation_file = output_dir / "rotacion_error_boxplot.html"
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
    hover_data=["image0","image1","mkpts0","mkpts1","Total_img","Nimg_percentage"]
)

# Si quieres ampliar el rango (p.ej. hasta 5), haz:
# fig.update_yaxes(range=[0, 5])

fig.update_layout(
    title="Distribución de Error de Translación por Submap (Interactivo)",
    yaxis_title="Trans. Error"
)

translation_file = output_dir / "traslacion_error_boxplot.html"
fig.write_html(translation_file)
print(f"Saved translation boxplot to {translation_file}")

# ------------------------------------------------------------------
# 5) Histograma de rotaciones
#    Ajustar nbins y rango
# ------------------------------------------------------------------
fig = px.histogram(
    df,
    x="rot_error_deg",
    color="submap_id",
    marginal="box",
    hover_data=["image0","image1","mkpts0","mkpts1","Total_img","Nimg_percentage"],
    nbins=1800,         # Aumentar bins
    # range_x=[0, 180]  # Si quieres ver hasta 180°
)
fig.update_layout(
    title="Histograma de Errores de Rotación por Submap",
    xaxis_title="Rot. Error (deg)"
)

rotation_hist_file = output_dir / "rotacion_error_histogram.html"
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
    hover_data=["image0","image1","mkpts0","mkpts1","Total_img","Nimg_percentage"],
    nbins=1800,         # Ajusta según el rango y lo que desees
    # range_x=[0, 5]    # Si quieres llegar hasta 5
)
fig.update_layout(
    title="Histograma de Errores de Translación por Submap",
    xaxis_title="Trans. Error"
)

translation_hist_file = output_dir / "traslacion_error_histogram.html"
fig.write_html(translation_hist_file)
print(f"Saved translation histogram to {translation_hist_file}")
