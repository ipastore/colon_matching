import json
import pandas as pd
import plotly.express as px
from pathlib import Path

output_dir = Path('./output/plot_images/plotly')
output_dir.mkdir(parents=True, exist_ok=True)

with open('./output/error_measurement/seq_001/superpoint-lg/20250318_201146/seq_full_report_20250318_202931_complete.json', 'r') as f:
    data = json.load(f)

submaps_dict = data.get("submaps", {})

# Construimos una lista con un diccionario por cada par de imágenes
rows = []
for submap_id, submap_data in submaps_dict.items():
    pairs = submap_data.get("pairs", [])
    for pair in pairs:
        row = {
            'submap_id': submap_id,
            'image0': pair["image0"],
            'image1': pair["image1"],
            'rot_error_deg': pair["rot_error_deg"],
            'trans_error': pair["trans_error"]
            # Puedes añadir mkpts0, mkpts1, etc.
        }
        rows.append(row)

df = pd.DataFrame(rows)
print(df.head())  # Para ver las primeras filas

# Boxplot de rotación (con puntos individuales a la vista)
fig = px.box(
    df, 
    x="submap_id", 
    y="rot_error_deg", 
    points="all",                 # "all" para ver todos los puntos, o "outliers" solo para atípicos
    hover_data=["image0","image1"]  # Esto muestra los nombres de las imágenes en el hover
)
fig.update_layout(title="Distribución de Error de Rotación por Submap (Interactivo)")

# Save the rotation boxplot as an HTML file
rotation_file = output_dir / "rotacion_error_boxplot.html"
fig.write_html(rotation_file)
print(f"Saved rotation boxplot to {rotation_file}")

# Boxplot de traslación
fig = px.box(
    df, 
    x="submap_id", 
    y="trans_error", 
    points="all",
    hover_data=["image0","image1"]
)
fig.update_layout(title="Distribución de Error de Translación por Submap (Interactivo)")

# Save the translation boxplot as an HTML file
translation_file = output_dir / "traslacion_error_boxplot.html"
fig.write_html(translation_file)
print(f"Saved translation boxplot to {translation_file}")

# Additionally, create a histogram for rotation errors by submap
fig = px.histogram(
    df,
    x="rot_error_deg",
    color="submap_id",
    marginal="box",  # adds a boxplot on the top
    hover_data=["image0", "image1"]
)
fig.update_layout(title="Histograma de Errores de Rotación por Submap")

# Save the rotation histogram
rotation_hist_file = output_dir / "rotacion_error_histogram.html"
fig.write_html(rotation_hist_file)
print(f"Saved rotation histogram to {rotation_hist_file}")

# Create a histogram for translation errors by submap
fig = px.histogram(
    df,
    x="trans_error",
    color="submap_id",
    marginal="box",
    hover_data=["image0", "image1"]
)
fig.update_layout(title="Histograma de Errores de Translación por Submap")

# Save the translation histogram
translation_hist_file = output_dir / "traslacion_error_histogram.html"
fig.write_html(translation_hist_file)
print(f"Saved translation histogram to {translation_hist_file}")