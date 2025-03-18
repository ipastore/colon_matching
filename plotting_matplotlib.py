import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# ---------------------------------------------------------------------
# Create output directory for plots
# ---------------------------------------------------------------------
output_dir = Path('./output/plot_images')
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1) Cargar el JSON
# ---------------------------------------------------------------------
with open('./output/error_measurement/seq_001/superpoint-lg/20250318_201146/seq_full_report_20250318_202931_complete.json', 'r') as f:
    data = json.load(f)

# Extraer submaps
submaps = data.get("submaps", {})
submap_ids = list(submaps.keys())  

# ---------------------------------------------------------------------
# 2) Graficar histogramas de rotación y traslación POR CADA submap
# ---------------------------------------------------------------------
for submap_id in submap_ids:
    submap_data = submaps[submap_id]
    pairs = submap_data.get("pairs", [])

    # Extraemos errores de rotación y traslación
    rot_errors = [p["rot_error_deg"] for p in pairs]
    trans_errors = [p["trans_error"] for p in pairs]

    # Rotation histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rot_errors, bins=20)
    plt.title(f'Rotation Histogram - Submap {submap_id}')
    plt.xlabel('Rotation error (degrees)')
    plt.ylabel('Frequency')
    # Save the plot instead of showing it
    plt.savefig(output_dir / f'submap_{submap_id}_rotation_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    # Translation histogram
    plt.figure(figsize=(10, 6))
    plt.hist(trans_errors, bins=20)
    plt.title(f'Translation Histogram - Submap {submap_id}')
    plt.xlabel('Translation error (relative units)')
    plt.ylabel('Frequency')
    # Save the plot instead of showing it
    plt.savefig(output_dir / f'submap_{submap_id}_translation_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory


# ---------------------------------------------------------------------
# 3) Boxplot comparando TODOS los submaps
#    (muestra la distribución de los errores de rot y trans en cada submap)
# ---------------------------------------------------------------------
all_rot_errors = []
all_trans_errors = []
labels = []

for submap_id in submap_ids:
    pairs = submaps[submap_id].get("pairs", [])
    rot_errors = [p["rot_error_deg"] for p in pairs]
    trans_errors = [p["trans_error"] for p in pairs]

    # Guardamos para boxplot
    all_rot_errors.append(rot_errors)
    all_trans_errors.append(trans_errors)
    labels.append(submap_id)

# Boxplot of rotation errors
plt.figure(figsize=(12, 8))
plt.boxplot(all_rot_errors, labels=labels)
plt.title('Distribution of Rotation Errors by Submap')
plt.xlabel('Submap')
plt.ylabel('Rotation Error (degrees)')
# Save the plot instead of showing it
plt.savefig(output_dir / 'all_submaps_rotation_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

# Boxplot of translation errors
plt.figure(figsize=(12, 8))
plt.boxplot(all_trans_errors, labels=labels)
plt.title('Distribution of Translation Errors by Submap')
plt.xlabel('Submap')
plt.ylabel('Translation Error')
# Save the plot instead of showing it
plt.savefig(output_dir / 'all_submaps_translation_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory


# ---------------------------------------------------------------------
# 4) Bar plot con alguna métrica agregada por submap
#    Por ejemplo: submap_rmse_rotation_deg, submap_AA, etc.
# ---------------------------------------------------------------------
rmse_vals = []
AA_vals = []
for submap_id in submap_ids:
    submap_data = submaps[submap_id]
    # Asegurarnos que estas llaves existan
    rmse = submap_data.get("submap_rmse_rotation_deg", 0.0)
    aa   = submap_data.get("submap_AA", 0.0)
    rmse_vals.append(rmse)
    AA_vals.append(aa)

# Bar plot of RMSE rotation
plt.figure(figsize=(12, 8))
plt.bar(labels, rmse_vals)
plt.title('Rotation RMSE by Submap')
plt.xlabel('Submap')
plt.ylabel('Rotation RMSE (degrees)')
# Save the plot instead of showing it
plt.savefig(output_dir / 'all_submaps_rmse_rotation_barplot.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

# Bar plot of Angular Accuracy (AA)
plt.figure(figsize=(12, 8))
plt.bar(labels, AA_vals)
plt.title('Angular Accuracy (AA) by Submap')
plt.xlabel('Submap')
plt.ylabel('AA')
# Save the plot instead of showing it
plt.savefig(output_dir / 'all_submaps_AA_barplot.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

print(f"All plots saved to {output_dir}")