from pathlib import Path
from colmap_opencv_helpers import compute_submap_diameter
import csv

seq = "seq_001"
# Make a list to get all the submaps in the sequence
submaps = [f for f in Path(f'data/{seq}/sparse').iterdir() if f.is_dir()]

# Open a CSV file to write the output
with open(f'data/{seq}/submap_diameters.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Submap', 'Diameter'])

    for submap in submaps:
        sparse_model_dir = submap
        d = compute_submap_diameter(sparse_model_dir)
        writer.writerow([submap.name, f"{d:.4f}"])
