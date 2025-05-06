import pycolmap
from collections import defaultdict
from pathlib import Path

def generate_covisibility_txt(sparse_path, output_path="camerasModel.txt"):
    recon = pycolmap.Reconstruction(sparse_path)
    image_id_to_name = {img.image_id: img.name for img in recon.images.values()}
    covisibility = defaultdict(set)

    # Build covisibility graph from shared 3D point tracks
    for point in recon.points3D.values():
        image_ids = [obs.image_id for obs in point.track.elements]
        for i in range(len(image_ids)):
            for j in range(i + 1, len(image_ids)):
                id1, id2 = image_ids[i], image_ids[j]
                covisibility[id1].add(id2)
                covisibility[id2].add(id1)

    # Write to file
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        for img_id, neighbors in sorted(covisibility.items()):
            img_name = image_id_to_name[img_id]
            neighbor_names = [image_id_to_name[n_id] for n_id in sorted(neighbors)]
            line = " ".join([img_name] + neighbor_names)
            f.write(line + "\n")
    
    print(f"Written covisibility file with {len(covisibility)} entries to {output_path}")


def main():

    sparse_path = "./data/graham-hall/sparse"

    generate_covisibility_txt(sparse_path)
    return


if __name__ == "__main__":
    main()
