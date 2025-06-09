import os
import shutil
from pathlib import Path

def copy_submap_images(
    seq_name: str,
    image_train_root: Path,
    output_root: Path,
    image_list_suffix: str = ".txt",
):
    """
    For each submap text file in 'data/{seq_name}/cluster_list/',
    read the text file that lists the images (e.g. "0.txt", "1.txt", etc.),
    then copy them from 'image_train_root' into a corresponding folder under
    'output_root/submap_images/'.

    Args:
        seq_name (str): Name of the sequence folder (e.g. "seq_001")
        image_train_root (Path): The directory that has all your original images.
        output_root (Path): The directory where you'd like to create sub-folders
                            for each submap's images.
        image_list_suffix (str): Suffix for the text file that lists the images per submap.
                                 Default is ".txt"
    """
    # Construct cluster list directory path
    cluster_list_dir = Path(f"data/{seq_name}/cluster_list")
    
    # Ensure the cluster list directory exists
    if not cluster_list_dir.exists():
        raise FileNotFoundError(f"Cluster list directory not found: {cluster_list_dir}")
    
    # Create the output submap_images directory
    submap_images_dir = output_root / "submap_images"
    submap_images_dir.mkdir(parents=True, exist_ok=True)

    # Process all text files in the cluster list directory
    for cluster_file in sorted(cluster_list_dir.glob(f"*{image_list_suffix}")):
        submap_name = cluster_file.stem  # e.g. "0", "1", ...
        
        # Read the image names from the cluster file
        with open(cluster_file, "r") as f:
            image_names = [line.strip() for line in f if line.strip()]

        # Create an output folder for this submap
        submap_output_dir = submap_images_dir / submap_name
        submap_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy each image from image_train_root to submap_output_dir
        copied_count = 0
        for img_name in image_names:
            src = image_train_root / img_name
            dst = submap_output_dir / img_name
            if not src.exists():
                print(f"Warning: image {img_name} not found in {image_train_root}")
                continue
            shutil.copy2(src, dst)
            copied_count += 1

        print(f"Submap '{submap_name}': Copied {copied_count} out of {len(image_names)} images.")


if __name__ == "__main__":
    # Example usage:
    # Suppose your folder structure is like:
    # data/
    #   ├── seq_001/
    #   │    ├── cluster_list/
    #   │    │    ├── 0.txt   (list of images for this submap)
    #   │    │    ├── 1.txt
    #   │    │    └── ...
    #   ├── image_train/
    #   │    ├── img1.jpg
    #   │    ├── img2.jpg
    #   │    └── ...
    #   └── ...
    #
    # And you want to create "output_root/submap_images/" with all the images.
    
    seq_name = "seq_002"                      # Sequence name
    image_train_dir = Path("data/seq_002/img_train")  # Where all your original images are
    output_dir = Path(f"data/{seq_name}")              # Where to create submap_images directory
    
    copy_submap_images(
        seq_name=seq_name,
        image_train_root=image_train_dir,
        output_root=output_dir,
        image_list_suffix=".txt",  # Adjust if your submap image-list file has a different suffix
    )
