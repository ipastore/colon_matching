import pycolmap
import numpy as np

# Function to extract relative transformation between two images
def get_relative_pose(image1_id, image2_id, reconstruction):
    """ Compute the relative pose (R, t) from COLMAP images. """
    image1 = reconstruction.images[image1_id]
    image2 = reconstruction.images[image2_id]

    # Convert COLMAP's quaternion + translation to a rotation matrix
    R1 = image1.cam_from_world.rotation.matrix()
    t1 = image1.cam_from_world.translation

    R2 = image2.cam_from_world.rotation.matrix()
    t2 = image2.cam_from_world.translation

    # Compute relative rotation and translation
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    return R_rel, t_rel

# Define the path to your COLMAP sparse model
sparse_model_path = "../data/seq_001/33/sparse/0"  # Adjust the path accordingly

# Load the COLMAP reconstruction
reconstruction = pycolmap.Reconstruction(sparse_model_path)

# Print some basic information to verify the reconstruction
print(f"Number of images: {len(reconstruction.images)}")
print(f"Number of 3D points: {len(reconstruction.points3D)}")

# Print the first few images
for image_id, image in list(reconstruction.images.items())[:5]:
    print(f"Image ID: {image_id}, Camera ID: {image.camera_id}, Name: {image.name}")

# Print the first few 3D points
for point_id, point3D in list(reconstruction.points3D.items())[:5]:
    print(f"Point ID: {point_id}, XYZ: {point3D.xyz}, RGB: {point3D.color}")


# Select a pair of images (update these IDs with actual ones from your dataset)
image1_id = 12314
image2_id = 12313

# Get COLMAP's relative pose
R_colmap, t_colmap = get_relative_pose(image1_id, image2_id, reconstruction)

# Print COLMAP relative pose
print("COLMAP Relative Rotation:\n", R_colmap)
print("COLMAP Relative Translation:\n", t_colmap)
