import pycolmap
from matching import get_matcher
from specular_mask import *
from process_image_pairs import *
from pathlib import Path
from colmap_opencv_helpers import *
from config_error_measurement import *
from my_logging import setup_logging
import numpy as np


logger = setup_logging(DEBUG)

# Define the paths
sparse_model_dir = Path(f'data/{seq}/sparse/{submap}')
root_image_dir = Path(f'data/{seq}/img_train')

# Load the COLMAP reconstruction
reconstruction = pycolmap.Reconstruction(sparse_model_dir)

#Prints peeks of the reconstruction
# Print some basic information to verify the reconstruction
print(f"Number of images: {len(reconstruction.images)}")
print(f"Number of 3D points: {len(reconstruction.points3D)}")

# Print the first few images
for image_id, image in list(reconstruction.images.items())[:5]:
    print(f"Image ID: {image_id}, Camera ID: {image.camera_id}, Name: {image.name}")

# Print the first few cameras
for camera_id, camera in list(reconstruction.cameras.items())[:5]:
    print(f"Camera ID: {camera_id}, Model: {camera.model}, Params: {camera.params}")


image0 = reconstruction.find_image_with_name(image0_name)
image1 = reconstruction.find_image_with_name(image1_name)

absolute_pose0 = image0.cam_from_world
absolute_pose1 = image1.cam_from_world

R01_colmap, t01_colmap = compute_relative_pose(absolute_pose0, absolute_pose1)

# Print COLMAP relative pose
print("COLMAP Relative Rotation:\n", R01_colmap)
print("COLMAP Relative Translation:\n", t01_colmap)

### MATCHER ####
matcher = get_matcher(model_name, device=device)
img0_path = root_image_dir / image0_name
img1_path = root_image_dir / image1_name
output_dir = Path(f'output/error_measurement')
output_dir.mkdir(parents=True, exist_ok=True)

result = process_image_pairs(img0_path, img1_path, output_dir, model_name, matcher, logger = logger, resize = resize, masking = masking, plot_kpts=plot_kpts)

if len(result['matched_kpts0']) < 5:
    print("Not enough matches found")
    exit()

mkpts0 = result['matched_kpts0']
mkpts1 = result['matched_kpts1']

# Get camera intrinsics from pycolmap to estimate the relative pose with pycolmap
# Retrieve cameras from COLMAP
camera0 = reconstruction.camera(image0.camera_id)
camera1 = reconstruction.camera(image1.camera_id)

# Convert mkpts to COLMAP coordinates
corrected_mkpts0 = adapt_mkpts_to_colmap(mkpts0)
corrected_mkpts1 = adapt_mkpts_to_colmap(mkpts1)

# Compute E matrix
result_colmap = pycolmap.estimate_essential_matrix(corrected_mkpts0, corrected_mkpts1, camera0, camera1)

R01_est = result_colmap['cam2_from_cam1'].rotation.matrix()
t01_est = result_colmap['cam2_from_cam1'].translation

# rotation error
rot_err = rotation_error_deg(R01_colmap, R01_est)
# translation error
trans_err = translation_error_m(t01_colmap, t01_est)

# In a multi-pair scenario, store in lists
rot_errs = [rot_err]
trans_errs = [trans_err]

# 3) mAA
thresholds_r = np.linspace(1, 10, 10)
thresholds_t = np.geomspace(0.2, 5, 10)
my_mAA = compute_mAA(rot_errs, trans_errs, thresholds_r, thresholds_t)

print(f"Rotation error (deg): {rot_err:.3f}")
print(f"Translation error (colmap scale): {trans_err:.3f}")
print(f"mAA: {my_mAA:.3f}")