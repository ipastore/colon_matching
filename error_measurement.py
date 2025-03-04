import pycolmap
import numpy as np
from process_image_pairs import process_image_pairs
from matching.utils import get_default_device
from matching import get_matcher
from pathlib import Path
import cv2



def get_relative_pose(R0, t0, R1, t1):
    """ Compute the relative pose (R, t) from COLMAP. New reference frame is image1 """
    R_rel = R0.T @ R1  
    t_rel = R0.T @ (t1 - t0)  
    return R_rel, t_rel

def load_R_t(image_id, reconstruction):
    """ Load rotation matrix and translation vector for a given image from COLMAP reconstruction """
    image = reconstruction.images[image_id]

    # Convert COLMAP's quaternion + translation to a rotation matrix
    R = image.cam_from_world.rotation.matrix()
    t = image.cam_from_world.translation

    return R, t

def estimate_relative_pose(mkpts0, mkpts1, K):
    """
    Estimate relative pose (R, t) from matched keypoints using the essential matrix.
    :param mkpts0: Matched keypoints in image 1 (Nx2 array)
    :param mkpts1: Matched keypoints in image 2 (Nx2 array)
    :param K: Camera intrinsic matrix
    :return: Estimated rotation (R) and translation (t)
    """
    E, _ = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, mkpts0, mkpts1, K)
    return R, t

def load_camera_intrinsics(image_id, reconstruction):
    """ Load camera intrinsics from COLMAP reconstruction """
    image = reconstruction.images[image_id]
    camera = reconstruction.cameras[image.camera_id]
    K = np.array([[camera.params[0], 0, camera.params[2]],
                  [0, camera.params[1], camera.params[3]],
                  [0, 0, 1]])
    return K

def load_image_name_image_id_dict(reconstruction):
    """ Get image ID from image name """
    image_ids = {image.name: image_id for image_id, image in reconstruction.images.items()}
    return image_ids


# Define the paths
sparse_model_dir = Path(f'data/seq_001/33/sparse/0')
root_image_dir = Path(f'data/seq_001/33/img_train')

# Load the COLMAP reconstruction
reconstruction = pycolmap.Reconstruction(sparse_model_dir)

# Load dict
image_name_id = load_image_name_image_id_dict(reconstruction)

#Load image_ids
image0_name = 'image0.png'
image1_name = 'image1.png'
image0_id = image_name_id[image0_name]
image1_id = image_name_id[image1_name]

# Get COLMAP's relative pose
R0, t0 = load_R_t(image0_id, reconstruction)
R1, t1 = load_R_t(image1_id, reconstruction)
R_colmap, t_colmap = get_relative_pose(R0, t0, R1, t1)

# Print COLMAP relative pose
print("COLMAP Relative Rotation:\n", R_colmap)
print("COLMAP Relative Translation:\n", t_colmap)

### MATCHER ####
logger = None
resize = None
masking = True
model_name = 'superpoint-lg'
device = get_default_device()
matcher = get_matcher(model_name, device=device)
img0_path = root_image_dir + image0_name
img1_path = root_image_dir + image1_name
output_dir = Path(f'output/error_measurement')
output_dir.mkdir(parents=True, exist_ok=True)

result = process_image_pairs(img0_path, img1_path, output_dir, model_name, matcher, logger = logger, resize = resize, masking = masking)

mkpts0 = result['matched_kpts0']
mkpts1 = result['matched_kpts1']
K = load_camera_intrinsics(image0_id, reconstruction)
R_est, t_est = estimate_relative_pose(mkpts0, mkpts1, K)