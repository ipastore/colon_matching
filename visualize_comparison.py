import argparse
import numpy as np
import open3d
import sys
from pathlib import Path
import copy
import pycolmap


# Add COLMAP scripts directory to path to import read_write_model
colmap_scripts_path = "/Users/ignaciopastorebenaim/Documents/MGRCV/COLON/colmap/scripts/python"
sys.path.append(colmap_scripts_path)
from read_write_model import qvec2rotmat, rotmat2qvec, read_model, Image

# Import your existing functions
from visualize_3D import load_result_matcher_npz, rigid3d_to_matrix4x4, pose_to_transformation_matrix
from colmap_opencv_helpers import rotation_error_deg, translation_error_direction_deg, translation_error_relative_to_colmap, compute_relative_pose


class Model:
    def __init__(self):
        self.cameras = []
        self.images = []
        self.points3D = []
        self.__vis = None

    def read_model(self, path, ext=""):
        self.cameras, self.images, self.points3D = read_model(path, ext)

    def add_points(self, min_track_len=3, remove_statistical_outlier=True):
        pcd = open3d.geometry.PointCloud()

        xyz = []
        rgb = []
        for point3D in self.points3D.values():
            track_len = len(point3D.point2D_idxs)
            if track_len < min_track_len:
                continue
            xyz.append(point3D.xyz)
            rgb.append(point3D.rgb / 255)

        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)

        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )

        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()

    def add_cameras(self, scale=1, color=[0.8, 0.2, 0.8]):
        frames = []
        for img in self.images.values():
            # rotation
            R = qvec2rotmat(img.qvec)

            # translation
            t = img.tvec

            # invert
            t = -R.T @ t
            R = R.T

            # intrinsics
            cam = self.cameras[img.camera_id]

            if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = cam.params[0]
                cx = cam.params[1]
                cy = cam.params[2]
            elif cam.model in (
                "PINHOLE",
                "OPENCV",
                "OPENCV_FISHEYE",
                "FULL_OPENCV",
            ):
                fx = cam.params[0]
                fy = cam.params[1]
                cx = cam.params[2]
                cy = cam.params[3]
            else:
                raise Exception("Camera model not supported")

            # intrinsics
            K = np.identity(3)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy

            # create axis, plane and pyramid geometries that will be drawn
            cam_model = draw_camera(K, R, t, cam.width, cam.height, scale, color)
            frames.extend(cam_model)

        # add geometries to visualizer
        for i in frames:
            self.__vis.add_geometry(i)

    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()


def draw_camera(K, R, t, w, h, scale=1, color=[0.8, 0.2, 0.8]):
    """Create axis, plane and pyramid geometries in Open3D format."""
    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5 * scale
    )
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]

#TODO colon: instead of loading the npz, just load the the whole submap and the the pair_data within the dict
def visualize_camera_pair(npz_path, colmap_model_path, format=".txt", camera_scale=0.25):
    """
    Visualize a single camera pair using Open3D, showing:
    1. All 3D points from the COLMAP model
    2. C0 ground truth camera (blue)
    3. C1 ground truth camera (blue)
    4. C1 estimated camera (red)
    
    Args:
        npz_path: Path to the .npz file with matching results
        colmap_model_path: Path to COLMAP model directory
        format: Format of COLMAP model files (.bin or .txt)
    """
    # Load matching results from .npz
    match_data = load_result_matcher_npz(Path(npz_path))
    
    # Extract data from match_data
    matched_kpts0 = match_data["matched_kpts0"]
    matched_kpts1 = match_data["matched_kpts1"]
    inlier_kpts0 = match_data["inlier_kpts0"]
    inlier_kpts1 = match_data["inlier_kpts1"]
    R01_est = match_data["R_est"]
    t01_est = match_data["t_est"]
    image0_name = match_data["image0"]
    image1_name = match_data["image1"]
    parallax = match_data["parallax"]
    
    # Check if we have the estimation data
    if R01_est is None or t01_est is None:
        print(f"Error: No estimated pose found in {npz_path}")
        return
    
    # Load COLMAP model
    gt_model = Model()
    gt_model.read_model(colmap_model_path, ext=format)
    
    print("COLMAP model statistics:")
    print("  num_cameras:", len(gt_model.cameras))
    print("  num_images:", len(gt_model.images))
    print("  num_points3D:", len(gt_model.points3D))
    print("  num_matches:", len(matched_kpts0))
    print("  num_inliers:", len(inlier_kpts0))
    

    # Load images from the COLMAP model
    image0 = None
    image1 = None
    for img_id, img in gt_model.images.items():
        if img.name == image0_name or Path(img.name).stem == Path(image0_name).stem:
            image0 = img
        if img.name == image1_name or Path(img.name).stem == Path(image1_name).stem:
            image1 = img
        
        if image0 is not None and image1 is not None:
            break
    
    if image0 is None or image1 is None:
        print(f"Error: Could not find images in the COLMAP model: {image0_name}, {image1_name}")
        return
    
    # Load images from the reconstruction to recalculate the errors and extract the ground truth
    # Take ground truth camera 0 and camera 1 from the COLMAP reconstruction
    reconstruction = pycolmap.Reconstruction(colmap_model_path)

    # Find the images in the reconstruction
    image0_pycolmap = reconstruction.find_image_with_name(image0_name)
    image1_pycolmap = reconstruction.find_image_with_name(image1_name)

    # Get absolute poses (world to camera)
    T_w_c0_colmap = image0_pycolmap.cam_from_world.inverse()
    T_w_c1_colmap = image1_pycolmap.cam_from_world.inverse()
    
    # Get relative pose from pose0 to pose1 (for triangulation if needed)
    R01_gt, t01_gt = compute_relative_pose(T_w_c0_colmap, T_w_c1_colmap)

    # Get Transformation matrix from world to camera 0 in matrix form   
    T_c0_w_colmap = rigid3d_to_matrix4x4(image0_pycolmap.cam_from_world)
    
    # Calculate metrics
    rot_error = rotation_error_deg(R01_gt, R01_est)
    trans_error_deg = translation_error_direction_deg(t01_gt, t01_est)
    trans_error_rel = translation_error_relative_to_colmap(t01_gt, t01_est)
    
    # Scale to match COLMAP
    scale_factor = np.linalg.norm(t01_gt) / np.linalg.norm(t01_est)
    t01_est_scaled = t01_est * scale_factor

    # Create estimated 1->0 transformation
    T_c0_c1_est = pose_to_transformation_matrix(R01_est, t01_est_scaled)
    T_c1_c0_est = np.linalg.inv(T_c0_c1_est)

    # Convert to world coordinates: 
    T_c1_w_est = T_c1_c0_est @ T_c0_w_colmap
    # T_w_c1_est = np.linalg.inv(T_c1_w_est)
    R1_est = T_c1_w_est[:3, :3]
    t1_est = T_c1_w_est[:3, 3]
    
    # Create a new Model object for visualization
    vis_model = Model()
    vis_model.cameras = gt_model.cameras
    vis_model.points3D = gt_model.points3D
    
    # Create a filtered images dictionary with just camera 0
    vis_model.images = {image0.id: image0}
    
    # Initialize visualization
    vis_model.create_window()
    
    # Add 3D points from the reconstruction
    vis_model.add_points(min_track_len=3, remove_statistical_outlier=True)
    
    # Add camera 0 (ground truth) in blue
    vis_model.add_cameras(scale=camera_scale, color=[0, 0, 1])
    
    # Add camera 1 (ground truth) in blue
    vis_model.images = {image1.id: image1}
    vis_model.add_cameras(scale=camera_scale, color=[0.2, 0.8, 0.8])
    
    # Add camera 1 (estimated) in red
    # Create a new Image object with the estimated pose
    img1_est = Image(
        id=image1.id,
        qvec=rotmat2qvec(R1_est),
        tvec=t1_est,
        camera_id=image1.camera_id,
        name=image1.name,
        xys=image1.xys,
        point3D_ids=image1.point3D_ids
    )
    vis_model.images = {image1.id: img1_est}
    vis_model.add_cameras(scale=camera_scale * 0.8, color=[0.8, 0.2, 0.8])
    
    # Print error metrics
    print(f"Pair: {Path(image0_name).stem} -> {Path(image1_name).stem}")
    print(f"Rotation Error: {rot_error:.2f}°")
    print(f"Translation Error: {trans_error_deg:.2f}°")
    print(f"Translation Error Rel: {trans_error_rel:.2f}")
    if parallax is not None:
        print(f"Parallax: {parallax:.2f}°")
        
    # Show the visualization
    vis_model.show()
    
def parse_args():
    parser = argparse.ArgumentParser(description='Visualize a single camera pair')
    parser.add_argument('npz_path', type=str, help='Path to .npz file with matching results')
    parser.add_argument('colmap_model', type=str, help='Path to COLMAP model directory')
    parser.add_argument('--format', choices=['.bin', '.txt'], default='.txt', 
                        help='Format of COLMAP model files (.bin or .txt)')
    parser.add_argument('--camera-scale', type=float, default=0.25,
                        help='Scale factor for camera visualization (default: 0.25)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    visualize_camera_pair(args.npz_path, args.colmap_model, format=args.format, camera_scale=args.camera_scale)