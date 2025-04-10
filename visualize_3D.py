import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pycolmap
import cv2
from colmap_opencv_helpers import compute_relative_pose, translation_error_direction_deg, rotation_error_deg

def load_result_matcher_npz(npz_path: Path) -> dict:
    """
    Load result_matcher and metadata from .npz file.
    """
    data = np.load(npz_path, allow_pickle=True)
    return {
        "matched_kpts0": data["matched_kpts0"],
        "matched_kpts1": data["matched_kpts1"],
        "all_kpts0": data["all_kpts0"],
        "all_kpts1": data["all_kpts1"],
        "scores": data.get("scores", None),
        "inliers": data.get("inliers", None),
        "image0": str(data["image0"]),
        "image1": str(data["image1"]),
        "R_est": data.get("R_est", None),
        "t_est": data.get("t_est", None)
    }

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def triangulate_points(kpts0, kpts1, K0, K1, D0, D1, R, t):
    """
    Triangulate 3D points from matched keypoints and camera parameters
    
    Args:
        kpts0, kpts1: Matched keypoints in pixel coordinates (Nx2)
        K0, K1: Camera intrinsic matrices (3x3)
        R: Rotation matrix from camera 0 to camera 1 (3x3)
        t: Translation vector from camera 0 to camera 1 (3,)
    
    Returns:
        points_3d: Triangulated 3D points in camera 0 coordinate system (Nx3)
    """
    # Normalize keypoints by inverting the camera intrinsics
    # This converts pixel coordinates to normalized coordinates
    kpts0_norm = cv2.undistortPoints(kpts0.reshape(-1, 1, 2), K0, D0)
    kpts1_norm = cv2.undistortPoints(kpts1.reshape(-1, 1, 2), K1, D1)
    
    # Create projection matrices
    P0 = np.eye(3, 4)  # [I|0] for first camera
    P1 = np.hstack((R, t.reshape(3, 1)))  # [R|t] for second camera
    
    # Triangulate points
    points_4d = cv2.triangulatePoints(P0, P1, kpts0_norm, kpts1_norm)
    
    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d[:3] / points_4d[3]
    
    return points_3d.T  # Return as Nx3

def pose_to_transformation_matrix(R, t):
    """
    Convert rotation matrix and translation vector to 4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T

def visualize_3d_reconstruction(npz_path, sparse_model_dir, show_triangulated=False):
    """
    Visualize 3D reconstruction comparing COLMAP points with optional triangulated points
    
    Args:
        npz_path: Path to .npz file with matching results
        sparse_model_dir: Path to COLMAP sparse reconstruction directory
        show_triangulated: Whether to show triangulated points (default: False)
    """
    # Load matching results
    match_data = load_result_matcher_npz(Path(npz_path))
    
    # Extract data from match_data
    matched_kpts0 = match_data["matched_kpts0"]
    matched_kpts1 = match_data["matched_kpts1"]
    inliers_mask = match_data["inliers"]
    R01_est = match_data["R_est"]
    t01_est = match_data["t_est"]
    image0_name = match_data["image0"]
    image1_name = match_data["image1"]
    
    if inliers_mask is None:
        print("No inliers mask found in the data. Assuming all points are inliers.")
        inliers_mask = np.ones(len(matched_kpts0), dtype=bool)
    
    # Load COLMAP reconstruction
    reconstruction = pycolmap.Reconstruction(sparse_model_dir)
    
    # Find the images in the reconstruction
    image0 = reconstruction.find_image_with_name(image0_name)
    image1 = reconstruction.find_image_with_name(image1_name)
    
    if image0 is None or image1 is None:
        print(f"Could not find images in the reconstruction: {image0_name}, {image1_name}")
        return
    
    # Get camera objects
    camera0 = reconstruction.camera(image0.camera_id)
    camera1 = reconstruction.camera(image1.camera_id)
    
    # Get intrinsic matrices
    K0 = camera0.calibration_matrix()
    K1 = camera1.calibration_matrix()
    
    # Get distortion coefficients (if any)
    if camera0.model_name() in ["OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]:
        D0 = np.array(camera0.params[4:])
        D1 = np.array(camera1.params[4:])
    else:
        D0 = None
        D1 = None
    
    # Get absolute poses (world to camera)
    T_w_c0_colmap = image0.cam_from_world.inverse().matrix()
    T_w_c1_colmap = image1.cam_from_world.inverse().matrix()
    
    # Get relative pose from pose0 to pose1 (for triangulation if needed)
    R01_colmap, t01_colmap = compute_relative_pose(T_w_c0_colmap, T_w_c1_colmap)
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw world coordinate system
    world_frame = np.eye(4)
    drawRefSystem(ax, world_frame, '-', 'World')
    
    # Draw camera coordinate systems in world frame
    T_c0_w_colmap = T_w_c0_colmap.inverse().matrix()
    T_c1_w_colmap = T_w_c1_colmap.inverse().matrix()
    
    drawRefSystem(ax, T_c0_w_colmap, '-', 'Cam0 (COLMAP)')
    drawRefSystem(ax, T_c1_w_colmap, '-', 'Cam1 (COLMAP)')
    
    # If we have estimated pose, convert to world coordinates and plot
    if R01_est is not None and t01_est is not None:
        # Create estimated 1->0 transformation
        T_c0_c1_est = pose_to_transformation_matrix(R01_est, t01_est)
        T_c1_c0_est = np.linalg.inv(T_c0_c1_est)
        
        # Convert to world coordinates: 
        T_c1_w_est = T_c1_c0_est @ T_c0_w_colmap 
        
        drawRefSystem(ax, T_c1_w_est, '--', 'Cam1 (Est)')
    
    # Extract COLMAP 3D points visible in the images
    points0 = {pt.point3D_id for pt in image0.points2D if pt.has_point3D()}
    points1 = {pt.point3D_id for pt in image1.points2D if pt.has_point3D()}
    
    # Points visible in both images
    shared_points = points0.intersection(points1)
    
    # Points visible only in image 0
    only0_points = points0 - shared_points
    
    # Points visible only in image 1
    only1_points = points1 - shared_points
    
    # Collect and plot shared points
    shared_points_xyz = []
    for pid in shared_points:
        point_world = reconstruction.points3D[pid].xyz
        shared_points_xyz.append(point_world)
    
    if shared_points_xyz:
        shared_points_xyz = np.array(shared_points_xyz)
        ax.scatter(shared_points_xyz[:, 0], shared_points_xyz[:, 1], shared_points_xyz[:, 2], 
                  c='blue', marker='.', s=5, alpha=0.7, label='COLMAP shared points')
    
    # Collect and plot points only in image 0
    only0_points_xyz = []
    for pid in only0_points:
        point_world = reconstruction.points3D[pid].xyz
        only0_points_xyz.append(point_world)
    
    if only0_points_xyz:
        only0_points_xyz = np.array(only0_points_xyz)
        ax.scatter(only0_points_xyz[:, 0], only0_points_xyz[:, 1], only0_points_xyz[:, 2], 
                  c='cyan', marker='.', s=3, alpha=0.5, label='COLMAP points (only cam0)')
    
    # Collect and plot points only in image 1
    only1_points_xyz = []
    for pid in only1_points:
        point_world = reconstruction.points3D[pid].xyz
        only1_points_xyz.append(point_world)
    
    if only1_points_xyz:
        only1_points_xyz = np.array(only1_points_xyz)
        ax.scatter(only1_points_xyz[:, 0], only1_points_xyz[:, 1], only1_points_xyz[:, 2], 
                  c='magenta', marker='.', s=3, alpha=0.5, label='COLMAP points (only cam1)')
    
    if show_triangulated and inliers_mask is not None:
        # Extract only inlier matches for triangulation
        inlier_kpts0 = matched_kpts0[inliers_mask]
        inlier_kpts1 = matched_kpts1[inliers_mask]
        
        if len(inlier_kpts0) > 0:
            # Triangulate points using the COLMAP relative pose
            points_3d_tri = triangulate_points(inlier_kpts0, inlier_kpts1, K0, K1, D0, D1, R01_colmap, t01_colmap)
            
            # Convert triangulated points to world coordinates
            points_3d_world = []
            for pt in points_3d_tri:
                # Transform from camera 0 to world
                pt_homogeneous = np.append(pt, 1)
                pt_world = T_w_c0_colmap @ pt_homogeneous
                points_3d_world.append(pt_world[:3])
            
            points_3d_world = np.array(points_3d_world)
            
            # Plot triangulated points
            ax.scatter(points_3d_world[:, 0], points_3d_world[:, 1], points_3d_world[:, 2], 
                      c='green', marker='.', s=10, label='Triangulated points')
            
            if R01_est is not None and t01_est is not None:
                # Also triangulate using estimated pose
                points_3d_est = triangulate_points(inlier_kpts0, inlier_kpts1, K0, K1, D0, D1, R01_est, t01_est)
                
                # Convert to world coordinates
                points_est_world = []
                for pt in points_3d_est:
                    pt_homogeneous = np.append(pt, 1)
                    pt_world = T_w_c0_colmap @ pt_homogeneous
                    points_est_world.append(pt_world[:3])
                
                points_est_world = np.array(points_est_world)
                
                # Plot estimated triangulated points
                ax.scatter(points_est_world[:, 0], points_est_world[:, 1], points_est_world[:, 2], 
                          c='orange', marker='.', s=10, label='Est. triangulated points')
    
    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Reconstruction: {Path(image0_name).stem} -> {Path(image1_name).stem}')
    
    # Add legend
    ax.legend()
    
    # Adjust view to see the scene better
    ax.view_init(elev=30, azim=-60)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Print camera pose errors if available
    if R01_est is not None and t01_est is not None:
        rot_error_deg = rotation_error_deg(R01_colmap, R01_est)
        trans_error_deg = translation_error_direction_deg(t01_colmap, t01_est)
        
        print(f"Rotation error: {rot_error_deg:.2f} degrees")
        print(f"Translation direction error: {trans_error_deg:.2f} degrees")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize 3D reconstruction from matched keypoints')
    parser.add_argument('npz_path', type=str, help='Path to the .npz file with matching results')
    parser.add_argument('colmap_path', type=str, help='Path to the COLMAP sparse reconstruction directory')
    parser.add_argument('--show-triangulated', action='store_true', help='Show triangulated points')
    
    args = parser.parse_args()
    
    # Default values for flags when running the script directly
    npz_path = args.npz_path if args.npz_path else "path/to/default.npz"
    colmap_path = args.colmap_path if args.colmap_path else "path/to/default/colmap"
    show_triangulated = args.show_triangulated if hasattr(args, 'show_triangulated') else False
    
    # Allow tuning flags directly in the script
    if not args.npz_path or not args.colmap_path:
        print("Running with default paths and flags. Modify below if needed.")
        npz_path = "path/to/default.npz"
        colmap_path = "path/to/default/colmap"
        show_triangulated = True  # Set to False to disable
    
    visualize_3d_reconstruction(
        npz_path,
        colmap_path,
        show_triangulated=show_triangulated
    )

if __name__ == "__main__":
    main()