from pathlib import Path
import pycolmap
from pycolmap import CameraModelId, Camera

def main():
    
    # Define paths
    colmap_path = Path("./data/seq_001_submap6")  
    image_path = colmap_path       
    database_path = colmap_path / "database.db"
    # camera_mask_path = "/path/to/camera_mask.png"
    mask_path = colmap_path / "masks"  # Path to the mask directory

    # Camera parameters for OPENCV_FISHEYE model
    fx, fy, cx, cy = 717.21, 717.482, 735.357, 552.798
    k1, k2, k3, k4 = -0.138927, -0.00123961, 0.000912582, -4.07161e-05

    # k1, k2, k3, k4 = 0, 0, 0, 0  # Set to zero if not used
    # camera_params_list = [fx, fy, cx, cy, k1, k2, k3, k4]

    # # Build space-separated string for ImageReaderOptions
    # camera_params_str = ' '.join(str(p) for p in camera_params_list)  

    camera_params = f"{fx},{fy},{cx},{cy},{k1},{k2},{k3},{k4}"
    width= 1440
    height = 1080  # Set your actual image width and height

    # 0. Create a new database
    db = pycolmap.Database()
    db.open(str(database_path))
    camera = Camera(
        model = CameraModelId.OPENCV_FISHEYE,
        width = width,  # set your actual image width
        height = height,  # set your actual image height
    )
    camera.set_params_from_string(camera_params)  # "fx,fy,cx,cy,k1,k2,k3,k4"
    
    # Check
    if not camera.verify_params():
        raise ValueError("Camera parameters are invalid!")

    print("New Camera to Insert:")
    print(camera.summary())

    camera_id = db.write_camera(camera)
    print(f"Inserted new camera with ID: {camera_id}")

    db.close()
    

    # 1. Feature extraction
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.existing_camera_id = camera_id
    # reader_options.camera_mask_path = camera_mask_path
    reader_options.mask_path = str(mask_path)

    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.peak_threshold = 0.00067
    sift_options.edge_threshold = 50
    sift_options.max_image_size = 1000

    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_path),
        reader_options=reader_options,
        sift_options=sift_options,
    )

    # print camera_params from db
    db = pycolmap.Database()
    db.open(str(database_path))
    camera = db.read_camera(camera_id)
    db.close()
    print(f"Camera parameters: {camera.params}")
    print(f"Camera model: {camera.model}")
    

    # 2. Exhaustive matching
    sift_matching_options = pycolmap.SiftMatchingOptions()
    sift_matching_options.guided_matching = True

    pycolmap.match_exhaustive(
        database_path=str(database_path),
        sift_options=sift_matching_options
    )

    # 3. Incremental mapping (Sparse reconstruction)
    mapping_options = pycolmap.IncrementalPipelineOptions()
    mapping_options.triangulation.ignore_two_view_tracks = True
    mapping_options.ba_refine_focal_length = False
    mapping_options.ba_refine_principal_point = False
    mapping_options.ba_refine_extra_params = False
    mapping_options.mapper.init_min_tri_angle = 2

    pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_path),
        output_path=str(colmap_path),
        options=mapping_options,
    )
        
    return


if __name__ == "__main__":
    main()