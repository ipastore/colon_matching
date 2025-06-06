#!/bin/bash

# Set your paths here
COLMAP_PATH="./data/seq_001_v2/seq_001_submap6"
IMAGE_PATH="${COLMAP_PATH}/img_train"
MASK_PATH="${COLMAP_PATH}/masks"
CAMERA_PARAMS="717.21,717.482,735.357,552.798,-0.138927,-0.00123961,0.000912582,-4.07161e-05"

# Step 1: Feature extraction
GLOG_v=2 colmap feature_extractor \
   --database_path "${COLMAP_PATH}/database.db" \
   --image_path "${IMAGE_PATH}"  \
   --ImageReader.single_camera 1 \
   --ImageReader.camera_model OPENCV_FISHEYE \
   --ImageReader.camera_params "${CAMERA_PARAMS}" \
   --SiftExtraction.peak_threshold 0.00067 \
   --SiftExtraction.edge_threshold 50 \
   --SiftExtraction.max_image_size 1000 \
   --ImageReader.mask_path "${MASK_PATH}"

# Step 2: Exhaustive matching
GLOG_v=3 colmap exhaustive_matcher \
   --database_path "${COLMAP_PATH}/database.db" \
   --SiftMatching.guided_matching 1 \
   --log_to_stderr 1 \
   --log_level 2

# Step 3: Mapping
GLOG_v=2 colmap mapper \
   --database_path "${COLMAP_PATH}/database.db" \
   --image_path "${IMAGE_PATH}" \
   --output_path "${COLMAP_PATH}/sparse" \
   --Mapper.tri_ignore_two_view_tracks 1 \
   --Mapper.ba_refine_focal_length 0 \
   --Mapper.ba_refine_principal_point 0 \
   --Mapper.ba_refine_extra_params 0 \
   --Mapper.init_min_tri_angle 2
