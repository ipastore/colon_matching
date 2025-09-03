import os
import itertools
from matching import get_matcher
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from specular_mask import *
from match_image_pairs import *
import cv2
from my_logging import setup_logging, debug_log

from pathlib import Path
from matching.utils import get_default_device


############################# CONFIG #############################
############################# Logging ############################# 
DEBUG = True  # Global debug flag. Set to False to disable extra debug logging.
############################# CHOOSE LEVELS #############################
# levels = ['easy','medium','hard']
# levels = ['medium','hard']
levels = ['easy']
############################# CHOOSE SUBMAPS #############################
submaps_medium = [('093', '094'),('093', '095'), ('094', '095')]
submaps_hard = [('118', '093'), ('118', '094'), ('118', '095')]
############################# CHOOSE MODELS #############################
# models = ['superpoint-lg', 'sift-lg','tiny-roma', 'sift-nn', 'gim-lg']
# models = ['superpoint-lg', 'sift-lg']
# models = ['sift-nn']
# models = ['gim-lg']
# models = ['tiny-roma']
# models = ['sift-lg']
models = ['superpoint-lg']
# models = ['roma']
############################# CHOOSE DEVICE #############################
device = get_default_device()
############################# CHOOSE IMAGE DIRECTORY #############################
image_dir = Path(f'data')
############################# RESIZE #############################
# resize = 512
resize = None
############################# MASK #############################
# specular_mask = False
masking = True
############################# Key Points #############################
plot_kpts = True
############################# CONFIG #############################


############################# Logging ############################# 
DEBUG = True  # Global debug flag. Set to False to disable extra debug logging.
activated_debug_flags = {"match_image_pairs", "error_measurement"} #filter_image_pairs
############################# Logging #############################

logger = setup_logging(DEBUG, activated_debug_flags)

debug_log(logger, 'easy_medium_hard', 'Logging setup complete')
debug_log(logger, 'easy_medium_hard', f'Levels: {levels}')
debug_log(logger, 'easy_medium_hard', f'Submaps medium: {submaps_medium}')
debug_log(logger, 'easy_medium_hard', f'Submaps hard: {submaps_hard}')
debug_log(logger, 'easy_medium_hard', f'Models: {models}')
debug_log(logger, 'easy_medium_hard', f'Device: {device}')
debug_log(logger, 'easy_medium_hard', f'Root Image directory: {image_dir}')

# Easy case
if 'easy' in levels:
    logger.info('Starting easy case')
    image_dir1 = Path(f'{image_dir}/easy/095')
    image_paths = list(image_dir1.glob('*.png'))
    pairs = list(itertools.combinations(image_paths, 2))

    for model_name in models:
        logger.info(f'Starting model: {model_name} for easy case')
        # Create output directory for the model
        output_dir = Path(f'output/easy/{model_name}')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the matcher
        matcher = get_matcher(model_name, device=device)

        # Process each pair of images
        logger.info(f'Starting model: {model_name}')
        for img_path0, img_path1 in pairs:
            
            result = match_image_pairs(img_path0, img_path1, output_dir, model_name, matcher, logger, resize, masking, plot_kpts)

# Medium case
if 'medium' in levels:
    logger.info('Starting medium case')

    for submap1, submap2 in submaps_medium:
        logger.info(f'Starting submap pair: {submap1} and {submap2} for medium case')
        image_dir1 = Path(f'{image_dir}/medium/{submap1}')
        image_dir2 = Path(f'{image_dir}/medium/{submap2}')
        image_paths1 = list(image_dir1.glob('*.png'))
        image_paths2 = list(image_dir2.glob('*.png'))
        pairs = list(itertools.product(image_paths1, image_paths2))

        for model_name in models:
            logger.info(f'Starting model: {model_name} for submap pair: {submap1} and {submap2}')
            # Create output directory for the model
            output_dir = Path(f'output/medium/{submap1}_{submap2}/{model_name}')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize the matcher
            matcher = get_matcher(model_name, device=device)

            # Process each pair of images
            for img_path0, img_path1 in pairs:
                result = match_image_pairs(img_path0, img_path1, output_dir, model_name, matcher, logger, resize, masking, plot_kpts)

# Hard case
if 'hard' in levels:
    logger.info('Starting hard case')

    for submap1, submap2 in submaps_hard:
        logger.info(f'Starting submap pair: {submap1} and {submap2} for hard case')
        image_dir1 = Path(f'{image_dir}/hard/{submap1}')
        image_dir2 = Path(f'{image_dir}/medium/{submap2}')
        image_paths1 = list(image_dir1.glob('*.png'))
        image_paths2 = list(image_dir2.glob('*.png'))
        pairs = list(itertools.product(image_paths1, image_paths2))

        for model_name in models:
            logger.info(f'Starting model: {model_name} for submap pair: {submap1} and {submap2}')
            # Create output directory for the model
            output_dir = Path(f'output/hard/{submap1}_{submap2}/{model_name}')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize the matcher
            matcher = get_matcher(model_name, device=device)

            # Process each pair of images
            for img_path0, img_path1 in pairs:
                result = match_image_pairs(img_path0, img_path1, output_dir, model_name, matcher, logger, resize, masking, plot_kpts)
                
logger.info('Finished running models')