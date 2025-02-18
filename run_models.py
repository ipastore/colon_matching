import os
import itertools
from matching import get_matcher
from matching.utils import get_default_device
from pathlib import Path
import matplotlib.pyplot as plt
from my_logging import setup_logging
import numpy as np
from specular_mask import *
from process_image_pairs import process_image_pairs
import cv2

########################################################## CONFIG ##########################################################
logger = setup_logging()
############################# CHOOSE LEVELS #############################
# levels = ['easy','medium','hard']
# levels = ['medium','hard']
levels = ['easy']
############################# CHOOSE SUBMAPS #############################
submaps_medium = [('093', '094'),('093', '095'), ('094', '095')]
submaps_hard = [('118', '093'), ('118', '094'), ('118', '095')]
############################# CHOOSE MODELS #############################
# models = ['superpoint-lg', 'sift-lg','tiny-roma', 'sift-nn']
# models = ['superpoint-lg', 'sift-lg']
models = ['sift-nn']
########################################################## CONFIG ##########################################################
image_dir = Path(f'data')

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
        device = get_default_device()
        matcher = get_matcher(model_name, device=device)

        # Process each pair of images
        logger.info(f'Starting model: {model_name}')
        for img_path0, img_path1 in pairs:
            
            process_image_pairs(img_path0,img_path1, output_dir, model_name, matcher, logger)

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
            device = get_default_device()
            matcher = get_matcher(model_name, device=device)

            # Process each pair of images
            for img_path0, img_path1 in pairs:
                process_image_pairs(img_path0,img_path1, output_dir, model_name, matcher, logger)
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
            device = get_default_device()
            matcher = get_matcher(model_name, device=device)

            # Process each pair of images
            for img_path0, img_path1 in pairs:
                process_image_pairs(img_path0,img_path1, output_dir, model_name, matcher, logger)
logger.info('Finished running models')