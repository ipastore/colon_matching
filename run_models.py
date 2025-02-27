import os
import itertools
from matching import get_matcher
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from specular_mask import *
from process_image_pairs import *
import cv2
from config import *
from my_logging import setup_logging


logger = setup_logging(DEBUG)

logger.debug('Logging setup complete')
logger.debug(f'Levels: {levels}')
logger.debug(f'Submaps medium: {submaps_medium}')
logger.debug(f'Submaps hard: {submaps_hard}')
logger.debug(f'Models: {models}')
logger.debug(f'Device: {device}')
logger.debug(f'Root Image directory: {image_dir}')


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
            
            process_image_pairs(img_path0, img_path1, output_dir, model_name, matcher, logger, resize, masking)

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
                process_image_pairs(img_path0, img_path1, output_dir, model_name, matcher, logger, resize, masking)

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
                process_image_pairs(img_path0, img_path1, output_dir, model_name, matcher, logger, resize, masking)
                
logger.info('Finished running models')