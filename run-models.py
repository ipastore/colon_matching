from matching import get_matcher, available_models
from matching.utils import *
from matching.viz import *
from pathlib import Path
import torch
import warnings
import os
import matplotlib.pyplot as plt
import itertools

################### Code Snipet as example ###################

# warnings.filterwarnings("ignore")

# #Print current cwd
# print(os.getcwd())

# #Print device
# device = get_default_device()
# print(f'Using device: {device}')

# ransac_kwargs = {'ransac_reproj_thresh':3, 
#                   'ransac_conf':0.95, 
#                   'ransac_iters':2000} # optional ransac params

# matcher = get_matcher(['superpoint-lg'], device=device, **ransac_kwargs) #try an ensemble!

# #Print device
# device = get_default_device()
# print(f'Using device: {device}')

# ransac_kwargs = {'ransac_reproj_thresh':3, 
#                   'ransac_conf':0.95, 
#                   'ransac_iters':2000} # optional ransac params

# matcher = get_matcher(['superpoint-lg'], device=device, **ransac_kwargs) #try an ensemble!# Set the directory containing your images
# image_dir = Path('data/easy/095')
# image_paths = list(image_dir.glob('*.png'))
# pairs = list(itertools.combinations(image_paths, 2))

# image_size = 512
# plot_kpts = False

# for img_path0, img_path1 in pairs:
#     print(f"Matching {img_path0} with {img_path1}\n")
#     img0 = matcher.load_image(img_path0, resize=image_size)
#     img1 = matcher.load_image(img_path1, resize=image_size)

#     result = matcher(img0, img1)
#     num_inliers, H, mkpts0, mkpts1 = result['num_inliers'], result['H'], result['inlier_kpts0'], result['inlier_kpts1']

#     ax = plot_matches(img0, img1, result, show_matched_kpts=True, show_all_kpts=True)
#     plt.show()
    
#     if plot_kpts:
#         result0 = matcher.extract(img0)
#         result1 = matcher.extract(img1)

#         ax1 = plot_kpts(img0,result0)
#         ax2 = plot_kpts(img1,result1)
#         plt.show()

################### Code Snipet as example ###################



################### TASKS ###################
# Make iterative process to compare 3 models with the cases given in the README.md:

## Information about cases:

# Son 3 carpetas: easy, medium and hard

# - En easy hay 1 submapa, el objetivo es emparejar las imágenes entre sí. 
# - En medium hay 3 submapas de la misma secuencia, el objetivo es emparejar submapas de un submapa contra imágenes de otros submapas. 
# - En hard hay 1 submapa de otra secuencia, el objetivo es emparejar submapas de medium contra él (misma idea que el trabajo de Computer Vision)

# The folder structure is:

#Easy example, 1 image for the unique submap : /Users/ignaciopastorebenaim/Documents/MGRCV/COLON/colon_matching/data/easy/095/f_0064786.png
#medium example 1 omage for 1 of the 3 submaps: /Users/ignaciopastorebenaim/Documents/MGRCV/COLON/colon_matching/data/medium/093/f_0063782.png
#Hard example, 1 image for the unique submap : /Users/ignaciopastorebenaim/Documents/MGRCV/COLON/colon_matching/data/hard/f_0081950.png

# Save matching plots in structured folder as:
# -output
#   -easy
#       -Sp-Lg
#           -image_pair1 (as filename1_filename2_Sp-Lg.png)
#           -image_pairn (s filenamen_filenamem_Sp-Lg.png)
#       -SIFT-Lg
#           -image_pair1 (as filename1_filename2_SIFT-Lg.png)
#           -image_pairn (s filenamen_filenamem_SIFT-Lg.png)
#       -Roma
#           -image_pair1 (as filename1_filename2_Roma.png)
#           -image_pairn (s filenamen_filenamem_Roma.png)
#   -medium
#       -Sp-Lg
#           -image_pair1 (as filename1_filename2_Sp-Lg.png)
#           -image_pairn (s filenamen_filenamem_Sp-Lg.png)
#       -SIFT-Lg
#           -image_pair1 (as filename1_filename2_SIFT-Lg.png)
#           -image_pairn (s filenamen_filenamem_SIFT-Lg.png)
#       -Roma
#           -image_pair1 (as filename1_filename2_Roma.png)
#           -image_pairn (s filenamen_filenamem_Roma.png)
#   -Hard
#       -Sp-Lg
#           -image_pair1 (as filename1_filename2_Sp-Lg.png)
#           -image_pairn (s filenamen_filenamem_Sp-Lg.png)
#       -SIFT-Lg
#           -image_pair1 (as filename1_filename2_SIFT-Lg.png)
#           -image_pairn (s filenamen_filenamem_SIFT-Lg.png)
#       -Roma
#           -image_pair1 (as filename1_filename2_Roma.png)
#           -image_pairn (s filenamen_filenamem_Roma.png)

################### TASKS ###################
import os
import itertools
from matching import get_matcher
from matching.utils import get_default_device
from matching.viz import plot_matches
from pathlib import Path
import matplotlib.pyplot as plt
import logging

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
file_handler = logging.FileHandler('no_matches.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)

# Define the directories and models
# levels = ['easy', 'medium, hard']
levels = ['hard']
models = ['superpoint-lg', 'sift-lg', 'tiny-roma']

# Easy case
if 'easy' in levels:
    image_dir = Path('data/easy/095')
    image_paths = list(image_dir.glob('*.png'))
    pairs = list(itertools.combinations(image_paths, 2))

    for model_name in models:
        # Create output directory for the model
        output_dir = Path(f'output/easy/{model_name}')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the matcher
        device = get_default_device()
        matcher = get_matcher(model_name, device=device)

        # Process each pair of images
        for img_path0, img_path1 in pairs:
            print(f"Matching {img_path0} with {img_path1} using {model_name}")
            img0 = matcher.load_image(img_path0, resize=512)
            img1 = matcher.load_image(img_path1, resize=512)

            result = matcher(img0, img1)
            if len(result['matched_kpts1']) == 0:
                logging.info(f'No matches found for pair: {img_path0} and {img_path1} using {model_name} in easy level')
                continue
            plot_path = output_dir / f'{img_path0.stem}_{img_path1.stem}_{model_name}.png'
            plot_matches(img0, img1, result, save_path=plot_path)
            print(f'Saved plot to {plot_path}')

# Medium case
if 'medium' in levels:
    # submaps = [('093', '094'), ('093', '095'), ('094', '095')]
    submaps = [('093', '094'), ('093', '095')]

    for submap1, submap2 in submaps:
        image_dir1 = Path(f'data/medium/{submap1}')
        image_dir2 = Path(f'data/medium/{submap2}')
        image_paths1 = list(image_dir1.glob('*.png'))
        image_paths2 = list(image_dir2.glob('*.png'))
        pairs = list(itertools.product(image_paths1, image_paths2))

        for model_name in models:
            # Create output directory for the model
            output_dir = Path(f'output/medium/{submap1}_{submap2}/{model_name}')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize the matcher
            device = get_default_device()
            matcher = get_matcher(model_name, device=device)

            # Process each pair of images
            for img_path0, img_path1 in pairs:
                print(f"Matching {img_path0} with {img_path1} using {model_name}")
                img0 = matcher.load_image(img_path0, resize=512)
                img1 = matcher.load_image(img_path1, resize=512)

                result = matcher(img0, img1)
                if len(result['matched_kpts1']) == 0:
                    print(f'No matches found for pair: {img_path0} and {img_path1} using {model_name}')
                    logging.info(f'No matches found for pair: {img_path0} and {img_path1} using {model_name} in medium level with {submap1} and {submap2}')
                    continue
                # plot_path = output_dir / f'{img_path0.stem}_{img_path1.stem}_{model_name}.png'
                # plot_matches(img0, img1, result, save_path=plot_path)
                # print(f'Saved plot to {plot_path}')
                print(f'Done matching {img_path0} with {img_path1} using {model_name}')

# Hard case
if 'hard' in levels:
    submaps = [('118', '093'), ('118', '094'), ('118', '095')]

    for submap1, submap2 in submaps:
        image_dir1 = Path(f'data/hard/{submap1}')
        image_dir2 = Path(f'data/medium/{submap2}')
        image_paths1 = list(image_dir1.glob('*.png'))
        image_paths2 = list(image_dir2.glob('*.png'))
        pairs = list(itertools.product(image_paths1, image_paths2))

        for model_name in models:
            # Create output directory for the model
            output_dir = Path(f'output/hard/{submap1}_{submap2}/{model_name}')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize the matcher
            device = get_default_device()
            matcher = get_matcher(model_name, device=device)

            # Process each pair of images
            for img_path0, img_path1 in pairs:
                print(f"Matching {img_path0} with {img_path1} using {model_name}")
                img0 = matcher.load_image(img_path0, resize=512)
                img1 = matcher.load_image(img_path1, resize=512)

                result = matcher(img0, img1)
                if len(result['matched_kpts1']) == 0:
                    logging.info(f'No matches found for pair: {img_path0} and {img_path1} using {model_name} in hard level with {submap1} and {submap2}')
                    continue
                plot_path = output_dir / f'{img_path0.stem}_{img_path1.stem}_{model_name}.png'
                plot_matches(img0, img1, result, save_path=plot_path)
                print(f'Saved plot to {plot_path}')