from matching import get_matcher, available_models
from matching.viz import *
import warnings
import torch
from pathlib import Path


warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
ransac_kwargs = {'ransac_reproj_thresh':3, 
                  'ransac_conf':0.95, 
                  'ransac_iters':2000} # optional ransac params
matcher = get_matcher(['superpoint-lg'], device=device, **ransac_kwargs) #try an ensemble!

asset_dir = Path('assets/example_pairs')
pairs = list(asset_dir.iterdir())
image_size = 512
for pair in pairs:
    pair = list(pair.iterdir())
    img0 = matcher.load_image(pair[0], resize=image_size)
    img1 = matcher.load_image(pair[1], resize=image_size)

    result = matcher(img0, img1)
    num_inliers, H, mkpts0, mkpts1 = result['num_inliers'], result['H'], result['inlier_kpts0'], result['inlier_kpts1']

    plot_matches(img0, img1, result)