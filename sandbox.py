from matching import get_matcher
from matching.viz import plot_matches
import warnings

warnings.filterwarnings("ignore")

device = 'mps'  # 'cpu', 'cuda', 'mps'

matcher = get_matcher('superpoint-lg', device=device)  # Choose any of our ~30+ matchers listed below
img_size = 512  # optional

img0 = matcher.load_image('utils/image-matching-models/assets/example_pairs/outdoor/montmartre_close.jpg', resize=img_size)
img1 = matcher.load_image('utils/image-matching-models/assets/example_pairs/outdoor/montmartre_far.jpg', resize=img_size)

result = matcher(img0, img1)
num_inliers, H, inlier_kpts0, inlier_kpts1 = result['num_inliers'], result['H'], result['inlier_kpts0'], result['inlier_kpts1']
# result.keys() = ['num_inliers', 'H', 'all_kpts0', 'all_kpts1', 'all_desc0', 'all_desc1', 'matched_kpts0', 'matched_kpts1', 'inlier_kpts0', 'inlier_kpts1']
plot_matches(img0, img1, result, save_path='plot_matches.png')