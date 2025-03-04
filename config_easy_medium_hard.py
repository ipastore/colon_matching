from pathlib import Path
from matching.utils import get_default_device

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
models = ['sift-nn']
# models = ['gim-lg']
# models = ['tiny-roma']
# models = ['sift-lg']
# models = ['superpoint-lg']
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
