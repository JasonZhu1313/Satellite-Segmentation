import sys
import os

workspace = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'defineOutlier')
sys.path.append(workspace)
from overLapMaskOnImg import *
from seperateOut import *
from glob import glob
from skimage.io import imread,imsave

# get the training path
train_paths = glob(os.path.join('../../data/train', '*_sat.jpg'))

ig = img_gen(train_paths, '../../data')

for i in range(len(train_paths)):
    img, mask, img_id, origin_img, origin_mask = next(ig)
    # augment the image and save it into the same folder
    # save the vertical flip
    imsave(os.path.join('../../data/train', img_id+'_aug1_sat.jpg'), origin_img[:,::-1])
    imsave(os.path.join('../../data/train', img_id + '_aug1_msk.png'), origin_mask[:,::-1])
    # save the horizontal flip
    imsave(os.path.join('../../data/train', img_id+'_aug2_sat.jpg'), origin_img[::-1,:])
    imsave(os.path.join('../../data/train', img_id + '_aug2_msk.png'), origin_mask[::-1,:])

# remove the outlier data from the existing folder
move_outlier()
