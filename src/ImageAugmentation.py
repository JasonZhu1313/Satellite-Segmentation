import os
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np

def get_img_id(img_path):
    img_basename = os.path.basename(img_path)
    img_id = os.path.splitext(img_basename)[0][:-len('_sat')]
    return img_id


# Write it like a normal function
def image_gen(img_paths, img_size=(512, 512), gen_mask=True):
    # Iterate over all the image paths
    for img_path in img_paths:

        # Construct the corresponding mask path
        img_id = get_img_id(img_path)
        mask_path = os.path.join(os.path.join('../data/train/',img_id + '_msk.png'))

        # Load the image and mask, and normalize it to 0-1 range
        color_img = imread(img_path)

        gray_img = color_img / 255.

        # Resize the images
        color_img = resize(color_img, img_size, preserve_range=True)
        gray_img = resize(gray_img, img_size, preserve_range=True)

        if gen_mask:
            mask = imread(mask_path, as_gray=True)
            mask = resize(mask, img_size, mode='constant', preserve_range=True)
            # Turn the mask back into a 0-1 mask
            mask = (mask >= 0.5).astype(float)
        else:
            mask = []

        # Yield the image mask pair
        yield gray_img, mask, img_id


def aug_outlier(path):
    '''
    This function iterate the training set and select outlier and augmented outlier image and save to path
    :param path: the path to save augmented image
    '''
    glob_train_imgs = '*_sat.jpg'
    glob_train_masks = '*_msk.png'
    glob_train_imgs_path = os.path.join('../data/train/',glob_train_imgs)
    glob_train_masks_path = os.path.join('../data/train/',glob_train_masks)
    train_img_paths = glob(glob_train_imgs_path)
    train_mask_paths = glob(glob_train_masks_path)
    ig = image_gen(train_img_paths)
    for i in range(len(train_img_paths) - 1):
        print "a"
        train_img, mask, img_id = next(ig)
        # if the image is outlier
        if len(np.where(mask[:,:,0].T.flatten()!=0)[0] ) == 0:
            horizontal_flip = train_img[:, ::-1]
            vertical_flip = train_img[::-1, :]
            hor_file_name = img_id + '_hor'+'_sat.jpg'
            ver_file_name = img_id + '_ver'+'_sat.jpg'
            os.remove(os.path.join('../data/train/',img_id + '_sat.jpg'))
            if os.path.exists(path):
                imsave(os.path.join(path, hor_file_name), horizontal_flip)
                imsave(os.path.join(path, ver_file_name), vertical_flip)
                imsave(os.path.join(path, img_id + '_sat.jpg'), train_img)
                break
            else:
                print "path not exist..."

if __name__ == '__main__':
    aug_outlier('../data/aug/')