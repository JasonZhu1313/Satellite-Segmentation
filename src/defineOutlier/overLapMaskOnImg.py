import os
# img generator
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from PIL import Image
import numpy as np
from glob import glob
import random

import matplotlib.pyplot as plt
import os
work_space = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(work_space)
def get_img_id(img_path):
    img_basename = os.path.basename(img_path)
    img_id = os.path.splitext(img_basename)[0][:-len('_sat')]
    return img_id

def img_gen(img_paths, path, img_size=(512, 512)):
    # Iterate over all image paths
    for img_path in img_paths:
        img_id = get_img_id(img_path)
        mask_path = os.path.join(path, img_id + '_msk.png')
        img = imread(img_path)/256
        #img = Image.open(img_path).convert('L')

        mask = rgb2gray(imread(mask_path))
        mask = (mask >= 0.5).astype(float)
        mask = np.reshape(mask, (512, 512, 1))
        yield img, mask, img_id

def overlap_img(img, mask):
    overLap= (img + mask)/2;
    return overLap


def main():
    train_path = work_space+'/data/train/'
    glob_train_imgs = os.path.join(train_path, '*_sat.jpg')


    glob_train_masks = os.path.join(train_path, '*_msk.png')

    train_img_paths = glob(glob_train_imgs)
    print(train_img_paths)
    # train_img_paths = random.shuffle(train_img_paths)
    train_mask_paths = glob(glob_train_masks)
    ig = img_gen(train_img_paths, train_path)

    for j in range(0, len(glob_train_imgs)-1, 10):
        fig, axes = plt.subplots(2, 10, figsize = (80,50))

        for i in range(0, 10):
            img, mask, img_id = next(ig)
            overLap = overlap_img(img, mask)
            axes[0,i].imshow(overLap)
            axes[0,i].set_title(str(img_id))
            overLap = np.array(overLap*256)
            arr = overLap.flatten()
            # n, bins, patches = plt.hist(arr, bins=256, normed=0, facecolor='green', alpha=0.75)
            axes[1, i].hist(arr, bins=256, normed=0, facecolor='green', alpha=0.75)

        plt.savefig(work_space+"/data/hist/"+str(j)+".jpg")
        # plt.show()


if __name__ == "__main__":
    main()