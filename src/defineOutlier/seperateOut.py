import os
import csv
import os
from glob import glob
from skimage.io import imread, imsave
import numpy as np


work_space = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(work_space)
path_to_train = work_space+'/data/train/'
outlier_path = work_space+'/data/647outlier/'
dicescore_csv = work_space+'/data/cp-id4-37-sb-coef.csv'


def move_outlier():
    csv_file = open(dicescore_csv, 'r')
    all_lines = csv_file.readlines()
    for line in all_lines[0:646]:
        list = line.split(',')
        img_id = list[0]
        img = imread(path_to_train+img_id+'_sat.jpg')
        imsave(outlier_path+img_id+'_out.jpg', img)
        imsave(outlier_path + img_id + 'a_out.jpg', img[:,::-1])
        imsave(outlier_path + img_id + 'b_out.jpg', img[::-1,:])
        os.remove(path_to_train + img_id + '_sat.jpg')

def choose_normal():
    normal_paths = glob(path_to_train + "/*_sat.jpg")
    normal_imgs = np.random.choice(normal_paths, 3*647)
    for path in normal_imgs:
        img = imread(path)
        img_basename = os.path.basename(path)
        img_id = os.path.splitext(img_basename)[0][:-len('_sat')]
        imsave(outlier_path+img_id+'_sat.jpg', img)


if __name__ == '__main__':
    move_outlier()
    choose_normal()



