from keras.models import load_model
from skimage.io import imread, imsave
from skimage.transform import resize
import pandas as pd
import os
import numpy as np
import keras.backend as K
from glob import glob
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

smooth = 1e-9

# This is the competition metric implemented using Keras
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * (K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

# We'll construct a Keras Loss that incorporates the DICE score
def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def bce_dice_loss(y_true, y_pred):
    return 0.4 * binary_crossentropy(y_true, y_pred) + 0.6 * dice_loss(y_true, y_pred)

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
        # Load the image and mask, and normalize it to 0-1 range
        color_img = imread(img_path)
        gray_img = color_img / 255.
        # Resize the images
        gray_img = resize(gray_img, img_size, preserve_range=True)
        if gen_mask:
            mask_path = os.path.join(os.path.join(os.path.dirname(img_path), img_id + '_msk.png'))
            mask = imread(mask_path, as_gray=True)
            mask = resize(mask, img_size, mode='constant', preserve_range=True)
            # Turn the mask back into a 0-1 mask
            mask = (mask >= 0.5).astype(float)
        else:
            mask = []
        # Yield the image mask pair
        yield color_img, gray_img, mask, img_id


def visualize(model_path, img_path, save_path, on_train = True):
    '''
    This method will load the model and predict data in img_path, and it will rank the image id based on dice_score
    :param model_path: the path of the model you want to use
    :param img_path: the data you want the model to predict
    :param save_path: the path to save the predicted result
    :param on_train: whether predict on training set or test set
    :return:
    '''
    score_board_coef = pd.DataFrame(columns={'ImageId','dice_coef'})
    score_board_loss = pd.DataFrame(columns={'ImageId', 'dice_loss'})
    image_id_list = []
    dice_score_list = []
    loss_list = []
    # load the model from model path
    model = load_model(model_path, custom_objects={'bce_dice_loss':bce_dice_loss,'dice_coef':dice_coef})
    image_to_predict = os.path.join(img_path, '*_sat.jpg')
    val_img_paths = glob(image_to_predict)
    if on_train:
        ig = image_gen(val_img_paths, gen_mask=True)
    else:
        ig = image_gen(val_img_paths, gen_mask=False)

    for color_img, gray_img, mask, img_id in ig:
        gray_img = np.array([gray_img])
        print(gray_img.shape)
        # First make the prediction
        result = model.predict(gray_img)
        if len(mask) != 0:
            # get the dice score
            dice_score = dice_coef(mask, result)
            loss = dice_loss(mask, result)
            image_id_list.append(img_id)
            dice_score_list.append(dice_score)
            loss_list.append(loss)

            if os.path.exists(save_path):
                mask_path = os.path.join(save_path, img_id + '_msk.png')
                predict_mask_path = os.path.join(save_path, img_id + '_pre_msk.png')
                imsave(mask_path, mask)
                imsave(predict_mask_path, result)
            else:
                raise Exception('path not found!')
        else:
            if os.path.exists(save_path):
                origin_path = os.path.join(save_path, img_id + '_sat.jpg')
                predict_mask_path = os.path.join(save_path, img_id + '_pre_msk.png')
                imsave(predict_mask_path, result)
                imsave(origin_path, color_img)
            else:
                raise Exception('path not found!')
    score_board_coef['ImageId'] = image_id_list
    score_board_loss['ImageId'] = image_id_list
    score_board_coef['dice_coef'] = dice_score_list
    score_board_loss['dice_loss'] = loss_list
