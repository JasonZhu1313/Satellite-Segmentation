import os
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
from skimage.color import rgb2gray


def get_img_id(img_path):
    img_basename = os.path.basename(img_path)
    img_id = os.path.splitext(img_basename)[0][:-len('_sat')]
    return img_id


def img_gen_1(img_paths, img_size=(512, 512)):
    # Iterate over all image paths
    for img_path in img_paths:
        img_id = get_img_id(img_path)
        mask_path = os.path.join('../data/train/', img_id + '_msk.png')

        img = imread(img_path) / 255
        mask = rgb2gray(imread(mask_path))

        #     img = resize(img, img_size, preserve_range = True)
        #     mask = resize(mask, img_size, mode='constant', preserve_range = True)

        mask = (mask >= 0.5).astype(float)
        mask = np.reshape(mask, (512, 512, 1))
        yield img, mask, img_id

def aug_outlier(path):
    '''
    This function iterate the training set and select outlier and augmented outlier image and save to path
    :param path: the path to save augmented image
    '''
    glob_train_imgs = '*_sat.jpg'
    glob_train_masks = '*_msk.png'
    glob_train_imgs_path = os.path.join('../data/train/', glob_train_imgs)
    glob_train_masks_path = os.path.join('../data/train/', glob_train_masks)
    train_img_paths = glob(glob_train_imgs_path)
    train_mask_paths = glob(glob_train_masks_path)
    ig = img_gen_1(train_img_paths)
    for i in range(len(train_img_paths) - 1):
        if i % 500 == 0:
            print(i)
        train_img, mask, img_id = next(ig)
        # if the image is outlier
        if len(np.where(mask[:, :, 0].T.flatten() != 0)[0]) == 0:

            horizontal_flip = train_img[:, ::-1]
            vertical_flip = train_img[::-1, :]
            hor_file_name = img_id + '_hor' + '_sat.jpg'
            ver_file_name = img_id + '_ver' + '_sat.jpg'
            os.remove(os.path.join('../data/train/', img_id + '_sat.jpg'))
            os.remove(os.path.join('../data/train/', img_id + '_msk.png'))
            
            if os.path.exists(path):
                imsave(os.path.join(path, hor_file_name), horizontal_flip)
                imsave(os.path.join(path, ver_file_name), vertical_flip)
                imsave(os.path.join(path, img_id + '_sat.jpg'), train_img)

            else:
                print("path not exist...")
        else:
            os.remove(os.path.join('../data/train/', img_id + '_msk.png'))

def img_gen(data_path, label_path):
  #Iterate over all image paths
  for i, img_path in enumerate(data_path):
    img = imread(data_path[i])/255
    yield img, label_path[i]

def generate_batch(data_path, lable_path, batchsize = 8):
  while True:
    ig = img_gen(data_path,lable_path)
    image_data = []
    label_data = []
    for img, label in ig:
      if label == 0:
        label_data.append([1,0])
      else:
        label_data.append([0,1])
      image_data.append(img)
      if len(image_data) == batchsize:
        yield np.stack(image_data, axis=0), np.stack(label_data, axis=0)
        image_data, label_data = [], []
    # If we have an nonempty batch left, yield it out and reset
    if len(image_data) != 0:
      yield np.stack(image_data, axis=0), np.stack(label_data, axis=0)
      image_data, label_data = [], []
aug_outlier('../data/aug/')
aug_path = glob('../data/aug/' + "*_sat.jpg")
print(len(aug_path))


# read image to list and generate batch
train_pos_path = glob("../data/train/"+"*_sat.jpg")
train_neg_path = glob("../data/aug/"+"*_sat.jpg")
# select length number of postive training example
len_neg = len(train_neg_path)
train_pos_path = np.random.choice(train_pos_path, len_neg)

train_pos_path = np.concatenate((train_pos_path,train_neg_path))
label_pos_batch = [0]*len_neg
label_pos_batch.extend([1]*len_neg)

label_batch = np.array(label_pos_batch)
s = np.arange(train_pos_path.shape[0])
np.random.shuffle(s)
train_path = train_pos_path[s]
label_batch = label_batch[s]

print("size of all data {}".format(train_path.shape))
print("size of all label {}".format(label_batch.shape))

len_val = int(len_neg*2*0.1)
train_path = train_path[len_val:]
train_label = label_batch[len_val:]


val_path = train_path[:len_val]
val_label = label_batch[:len_val]

print("size of training data {}".format(train_path.shape))
print("size of training label {}".format(train_label.shape))

print("size of val data {}".format(val_path.shape))
print("size of val label {}".format(val_label.shape))

traingen = generate_batch(train_path, train_label)
valgen = generate_batch(val_path, val_label)

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE    = (512, 512)
NUM_CLASSES   = 2
BATCH_SIZE    = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 50
WEIGHTS_FINAL = 'model-resnet50-final.h5'

net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))

x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
model = Model(inputs=net.input, outputs=output_layer)
# for layer in model.layers[:FREEZE_LAYERS]:
#     layer.trainable = False
# for layer in model.layers[FREEZE_LAYERS:]:
#     layer.trainable = True
model.compile(Adam(1e-4), loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

from keras.callbacks import *
def calc_steps(data_len, batchsize):
    return (data_len + batchsize - 1) // batchsize

# Calculate the steps per epoch
train_steps = calc_steps(len(train_path), 8)
val_steps = calc_steps(len(val_path), 8)


checkpointer = ModelCheckpoint('cp-{epoch:02d}-{val_loss:.4f}-od-resnet50.h5', verbose=1)
# Train the model
history = model.fit_generator(
    traingen,
    steps_per_epoch=train_steps,
    epochs=20, # Change this to a larger number to train for longer
    validation_data=valgen,
    validation_steps=val_steps,
    verbose=1,
    max_queue_size=5  # Change this number based on memory restrictions
)

model.save('outlier_detector_resnet50.h5')


model.save_weights('model_weights.h5')

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())


