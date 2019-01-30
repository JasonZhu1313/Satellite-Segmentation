import os,sys
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from Config import Config
config = Config()
def get_filename_list(path, prefix = None, is_train = True):
    if os.path.exists(path):
        file_list = os.listdir(path)
    image_filenames = []
    label_filenames = []
    for file_name in file_list:
        if file_name[len(file_name)-3:] == 'jpg':
            file_id = file_name.split("_")[0]
            image_filenames.append(os.path.join(prefix if prefix is not None else "",file_id+"_sat.jpg"))
            if is_train:
                label_filenames.append(os.path.join(prefix if prefix is not None else "",file_id+"_msk.png"))
    return image_filenames, label_filenames

def map_fn(path, label):
    # path/label represent values for a single example
    image = tf.image.decode_jpeg(tf.read_file(path))
    label = tf.image.decode_png(tf.read_file(label))
    label = tf.image.rgb_to_grayscale(label)
    # some mapping to constant size - be careful with distorting aspec ratios
    image = tf.image.resize_images(image,[512,512])
    # color normalization - just an example
    image = tf.to_float(image) * (2. / 255) - 1

    label = tf.map_fn(lambda x: x / 255, label)
    return image, label


def get_dataset(image_filenames, label_filenames, batch_size):
    image_paths = tf.convert_to_tensor(image_filenames, dtype=tf.string)
    labels = tf.convert_to_tensor(label_filenames, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    # num_parallel_calls > 1 induces intra-batch shuffling
    dataset = dataset.map(map_fn, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    return dataset



