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



def satellite_inputs(image_filenames, label_filenames, batch_size):
    images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)

    filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True, num_epochs=30)

    image, label = satellite_reader(filename_queue)
    reshaped_image = tf.cast(image, tf.float32)
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(config.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    print ('Filling queue with %d satellite images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)
    # Generate a batch of images and labels by building up a queue of examples.

    images, label_batch = tf.train.batch(
        [reshaped_image, label],
        batch_size=batch_size,
        capacity=3 * batch_size)

    return images, label_batch


def satellite_reader(filename_queue):
    image_filenames = filename_queue[0]
    label_filenames = filename_queue[1]

    image_value = tf.read_file(image_filenames)
    label_value = tf.read_file(label_filenames)

    image_bytes = tf.image.decode_jpeg(image_value)
    label_bytes = tf.image.decode_png(label_value)

    image = tf.reshape(image_bytes, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_DEPTH))

    label = tf.reshape(label_bytes, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 2))

    return image, label


