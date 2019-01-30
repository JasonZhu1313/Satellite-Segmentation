import tensorflow as tf
import readfile
image_filenames, label_filenames = readfile.get_filename_list("../data/train", prefix="../data/train")
print image_filenames
print label_filenames
import numpy as np
# Make a Dataset of file names including all the PNG images files in
# the relative image directory.
image_paths = tf.convert_to_tensor(image_filenames, dtype=tf.string)
labels = tf.convert_to_tensor(label_filenames, dtype=tf.string)
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

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

# num_parallel_calls > 1 induces intra-batch shuffling
dataset = dataset.map(map_fn, num_parallel_calls=8)

dataset = dataset.batch(5)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    result_tensor = sess.run(next_element)






