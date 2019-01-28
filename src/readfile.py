import os,sys
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from Config import Config
config = Config()
def get_filename_list(path):
    if os.path.exists(path):
        file_list = os.listdir(path)
    image_filenames = []
    label_filenames = []
    for file_name in file_list:
        if file_name[len(file_name)-3:] == 'jpg':
            file_id = file_name.split("_")[0]
            image_filenames.append(file_id+"_sat.jpg")
            label_filenames.append(file_id+"_msk.png")
    return image_filenames, label_filenames

def satellite_inputs(image_filenames, label_filenames, batch_size):
    images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)
    filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)
    image, label = satellite_reader(filename_queue)
    reshaped_image = tf.cast(image, tf.float32)
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(config.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d satellite images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(reshaped_image, label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def satellite_reader(filename_queue):
    image_filenames = filename_queue[0]
    label_filenames = filename_queue[1]

    image_value = tf.read_file(image_filenames)
    label_value = tf.read_file(label_filenames)

    image_bytes = tf.image.decode_jpeg(image_value)
    label_bytes = tf.image.decode_png(label_value)

    image = tf.reshape(image_bytes, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_DEPTH))
    label = tf.reshape(label_bytes, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))

    return image, label


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 3-D Tensor of [height, width, 1] type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 3D tensor of [batch_size, height, width ,1] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 1
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  # tf.image_summary('images', images)

  return images, label_batch



image_filenames, label_filenames = get_filename_list("../data/train/")
print image_filenames
print label_filenames