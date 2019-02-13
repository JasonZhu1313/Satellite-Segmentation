import tensorflow as tf
import numpy as np
import re
import Config
from PIL import Image
import os
import pandas as pd
import numpy as np
import gc

def _variable(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, initializer, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable(
      name,
      shape,
      initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % "tower name", '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def writemask(prediction, filename):

    # prediction is (1,512,512,2)
    one_hot = tf.argmax(prediction,axis=2)
    condition = tf.equal(one_hot,0)
    result_image = tf.where(condition, tf.fill(one_hot.shape,0.0),tf.fill(one_hot.shape, 255.0))
    image = tf.stack([result_image,result_image,result_image],axis=-1)

    # final_image = tf.image.encode_png(image)
    val_path = os.path.join("../data/predict_mask/",filename)
    #write = tf.write_file(val_path, final_image)
    im = Image.fromarray(np.uint8(image.eval()))
    im.save(val_path)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(predictions, labels):
    num_class = predictions.shape[3]
    batch_size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    return hist


def print_hist_summery(hist):
    acc_total = np.diag(hist).sum() / hist.sum()
    print ('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print ('mean IU  = %f' % np.nanmean(iu))
    for ii in range(hist.shape[0]):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f " % (ii, acc))


def per_class_acc(predictions, label_tensor):
    labels = label_tensor
    size = predictions.shape[0]
    num_class = predictions.shape[3]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    print ('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print ('mean IU  = %f' % np.nanmean(iu))
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f " % (ii, acc))


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    """
    x = numpyarray of size (height, width) representing the mask of an image
    if x[i,j] == 0:
        image[i,j] is not a road pixel
    if x[i,j] != 0:
        image[i,j] is a road pixel
    """
    dots = np.where(x.T.flatten() != 0)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1):
            run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    #print run_lengths
    return run_lengths


# Create submission DataFrame
def create_submission(csv_name, predictions, filenames):
    """
    csv_name -> string for csv ("XXXXXXX.csv")
    predictions -> numpyarray of size (num_examples, height, width, prediction_channel)
                In this case (num_examples, 512, 512,2)
    image_ids -> numpyarray or list of size (num_examples,)

    predictions[i] should be the prediciton of road for image_id[i]
    """

    sub = pd.DataFrame(columns=['ImageId','EncodedPixels','Height','Width'])

    #for i in range(len(predictions)):
    encodings = []
    image_id = []
    for i in range(len(predictions)):
        # predictions[i] is of shape [512,512,2], process it to one channel prediction
        one_hot = tf.argmax(predictions[i], axis=2)
        result_image = tf.where(tf.equal(one_hot, 0), tf.fill(one_hot.shape, 0.0), tf.fill(one_hot.shape, 255.0)).eval()

        # batch size need to be 5, so mannually added a image to the test set, just omit this image
        if filenames[i] == '../data/val/1_sat.jpg':
            continue
        else:
            image_id.append(filenames[i].split('/')[-1].split('_')[0])
            encodings.append( rle_encoding(result_image))
    # for i in range(num_images):
    #     if (i + 1) % (num_images // 10) == 0:
    #         print(i, num_images)
    #     encodings.append(rle_encoding(predictions[i]))
    num_images = len(image_id)
    sub['Height'] = [512] * num_images
    sub['Width'] = [512] * num_images
    sub['EncodedPixels'] = encodings
    sub['ImageId'] = image_id
    sub.to_csv(csv_name, index=False)

def construct_label_batch(shape):
    label = np.random.randint(2, size=shape)
    return label

# when process the whole result the memory will overflow, then we split them into different submission files and merge them together
def combine_submission(csv_list, file_name):
    submission_list = []
    for file in csv_list:
        submission_df = pd.read_csv(file)
        submission_list.append(submission_df)
    result = pd.concat(submission_list)
    print len(result)
    result.to_csv(file_name, index=False)

if __name__ == "__main__":
    file_list = ["../data/submission/subid2_1.csv","../data/submission/subid2_2.csv"]
    combine_submission(file_list,"../data/submission/i.csv")
