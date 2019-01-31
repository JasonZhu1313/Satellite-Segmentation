class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # Constants describing the training process.
    MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

    INITIAL_LEARNING_RATE = 0.001  # Initial learning rate.
    EVAL_BATCH_SIZE = 4
    BATCH_SIZE = 4

    # for Satellite image
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    IMAGE_DEPTH = 3

    NUM_CLASSES = 2
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
    NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1
    TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST / BATCH_SIZE

    finetune = False
    log_dir = "../data/Logs"
    image_dir = "../data/train"
    val_dir =  "../data/val"

    maxsteps = 20000
