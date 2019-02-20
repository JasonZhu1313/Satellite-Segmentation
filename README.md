# Satellite-Segmentation
# The functionality of each py file
Config.py - The hyperparameters of the network
customer_init.py - some initialization method
Model.py - abstract model
readfile.py - read a batch of images from file and send it to tensorflow to train (refer to Tensorflow Dataset)
SegnetModel - implementation of segnet model
util.py - util methods

# How to run the code
1. Make several dirs under the data folder:
-- data
    -- Logs
    -- train
    -- val
    -- submission
2. download data from kaggle website

3. put the training data under train folder and the test data under val dataset

4. You can train your model by writing code in the main function of SegnetModel

if __name__ == "__main__":
    segmodel = SegnetModel()
    segmodel.training()