# Description:

This folder contains the code ran on colab. In this project we not only use aws to train our model but also use colab as platform to make submission and train the model

### 647unet.ipynb
This notebook contains the training code for unet trained on data with 647outlier removed.

### CRF.ipynb
This notebook contains the code to do the Conditional Random Field (CRF) operation on predicted mask. CRF is a post processing method. But we didn't use this method in our submission. Because we found CRF is likely to remove the move predicted by the model and cause bad performance.

### TTA.ipynb
This notebook contains the code to do the Test Time Augmentation (TTA), we found that it will increase the score a little bit only when you train the model with augmentation.

### outlier_binary_detector.ipynb
This is a outlier binary detector, we use several pretrained model to do the outlier detecting.

### separate_predict_submission.ipynb
This is notebook that combines the prediction of model trained with non-outlier data and the predicted outlier; We tried two ways to do the combination.
1. Separte the test set to part1 and part2, part1 contains the image has pontential to be outlier. And part2 is normal image set. We use the model trained on whole training dataset to predict part1, and we use the model trained on data without outlier to predict part2. And then combine the two prediction as the final result.

2. Separte the test set to part1 and part2, part1 contains the image has pontential to be outlier. And part2 is normal image set. We directly set the run length encode to empty which means the mask is black, and we use the model trained on data without outlier to predict part2. And then combine the two prediction as the final result.

Finally we found that directly set the rle to be empty works better. Because it prevent false positive.

### unet-lr-decay-whole.ipynb
This notebook contains the code to train the model with decayed learning rate. We tried three learning rate mode.
1. Cyclical Learning Rate (base : 1e-4; max: 1e-3; mode: triangle2; max_step: 3000)
2. Decay the learning rate if the validation error of two continuous epochs don't decrease.
3. Decay the learning rate every fix epochs, say 10 epochs.

We see that 1 can lead faster training and convergence. while 3 works best. 2 is very likely to decay for SGD. So it the trainign will be slow in later stage.
