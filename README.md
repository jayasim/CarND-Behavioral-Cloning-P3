# Project 3: Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for P3, Behavioral Cloning.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting four files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)

Optionally, a video of your vehicle's performance can also be submitted with the project although this is optional. This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).


### My Approach


#### 1. Architecture
Below is the output of the Keras model.summary():

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 90, 32, 1)         4         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 88, 30, 3)         30        
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 42, 13, 6)         456       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 19, 5, 16)         2416      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 9, 2, 16)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 288)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 288)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               28900     
_________________________________________________________________
dense_2 (Dense)              (None, 25)                2525      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 26        
=================================================================
Total params: 34,357
Trainable params: 34,357
Non-trainable params: 0
_________________________________________________________________
```

#### 2. Overfitting Issues in the model

The model contains a dropout layer in order to reduce overfitting (see model.py, line 102).
Also, I've added Dropout with pool size 2, 2 in line 99

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (see model.py, line 106).

#### 4. Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road in my own recorded data. Additionally, I used the left and right camera images with a steering angle correction of ±0.2 to simulate recovery.

#### 5. Preprocessing and Augumentation

Used shuffle and added more images using flip for augumentation and preprocessing respectively 

#### 6. Problems faced in the Architecture and steps taken

* Initially I've kept my epoch size as 2 with steps_per_epoch as 8000 as I couldn't see much increase in accuracy so I decided to reduce the epoch to save training time. The problem with this approach is the network couldn't get a chance to learn from all the training data. So I've decided to increase epochs to 3 to cover the entire dataset.

* Data normalization is done in the second Keras Lambda layer

* To increase the overall amount of training data and make the model more robust and generalized, I decided to augment the data by flipping the image using np.fliplr using steering_offset = 0.2

* Applied a high dropout of 0.6 (60% of the data is lost) to reduce overfitting

* After the augmentation, I randomly shuffled the data set inside my train_generator - yield shuffle(Images, Steerings)

* The ideal number of epochs was 2-3 as the mean squared error only changed marginally after that. I used an adam optimizer so that manually changing the learning rate wasn't necessary.

* I've planned to work on improving the model's track 2 performance as well.

* From the start I wanted to achieve an accuracy of above 0.6,. I found out that even seemingly low accuracies yielded  acceptable autonomous driving. Even a ridiculously low accuracy below 1% yielded a safe center-of-lane driving for most of track 1.

#### 7. Dataset generation

I tried recording the manual dirving in both directions for my dataset but it came to me as a surprise that I am a poor driver of simulated cars :) So I've decided to go ahead with the training data provided as part of the exercise.