import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import csv
import matplotlib
import pickle
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from numpy import newaxis
from keras.models import Sequential, load_model
from keras.layers.core import Lambda, Flatten, Dense, Dropout


from sklearn.utils import shuffle

#use of more powerful networks
from keras.layers.convolutional import Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D



def train_generator(batch_size):
    """
    Generator function to load driving logs and input images.
    """
    while 1:
        with open('data/driving_log.csv') as driving_file:
            driving_log_reader = csv.DictReader(driving_file)
            count = 0
            Images = []
            Steerings = []
            #Data augmentation by flipping the image
            try:
                for row in driving_log_reader:
                    #correction factor to steer the vehicle
                    steering_offset = 0.2

                    centerImage = mpimg.imread('data/'+ row['center'].strip())
                    flippedCenterImage = np.fliplr(centerImage)
                    centerSteering = float(row['steering'])

                    leftImage = mpimg.imread('data/'+ row['left'].strip())
                    flippedLeftImage = np.fliplr(leftImage)
                    leftSteering = centerSteering + steering_offset

                    rightImage = mpimg.imread('data/'+ row['right'].strip())
                    flippedRightImage = np.fliplr(rightImage)
                    rightSteering = centerSteering - steering_offset

                    if count == 0:
                        Images = np.empty([0, 160, 320, 3], dtype=float)
                        Steerings = np.empty([0, ], dtype=float)
                    if count < batch_size:
                        Images = np.append(Images, np.array([centerImage, flippedCenterImage, leftImage, flippedLeftImage, rightImage, flippedRightImage]), axis=0)
                        Steerings = np.append(Steerings, np.array([centerSteering, -centerSteering, leftSteering, -leftSteering, rightSteering, -rightSteering]), axis=0)
                        count += 6
                    else:
                        #use of generators
                        yield shuffle(Images, Steerings)
                        count = 0
            except StopIteration:
                pass


batch_size = 100
use_transfer_learning = False

# define model
if use_transfer_learning:
    model = load_model('model.h5')
else:
    # define model
    model = Sequential()

    # crop extraneous parts of the image
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

    # normalize layer values
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # color space transformation
    model.add(Conv2D(1, (1, 1), activation="elu", strides=(1, 10), padding="valid"))
    #model.add(Conv2D(1, 1, 1, border_mode='valid', subsample=(1, 10), activation='elu'))
    
    # sharpen
    model.add(Conv2D(3, (3, 3), activation="elu", padding="valid"))
    #model.add(Conv2D(3, 3, 3, border_mode='valid', activation='elu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    # filter and sample
    model.add(Conv2D(6, (5, 5), activation="elu", strides=(2, 2), padding="valid"))
    #model.add(Conv2D(6, 5, 5, border_mode='valid', subsample=(2, 2), activation='elu'))

    #model.add(MaxPooling2D(pool_size=(2, 2)))
    # larger filter and sample
    model.add(Conv2D(16, (5, 5), activation="elu", strides=(2, 2), padding="valid"))
    #model.add(Conv2D(16, 5, 5, border_mode='valid', subsample=(2, 2), activation='elu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(25, activation='elu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

# train model
'''
steps_per_epoch: Total number of steps (batches of samples) to yield 
from generator before declaring one epoch finished and starting the next epoch. 
It should typically be equal to the number of unique samples of your dataset divided by the batch size.
'''
history_object = model.fit_generator(train_generator(batch_size), steps_per_epoch=8000, epochs=3, verbose=1)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['acc'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['Loss', 'Accuracy'], loc='upper right')
plt.show()
