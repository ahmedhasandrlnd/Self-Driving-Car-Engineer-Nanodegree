# Traffic Sign Classifier

## Concepts
1. [Vehicle Simulator](https://www.youtube.com/watch?time_continue=44&v=K8ROKGMm-mc)
1. [Intro to Behavioral Cloning Project](https://www.youtube.com/watch?time_continue=18&v=YXs-IwG9ISg)
1. Project Resources
This project will be completed and submitted via workspaces. The required files and resources, described below, are already present in Behavioral Cloning Project workspace. See the [rubric](https://review.udacity.com/#!/rubrics/1968/view) and the [writeup_template.md](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for more details about the expectations.

* The [GitHub repository](https://github.com/udacity/CarND-Behavioral-Cloning-P3) has the following files:
	* drive.py: a Python script that you can use to drive the car autonomously, once your deep neural network model is trained
	* writeup_template.md: a writeup template
	* video.py: a script that can be used to make a video of the vehicle when it is driving autonomously
* [sample driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) (optional) - if you choose to use a workspace, this is already included in your files. You can find it in /opt/carnd_p3/data/ (/opt is in the directory above /home, where your workspace is contained). Note that if you choose to only use your own training data, you'll want to save it to a different directory to make sure they are not accidentally combined.
* a simulator, containing two tracks
We encourage you to drive the vehicle in training mode and collect your own training data, but we have also included [sample driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) for the first track, which you can optionally use to train your network. You may need to collect additional data in order to get the vehicle to stay on the road. To review project completion requirements, please see the [Project Rubric](https://review.udacity.com/#!/rubrics/432/view).
1. Running the Simulator
Here are the latest updates to the simulator:
	* Steering is controlled via position mouse instead of keyboard. This creates better angles for training. Note the angle is based on the mouse distance. To steer hold the left mouse button and move left or right. To reset the angle to 0 simply lift your finger off the left mouse button.
	* You can toggle record by pressing R, previously you had to click the record button (you can still do that).
	* When recording is finished, saves all the captured images to disk at the same time instead of trying to save them while the car is still driving periodically. You can see a save status and play back of the captured data.
	* You can takeover in autonomous mode. While W or S are held down you can control the car the same way you would in training mode. This can be helpful for debugging. As soon as W or S are let go autonomous takes over again.
	* Pressing the spacebar in training mode toggles on and off cruise control (effectively presses W for you).
	* Track 2 was replaced from a mountain theme to Jungle with free assets , Note the track is challenging
	* You can use brake input in drive.py by issuing negative throttle values
	If you are interested here is the source code for the [simulator repository](https://github.com/udacity/self-driving-car-sim)
	When you first run the simulator, you’ll see a configuration screen asking what size and graphical quality you would like. We suggest running at the smallest size and the fastest graphical quality. We also suggest closing most other applications (especially graphically intensive applications) on your computer, so that your machine can devote its resources to running the simulator.
	[Training Mode](https://www.youtube.com/watch?time_continue=56&v=rKw8md-zVno)
1. [Data Collection Tactics](https://www.youtube.com/watch?time_continue=2&v=kTJiHXJe_t4)
1. Data Collection Strategies
	* the car should stay in the center of the road as much as possible
	* driving counter-clockwise can help the model generalize
	* flipping the images is a quick way to augment the data
	* collecting data from the second track can also help generalize the model
	* we want to avoid overfitting or underfitting when training the model
	* knowing when to stop collecting more data
1. [Data Visualization](https://www.youtube.com/watch?time_continue=1&v=_Gto6fQQWFI)
1. [Training Your Network](https://www.youtube.com/watch?v=iYH4UvsPgOY)
1. [Running Your Network](https://www.youtube.com/watch?v=1UGOJGg-0dU)
1. [Data Preprocessing](https://www.youtube.com/watch?v=Oc7cLOS03PE)
1. [More Networks](https://www.youtube.com/watch?time_continue=6&v=rVusn6F5i7s)
1. [Data Augmentation](https://www.youtube.com/watch?v=2oaB2_DhmF8)
```
import numpy as np
image_flipped = np.fliplr(image)
measurement_flipped = -measurement
```
1.[Using Multiple Cameras](https://www.youtube.com/watch?time_continue=1&v=GumTdw9mjL0)
1. [Cropping Images in Keras](https://www.youtube.com/watch?v=SpPxyW-869U)
1. [Even More Powerful Network](https://www.youtube.com/watch?time_continue=1&v=6vVPHcgQkLg)
1. [More Data Collection](https://www.youtube.com/watch?time_continue=16&v=cCZNlX3KLnY)
1. Visualizing Loss
```
from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
```
1. Generators
```
import os
import csv

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=3)

"""
If the above code throw exceptions, try 
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
"""
```
1. Recording Video in Autonomous Mode 
1. Project Workspace Instructions
Common Issues
	* "No VNC" or "Service is not running" error when launching simulator - this is related to either A) not having the workspace GPU enabled (the simulator needs a GPU to run), or B) the web browser being used. Safari is likely to produce this error, while Chrome should run the simulator fine.
	* "No session for PID" error when launching simulator - when the desktop simulator is opened, sometimes a "PID error" window will appear. This error does not impact the simulator itself and can be safely ignored or click OK, it is harmless.
	* Missing simulator icon - the simulator icon may fail to appear after a short wait within the Linux Desktop. If this is the case, click on the Terminal icon in the Desktop, and the simulator icon will typically appear. Please note that you still will use the actual Terminal within the primary workspace, and not the one in the Desktop