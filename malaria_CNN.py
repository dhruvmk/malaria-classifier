import os
from skimage.io import imread
from skimage.transform import resize


# Creating a simple pipeline that iterates through the folder and adds all the images to my lists
# The pipeline scales the images down to 30x30 pixels, and the pixel values are normalized to be between 0 and 1
# Every 5000 iterations, the number of iterations is printed to show progress

def pipeline(directory, container):
  i = 0
  for fn in os.listdir(directory):
    full_file = os.path.join(directory, fn)
    tensor = resize(imread(full_file, plugin='matplotlib')  , (30, 30, 3))
    tensor_normalize = tensor/255.
    container.append(tensor)
    if i%5000 == 0:
      print("Iteration:", i)
      
# Creating empty lists for each category
uninfected = []
infected = []

# Calling my pipeline on the seperate folders
dir1 = '/content/drive/My Drive/cell_images/Parasitized/'
dir2 = '/content/drive/My Drive/cell_images/Uninfected/'
pipeline(dir1, infected)
pipeline(dir2, uninfected)

# Combining the two different categories of images
x = np.concatenate((infected, uninfected))

# Generating labels
y = np.concatenate(([1 for x in range(13779)],[0 for x in range(13799)])).reshape(-1, 1)

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Shuffling data and splitting into train and test
x, y = shuffle(x, y)
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.2
)

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPool2D, Dropout, Dense, Flatten

# Building a convolutional neural network followed by a fully connected dense network
# Dropouts added to prevent overfitting

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (30, 30, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'sigmoid'))

# Model compiled
model.compile(optimizer = keras.optimizers.RMSprop(learning_rate=0.001, decay=1e-6),
              loss = "categorical_crossentropy",
              metrics = ["accuracy"]
)

# Training the model for 20 epochs
model.fit(train_x, train_y, batch_size=128, epochs=20, validation_data=(test_x, test_y), steps_per_epoch = len(train_x)//128)

# Saving the model so it can be used again without training
model.save('drive/My Drive/cell_images/malariaclassifier.h5')
