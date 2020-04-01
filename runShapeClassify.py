# TODO
# determine if i should use validation splits or just set the validation dataset in the fitting procedure
# see how to monitor how much GPU ram using/own RAM i'm using
# do i need to ever change any learning rates?
# should generate more images of fog
# what about maxpool layers?

# import statements
import os
import time
import matplotlib.pyplot as plt

# tensorflow statements
import tensorflow as tf
layers = tf.keras.layers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
tf.compat.v1.disable_eager_execution()

# import functions
from sf_nn_util import loadImagesFromDir, createConfMat, dispError, createData, visualizeWrongPredictions

################ IMPORTING IMAGES AND CREATION OF TRAIN/TEST DATA ################
# define desired image size (order is tailored to cv2.resize)
img_size = (100,100)

# initialize classes and samples
path2data = '/nrl_data/experimental/test_343/'
print('Loading Data...')
classes = os.listdir(path2data+'train/')
# shape_samples = [loadImagesFromDir(path2data+shape, img_size) for shape in classes]
train_samples = [loadImagesFromDir(path2data+'train/'+c, img_size) for c in classes]
test_samples = [loadImagesFromDir(path2data+'test/'+c, img_size) for c in classes]

x_train, y_train = createData(train_samples, len(train_samples[0]), img_size, len(classes))
x_test, y_test = createData(test_samples, len(test_samples[0]), img_size, len(classes))

# display data
display = True
if display:
	for shape_ind in range(len(classes)):
		start_ind = shape_ind*len(test_samples[0])
		plt.figure(figsize=(10,10))
		rows = 5
		cols = 5

		for i in range(rows*cols):
			plt.subplot(cols,rows,i+1)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(x_test[start_ind+i].reshape(img_size[1],img_size[0]), cmap=plt.cm.binary_r, vmin=0, vmax=1)
			plt.xlabel(classes[y_test[start_ind+i]])
		plt.show()

################ CREATE NETWORK AND TRAINING ################
# build the model and train it
nepochs = 100
print('Creating model and training start...')
time_start = time.time()
model = tf.keras.Sequential()

# cnn
model.add(layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', input_shape=(img_size[1],img_size[0],1)))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=1))
model.add(layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=1))
model.add(layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=1))
model.add(layers.Conv2D(16, kernel_size=3, activation='relu', padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(len(classes), activation='softmax'))
model.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=100, epochs=nepochs, validation_data=(x_test, y_test))

print('Execution time: %0.2f seconds' % (time.time() - time_start))

################ VISUALIZATION / RESULTS ################
# display confusion matrix
createConfMat(model, x_test, y_test, classes, tf)

# display history of training/testing
dispError(history)

# display wrong detections
visualizeWrongPredictions(model, x_test, y_test, classes, len(test_samples[0]))