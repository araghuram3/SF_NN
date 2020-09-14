# TODO
# determine if i should use validation splits or just set the validation dataset in the fitting procedure
# see how to monitor how much GPU ram using/own RAM i'm using
# do i need to ever change any learning rates?
# should generate more images of fog
# what about maxpool layers?

# import statements
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pickle

# tensorflow statements
import tensorflow as tf
layers = tf.keras.layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
tf.compat.v1.disable_eager_execution()

# import functions
from sf_nn_util import loadImagesFromDir, createConfMat, dispError, createData, visualizeWrongPredictions, displayClassification, augmentData

################ IMPORTING IMAGES AND CREATION OF TRAIN/TEST DATA ################
# define params
img_size = (100,100)
n_splits = 5
make_translations = False

# if make translations is true, create datageneration
if make_translations:
	samples_per_class = 1000
	datagen = ImageDataGenerator(width_shift_range=0.2,
							 height_shift_range=0.2,
							 rotation_range=90)
test_folders = ["410","413","415","416","418","419","420","421","422","423","424","425","426","427","428"]
cam_folders = ["FLIR","VIS","LWIR"]
# test_folders = ["all4"]
# cam_folders = ["FLIR"]

for tfolder in test_folders:
	for cfolder in cam_folders:
		# initialize classes and samples
		path2data = '/nrl_data/experimental/20200723_SFOPjpg/'+tfolder+'/'+cfolder+'/'
		print('Loading Data...')
		classes = os.listdir(path2data+'train/')
		accs = []
		val_accs = []

		# shape_samples = [loadImagesFromDir(path2data+shape, img_size) for shape in classes]
		train_samples = [loadImagesFromDir(path2data+'train/'+c, img_size) for c in classes]
		test_samples = [loadImagesFromDir(path2data+'val/'+c, img_size) for c in classes]
		
		# translate image
		if make_translations:
			x_train, y_train = augmentData(samples_per_class, classes, img_size, train_samples, datagen)
			x_test, y_test = augmentData(samples_per_class, classes, img_size, test_samples, datagen)
		else:
			x_train, y_train = createData(train_samples, img_size, len(classes))
			x_test, y_test = createData(test_samples, img_size, len(classes))

		# combine data
		x = np.concatenate((x_train, x_test))
		y = np.concatenate((y_train, y_test))

		for train_idx, test_idx in KFold(n_splits).split(x):

			# get specific data
			x_train = x[train_idx]
			x_test = x[test_idx]
			y_train = y[train_idx]
			y_test = y[test_idx]
			
			# display data
			display = False
			if display:
				for shape_ind in range(len(classes)):
					# start_ind = shape_ind*len(test_samples[0])
					start_ind = 600
					plt.figure(figsize=(10,10))
					rows = 5
					cols = 5

					for i in range(rows*cols):
						plt.subplot(cols,rows,i+1)
						plt.xticks([])
						plt.yticks([])
						plt.grid(False)
						plt.imshow(x_test[start_ind+i].reshape(img_size[1],img_size[0]), cmap=plt.cm.binary_r, vmin=0, vmax=1)
						plt.xlabel(classes[int(y_test[start_ind+i])])


					import matplotlib
					matplotlib.use("TkAgg")
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
			if 'acc' in history.history.keys():
				key1 = 'acc'
				key2 = 'val_acc'
			else:
				key1 = 'accuracy'
				key2 = 'val_accuracy'
			accs.append(history.history[key1][-1])
			val_accs.append(history.history[key2][-1])

		# write output to file
		with open('outputs.txt','a') as o:
			o.write(tfolder + ' ' + cfolder + ': ' + str(sum(val_accs)/float(len(val_accs)))+'\n')

		with open(path2data+'kfold_history.pickle','wb') as f:
			pickle.dump([accs, val_accs], f)


		print('Execution time: %0.2f seconds' % (time.time() - time_start))

		# save model
		# model.save('sf_nn_model.h5')

		# try loading and testing again
		# new_model = load_model('sf_nn_model.h5')
		# new_model.evaluate(x_test, y_test)

		################ VISUALIZATION / RESULTS ################
		# # display confusion matrix
		# createConfMat(model, x_test, y_test, classes, tf)

		# # display history of training/testing
		# dispError(history)

		# # display wrong detections
		# visualizeWrongPredictions(model, x_test, y_test, classes, len(test_samples[0]))

		# # create classification video
		# print(path2data)
		# displayClassification(model, x_test, y_test, img_size, path2data)