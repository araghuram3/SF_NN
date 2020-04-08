# initialize functions

# import statements
import os
import numpy as np
# import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

# load images form directory
def loadImagesFromDir(path_str, img_size):
	images = os.listdir(path_str)
	samples = np.empty((len(images),img_size[1],img_size[0],1))
	i = 0
	for fn in images:
		# img = cv2.imread(path_str+'/'+fn) # this will work with tif but should make it 8 bit instead of 16 bit
		# gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		# gray_img = cv2.resize(gray_img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
		gray_img = np.asarray(Image.open(path_str+'/'+fn).resize(img_size))
		samples[i] = gray_img.reshape(img_size[1],img_size[0],1)
		i += 1
	return samples

# create x and y data from loaded images
# currently rescales by 255 (need to change this for all data types)
def createData(samples, samples_per_class, img_size, num_classes):
	x_data = np.empty((samples_per_class*num_classes,img_size[1],img_size[0],1))
	y_data = np.empty(samples_per_class*num_classes, dtype=int)
	for class_ind in range(num_classes):
		x_data[class_ind*samples_per_class:(class_ind+1)*samples_per_class] = samples[class_ind]/255.
		y_data[class_ind*samples_per_class:(class_ind+1)*samples_per_class] = class_ind

	return x_data, y_data

# create x and y data for augmentations
def augmentData(samples_per_class, classes, img_size, samples, datagenerator, save_path=''):

	# create training and testing data
	x_data = np.empty((samples_per_class*len(classes),img_size[1],img_size[0]))
	y_data = np.empty(samples_per_class*len(classes), dtype=int)
	for class_ind in range(len(classes)):
		num_input_images = len(samples[class_ind])
		
		# create, fit datagenerator and get iterator
		it = datagenerator.flow(samples[class_ind], batch_size=num_input_images)

		# if saving files, delete all previous files in the folder
		if save_path is not '':
			# check to see if save_path exists; if it doesn't, then make empty folder
			if not os.path.exists(save_path+classes[class_ind]+'/'):
				os.makedirs(save_path+classes[class_ind]+'/')
			files = os.listdir(save_path+classes[class_ind]+'/')
			for f in files:
				os.remove(save_path+classes[class_ind]+'/'+f)

		# create xdata and ydata
		for i in range(0,samples_per_class,num_input_images):
			gray_img = it.next().reclass(num_input_images,img_size[1],img_size[0])
			x_data[class_ind*samples_per_class+i:class_ind*samples_per_class+i+num_input_images] = gray_img
			y_data[class_ind*samples_per_class+i:class_ind*samples_per_class+i+num_input_images] = class_ind
			if save_path is not '':
				cv2.imwrite(save_path+classes[class_ind]+'/foggy_'+classes[class_ind]+str(i)+'.jpg',gray_img.reclass(img_size[0],img_size[1]))

	return x_data, y_data

# create confusion matrix
def createConfMat(model, x_test, y_test, classes, tf):
	predictions = model.predict_classes(x_test)
	con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predictions).eval(session=tf.compat.v1.Session())
	con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
	con_mat_df = pd.DataFrame(con_mat_norm,
	                     index = classes, 
	                     columns = classes)

	figure = plt.figure(figsize=(8, 8))
	sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
	sns.set(font_scale=2)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()

	return con_mat_df

# display training and testing error
def dispError(history):
	# list all data in history
	if 'acc' in history.history.keys():
		key1 = 'acc'
		key2 = 'val_acc'
	else:
		key1 = 'accuracy'
		key2 = 'val_accuracy'

	# summarize history for accuracy
	plt.plot(history.history[key1])
	plt.plot(history.history[key2])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.tight_layout()
	plt.show()

# show wrong predictions 
# assumes model has already been fit
def visualizeWrongPredictions(model, x_test, y_test, classes, samples_per_class, disp_size=5):
	num_classes = len(classes)
	# evals = model.evaluate(x_test)
	predictions = model.predict(x_test)
	predict_vec = np.empty(len(x_test),dtype=int)
	for ind in range(len(predictions)):
		predict_vec[ind] = np.argmax(predictions[ind])

	# iterate through each class individually
	wrongly_predicted_disp = []
	wrong_labels_disp = []
	pred_val_label_disp = []
	for class_ind in range(num_classes):
		samples = x_test[class_ind*samples_per_class:(class_ind+1)*samples_per_class]
		labels = y_test[class_ind*samples_per_class:(class_ind+1)*samples_per_class]
		predicts = predict_vec[class_ind*samples_per_class:(class_ind+1)*samples_per_class]
		pred_vals = predictions[class_ind*samples_per_class:(class_ind+1)*samples_per_class]
		wrongly_predicted = samples[predicts != labels]
		wrong_labels = predicts[predicts != labels]
		pred_val_label = pred_vals[predicts != labels]
		wrongly_predicted_disp.append(wrongly_predicted[:(disp_size*disp_size)])
		wrong_labels_disp.append(wrong_labels[:(disp_size*disp_size)])
		pred_val_label_disp.append(pred_val_label[:(disp_size*disp_size)])

	# display the wrong predictions
	for class_ind in range(num_classes):
		plt.figure(figsize=(10,10))
		images = wrongly_predicted_disp[class_ind]
		label_ind = wrong_labels_disp[class_ind]
		plt.title("Wrongly predicted " + classes[class_ind])

		for i in range(len(images)):
			plt.subplot(disp_size,disp_size,i+1)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(np.squeeze(images[i]), cmap=plt.cm.binary_r, vmin=0, vmax=1)
			plt.xlabel(classes[label_ind[i]]+'\n'+np.array2string(np.around(pred_val_label_disp[class_ind][i],3)))
		plt.tight_layout()
		plt.show()
