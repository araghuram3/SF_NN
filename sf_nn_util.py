# initialize functions

# import statements
import os
import time
import numpy as np
# import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from PIL import Image
import seaborn as sns
from skimage import io
from skimage.transform import resize

# load images form directory
def loadImagesFromDir(path_str, img_size):
	images = os.listdir(path_str)
	samples = np.empty((len(images),img_size[1],img_size[0],1))
	i = 0
	for fn in images:
		# img = cv2.imread(path_str+'/'+fn) # this will work with tif but should make it 8 bit instead of 16 bit
		# gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		# gray_img = cv2.resize(gray_img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
		if 'nothing' in path_str:
			# load in image
			gray_img = np.asarray(Image.open(path_str+'/'+fn).crop((0,0,101,120)).resize(img_size))
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(gray_img.reshape(img_size[1],img_size[0]), cmap=plt.cm.binary_r, vmin=0, vmax=1)
			import matplotlib
			matplotlib.use("TkAgg")
			plt.show()
		elif fn[-3:] == 'tif':
			gray_img = resize(io.imread(path_str+'/'+fn),(img_size[1],img_size[1]),anti_aliasing=True)
		else:
			gray_img = np.asarray(Image.open(path_str+'/'+fn).resize(img_size))

		samples[i] = gray_img.reshape(img_size[1],img_size[0],1)
		i += 1
	return samples

# create x and y data from loaded images
# currently rescales by 255 (need to change this for all data types)
def createData(samples, img_size, num_classes):
	total_samples = 0
	for s_ind in range(len(samples)):
		total_samples += len(samples[s_ind])

	# x_data = np.empty((total_samples,img_size[1],img_size[0],1))
	y_data = np.empty(total_samples, dtype=int)
	x_data = samples[0]/255.
	y_data = 0*np.ones(len(samples[0]))
	for class_ind in range(1,num_classes):
		print(x_data.shape)
		print(samples[class_ind].shape)
		x_data = np.concatenate((x_data, samples[class_ind]/255.), axis=0)
		y_data = np.concatenate((y_data, class_ind*np.ones(len(samples[class_ind]))), axis=0)
		
		# x_data[class_ind*samples_per_class:(class_ind+1)*samples_per_class] = samples[class_ind]/255.
		# y_data[class_ind*samples_per_class:(class_ind+1)*samples_per_class] = class_ind

	return x_data, y_data

# create x and y data for augmentations
def augmentData(samples_per_class, classes, img_size, samples, datagenerator, save_path=''):

	# find number of samples
	total_samples = 0
	for s_ind in range(len(samples)):
		num_input_images = len(samples[s_ind])
		if samples_per_class % num_input_images != 0:
			samples_per_class -= samples_per_class % num_input_images
			total_samples += samples_per_class - (samples_per_class % num_input_images)

	# create training and testing data
	x_data = np.empty((total_samples,img_size[1],img_size[0],1))
	y_data = np.empty(total_samples, dtype=int)
	for class_ind in range(len(classes)):
		print(classes[class_ind])
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
		print(samples_per_class)
		print(num_input_images)
		for i in range(0,samples_per_class,num_input_images):
			print('this worked')
			gray_img = it.next().reshape(num_input_images,img_size[1],img_size[0],1)
			x_data[class_ind*samples_per_class+i:class_ind*samples_per_class+i+num_input_images] = gray_img/255.
			y_data[class_ind*samples_per_class+i:class_ind*samples_per_class+i+num_input_images] = class_ind
			if save_path is not '':
				im = gray_img.reshape(img_size[0],img_size[1])
				im = im.save(save_path+classes[class_ind]+'/foggy_'+classes[class_ind]+str(i)+'.jpg')
				# cv2.imwrite(save_path+classes[class_ind]+'/foggy_'+classes[class_ind]+str(i)+'.jpg',gray_img.reclass(img_size[0],img_size[1]))

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

# display all the test data and their classificaiton in a video format
def displayClassification(model, test_xdata, test_ydata, img_size, save_path):
	
	# initialize writing params
	FFMpegWriter = manimation.writers['ffmpeg']
	metadata = dict(title='Movie Test', artist='Matplotlib',
	                comment='Movie support!')
	writer = FFMpegWriter(fps=15, metadata=metadata)

	# make predictions using model
	predictions = model.predict(test_xdata)
	predict_vec = np.empty(len(test_xdata),dtype=int)
	score = 0
	for ind in range(len(predictions)):
		predict_vec[ind] = np.argmax(predictions[ind])
		if predict_vec[ind] == test_ydata[ind]:
			score += 1
	acc = int(100*score/len(test_ydata))

	# in loop, display left as image and right as the plot of classificaiton
	figsize = (12,4)
	dpi = 100
	fig = plt.figure(figsize=figsize,dpi=dpi)
	ax1, ax2 = fig.subplots(1,2)
	ax1.set_xticks([])
	ax1.set_yticks([])
	img = ax1.imshow(test_xdata[ind].reshape(img_size[1],img_size[0]), cmap=plt.cm.binary_r, vmin=0, vmax=1)
	ax1.grid(False)
	ax1.set_title("Input Image")
	ax2.plot(range(1,1+len(predict_vec)), predict_vec, label="Predictions")
	ax2.set_xlabel("Test Slice")
	ax2.set_ylabel("Class Prediction")
	ax2.set_yticks([0,1,2])
	ax2.set_title("Prediction (" + str(acc) )
	with writer.saving(fig, save_path+'vis_classify.mp4', dpi):
		for ind in range(len(test_xdata)):

			# display the image
			img.set_data(test_xdata[ind].reshape(img_size[1],img_size[0]))
			ax1.set_xlabel(test_ydata[ind])

			# display the plot and plot an extra line showing where along the slices you are
			l = ax2.plot([ind, ind], [0,max(predict_vec)], 'r', label="Current")

			# store frame
			writer.grab_frame()
			l.pop(0).remove()

	# close the figure at the end
	plt.close(fig)

