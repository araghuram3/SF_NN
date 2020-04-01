# initialize functions

# import statements
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# create foggy shapes
def loadImagesFromDir(path_str, img_size):
	images = os.listdir(path_str)
	samples = np.empty((len(images),img_size[1],img_size[0],1))
	i = 0
	for fn in images:
		img = cv2.imread(path_str+'/'+fn)
		gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		gray_img = cv2.resize(gray_img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
		samples[i] = gray_img.reshape(img_size[1],img_size[0],1)
		i += 1
	return samples

# create x and y data from loaded images
def createData(samples, samples_per_shape, img_size, num_shapes):
	x_data = np.empty((samples_per_shape*num_shapes,img_size[1],img_size[0],1))
	y_data = np.empty(samples_per_shape*num_shapes, dtype=int)
	for shape_ind in range(num_shapes):
		x_data[shape_ind*samples_per_shape:(shape_ind+1)*samples_per_shape] = samples[shape_ind]/255.
		y_data[shape_ind*samples_per_shape:(shape_ind+1)*samples_per_shape] = shape_ind

	return x_data, y_data

# create x and y data for augmentations
def augmentData(samples_per_shape, shapes, img_size, samples, datagenerator, save_path=''):

	# create training and testing data
	x_data = np.empty((samples_per_shape*len(shapes),img_size[1],img_size[0]))
	y_data = np.empty(samples_per_shape*len(shapes), dtype=int)
	for shape_ind in range(len(shapes)):
		num_input_images = len(samples[shape_ind])
		
		# create, fit datagenerator and get iterator
		it = datagenerator.flow(samples[shape_ind], batch_size=num_input_images)

		# if saving files, delete all previous files in the folder
		if save_path is not '':
			# check to see if save_path exists; if it doesn't, then make empty folder
			if not os.path.exists(save_path+shapes[shape_ind]+'/'):
				os.makedirs(save_path+shapes[shape_ind]+'/')
			files = os.listdir(save_path+shapes[shape_ind]+'/')
			for f in files:
				os.remove(save_path+shapes[shape_ind]+'/'+f)

		# create xdata and ydata
		for i in range(0,samples_per_shape,num_input_images):
			gray_img = it.next().reshape(num_input_images,img_size[1],img_size[0])
			x_data[shape_ind*samples_per_shape+i:shape_ind*samples_per_shape+i+num_input_images] = gray_img
			y_data[shape_ind*samples_per_shape+i:shape_ind*samples_per_shape+i+num_input_images] = shape_ind
			if save_path is not '':
				cv2.imwrite(save_path+shapes[shape_ind]+'/foggy_'+shapes[shape_ind]+str(i)+'.jpg',gray_img.reshape(img_size[0],img_size[1]))

	return x_data, y_data

# create confusion matrix
def createConfMat(model, x_test, y_test, shapes, tf):
	predictions = model.predict_classes(x_test)
	con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predictions).eval(session=tf.compat.v1.Session())
	con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
	con_mat_df = pd.DataFrame(con_mat_norm,
	                     index = shapes, 
	                     columns = shapes)

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
def visualizeWrongPredictions(model, x_test, y_test, shapes, samples_per_shape, disp_size=5):
	num_shapes = len(shapes)
	# evals = model.evaluate(x_test)
	predictions = model.predict(x_test)
	predict_vec = np.empty(len(x_test),dtype=int)
	for ind in range(len(predictions)):
		predict_vec[ind] = np.argmax(predictions[ind])

	# iterate through each shape individually
	wrongly_predicted_disp = []
	wrong_labels_disp = []
	pred_val_label_disp = []
	for shape_ind in range(num_shapes):
		samples = x_test[shape_ind*samples_per_shape:(shape_ind+1)*samples_per_shape]
		labels = y_test[shape_ind*samples_per_shape:(shape_ind+1)*samples_per_shape]
		predicts = predict_vec[shape_ind*samples_per_shape:(shape_ind+1)*samples_per_shape]
		pred_vals = predictions[shape_ind*samples_per_shape:(shape_ind+1)*samples_per_shape]
		wrongly_predicted = samples[predicts != labels]
		wrong_labels = predicts[predicts != labels]
		pred_val_label = pred_vals[predicts != labels]
		wrongly_predicted_disp.append(wrongly_predicted[:(disp_size*disp_size)])
		wrong_labels_disp.append(wrong_labels[:(disp_size*disp_size)])
		pred_val_label_disp.append(pred_val_label[:(disp_size*disp_size)])

	# display the wrong predictions
	for shape_ind in range(num_shapes):
		plt.figure(figsize=(10,10))
		images = wrongly_predicted_disp[shape_ind]
		label_ind = wrong_labels_disp[shape_ind]
		plt.title("Wrongly predicted " + shapes[shape_ind])

		for i in range(len(images)):
			plt.subplot(disp_size,disp_size,i+1)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(np.squeeze(images[i]), cmap=plt.cm.binary_r, vmin=0, vmax=1)
			plt.xlabel(shapes[label_ind[i]]+'\n'+np.array2string(np.around(pred_val_label_disp[shape_ind][i],3)))
		plt.tight_layout()
		plt.show()
