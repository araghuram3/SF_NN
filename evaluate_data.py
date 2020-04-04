# script to process data
# will depend on how the data is imported
# write now will assume it is placed in a folder "test" in the same directory

# import statments
import matplotlib.pyplot as plt

# tensorflow statements
import tensorflow as tf
layers = tf.keras.layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
tf.compat.v1.disable_eager_execution()

# import functions
from sf_nn_util import loadImagesFromDir, createConfMat, dispError, createData, visualizeWrongPredictions

# load in model
model_nameNpath = './sf_nn_model.h5'
new_model = load_model(model_nameNpath)

# evalueate the test data
path2data = './test/'
images = loadImagesFromDir(path2data, img_size) # might need an processing step to make it usable in testing

# test the data
predictions = new_model.predict(images)
predict_vec = np.empty(len(x_test),dtype=int)
for ind in range(len(predictions)):
	predict_vec[ind] = np.argmax(predictions[ind])

plt.plot(predict_vec)
plt.xlabel('Frames')
plt.ylabel('Prediction')
plt.show()

# write now no way to test it without providing the ground truth