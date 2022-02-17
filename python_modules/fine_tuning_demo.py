"""

This is a module copied from freecodecamp.org. Link to the video the code
was copied from: https://www.youtube.com/watch?v=qFJeN9V1ZsI

This model performs fine-tuning to VGG16 model.

"""

# Setup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set the different datasets paths
train_path = '../demo_ds/train'
valid_path = '../demo_ds/valid'
test_path = '../demo_ds/test'

# Load the sets
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
	.flow_from_directory(
			     directory=train_path,
			     target_size=(224,224),
			     classes=['Car','Truck','Bus','Van','Motorcycle'],
			     batch_size=5
			     )
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
	.flow_from_directory(
			    directory=valid_path,
			    target_size=(224,224),
			    classes=['Car','Truck','Bus','Van','Motorcycle'],
			    batch_size=5
			    )
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
	.flow_from_directory(
			    directory=test_path,
			    target_size=(224,224),
			    classes=['Car','Truck','Bus','Van','Motorcycle'],
			    batch_size=5,
			    shuffle=False
			    )

# Assure our datasets have been loaded correctly
assert train_batches.n == 2500
assert valid_batches.n == 500
assert test_batches.n == 250

if "fine_tuning_demo.h5" in os.listdir():
	
	model = keras.models.load_model('fine_tuning_demo.h5')
	
	model.summary()
	

else:
	# Download the model
	vgg16_model = tf.keras.applications.vgg16.VGG16()
	vgg16_model.summary()
	print(type(vgg16_model))

	# Convert VGG16 model to Sequential model
	model = Sequential()
	for layer in vgg16_model.layers[:-1]: # Add each layer except the last one
		model.add(layer)

	model.summary()

	for layer in model.layers: # Turn all layers to non-trainable
		layer.trainable = False

	model.add(Dense(units=5, activation='softmax')) # Only last layer is trainable
	model.summary()

	model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)
	model.save("fine_tuning_demo.h5")
	#assert model.history.history.get('accuracy')[-1] > 0.95

# Prediction

def plot_confusion_matrix(cm, classes,
			   normalize=False,
			   title='Confusion matrix',
			   cmap=plt.cm.Blues):
	
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print("Confusion matrix, without normalization")
	
	print(cm)
	
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
			horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel("True label")
	plt.xlabel("Predicted label")
	plt.show()

predictions = model.predict(x=test_batches, verbose=0)
print(test_batches.classes)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

print(test_batches.class_indices)

cm_plot_labels = ['Car', 'Truck', 'Bus', 'Van', 'Motorcycle']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
	
