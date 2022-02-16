"""

This is a module copied from freecodecamp.org. Link to the video the code
was copied from: https://www.youtube.com/watch?v=qFJeN9V1ZsI

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
#%matplotlib inline

original_dataset_path = '../partial_ds/train/train/'

# Organize data into train, valid, test dirs
if os.path.isdir('../demo_ds/') is False:
	if os.path.isdir('../demo_ds/train/Car') is False:
		os.makedirs('../demo_ds/train/Car')
		os.makedirs('../demo_ds/train/Truck')
		os.makedirs('../demo_ds/train/Bus')
		os.makedirs('../demo_ds/train/Van')
		os.makedirs('../demo_ds/train/Motorcycle')
		os.makedirs('../demo_ds/valid/Car')
		os.makedirs('../demo_ds/valid/Bus')
		os.makedirs('../demo_ds/valid/Truck')
		os.makedirs('../demo_ds/valid/Van')
		os.makedirs('../demo_ds/valid/Motorcycle')
		os.makedirs('../demo_ds/test/Car')
		os.makedirs('../demo_ds/test/Truck')
		os.makedirs('../demo_ds/test/Bus')
		os.makedirs('../demo_ds/test/Van')
		os.makedirs('../demo_ds/test/Motorcycle')
		
		# Take a portion of the original dataset
		for c in random.sample(glob.glob(original_dataset_path+'Car/*'), 500):
			shutil.copy(c, '../demo_ds/train/Car')
		for c in random.sample(glob.glob(original_dataset_path+'Truck/*'), 500):
			shutil.copy(c, '../demo_ds/train/Truck')
		for c in random.sample(glob.glob(original_dataset_path+'Bus/*'), 500):
			shutil.copy(c, '../demo_ds/train/Bus')
		for c in random.sample(glob.glob(original_dataset_path+'Van/*'), 500):
			shutil.copy(c, '../demo_ds/train/Van')
		for c in random.sample(glob.glob(original_dataset_path+'Motorcycle/*'), 500):
			shutil.copy(c, '../demo_ds/train/Motorcycle')
		for c in random.sample(glob.glob(original_dataset_path+'Car/*'), 100):
			shutil.copy(c, '../demo_ds/valid/Car')
		for c in random.sample(glob.glob(original_dataset_path+'Truck/*'), 100):
			shutil.copy(c, '../demo_ds/valid/Truck')
		for c in random.sample(glob.glob(original_dataset_path+'Bus/*'), 100):
			shutil.copy(c, '../demo_ds/valid/Bus')
		for c in random.sample(glob.glob(original_dataset_path+'Van/*'), 100):
			shutil.copy(c, '../demo_ds/valid/Van')
		for c in random.sample(glob.glob(original_dataset_path+'Motorcycle/*'), 100):
			shutil.copy(c, '../demo_ds/valid/Motorcycle')
		for c in random.sample(glob.glob(original_dataset_path+'Car/*'), 50):
			shutil.copy(c, '../demo_ds/test/Car')
		for c in random.sample(glob.glob(original_dataset_path+'Truck/*'), 50):
			shutil.copy(c, '../demo_ds/test/Truck')
		for c in random.sample(glob.glob(original_dataset_path+'Bus/*'), 50):
			shutil.copy(c, '../demo_ds/test/Bus')
		for c in random.sample(glob.glob(original_dataset_path+'Van/*'), 50):
			shutil.copy(c, '../demo_ds/test/Van')
		for c in random.sample(glob.glob(original_dataset_path+'Motorcycle/*'), 50):
			shutil.copy(c, '../demo_ds/test/Motorcycle')

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


imgs, labels = next(train_batches) # 5 images + 5 labels

"""
This function plots an array of images passed as a parameter.

"""
def plotImages(images_arr):
	fig, axes = plt.subplots(1, 5, figsize=(20, 20))
	axes = axes.flatten()
	for img, ax in zip( images_arr, axes):
		ax.imshow(img)
		ax.axis('off')
	plt.tight_layout()
	plt.show(block=True)

print(labels)
plotImages(imgs)

# Train or load our model
if "cnn_demo.h5" in os.listdir():
	model = keras.models.load_model('cnn_demo.h5')
	
	model.summary()
else:
	model = Sequential([
		Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224,224,3)),
		MaxPool2D(pool_size=(2, 2), strides=2),
		Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
		MaxPool2D(pool_size=(2, 2), strides=2),
		Flatten(),
		Dense(units=5, activation='softmax')
	])

	model.summary()

	model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

	model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)
	
	model.save('cnn_demo.h5')

# Prediction


test_imgs, test_labels = next(test_batches)
print(test_labels)
plotImages(test_imgs)
print(test_batches.classes)

predictions = model.predict(x=test_batches, verbose=0)
print(np.round(predictions))

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

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

print(test_batches.class_indices)

cm_plot_labels = ['Car', 'Truck', 'Bus', 'Van', 'Motorcycle']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels)

