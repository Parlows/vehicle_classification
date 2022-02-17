"""

This module implements a convolutional neural network with ResNet50 architecture,
pretrained with ImageNet. The net is intended to classify input images in 5 classes:
	- Car.
	- Truck.
	- Van.
	- Motorcycle.
	- Bus.

Author: Pablo Regodón Cerezo.
Date: February 2022.

"""

# Setup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Set the different datasets paths
train_path = '../partial_ds/train'
valid_path = '../partial_ds/valid'
test_path = '../partial_ds/test'

# Load the sets
train_batches = ImageDataGenerator(tf.keras.applications.resnet50.preprocess_input) \
	.flow_from_directory(
			      directory=train_path,
			      target_size=(224, 224),
			      classes=['Car','Truck','Bus','Van','Motorcycle'],
			      batch_size=5
			     )

valid_batches = ImageDataGenerator(tf.keras.applications.resnet50.preprocess_input) \
	.flow_from_directory(
			      directory=valid_path,
			      target_size=(224, 224),
			      classes=['Car','Truck','Bus','Van','Motorcycle'],
			      batch_size=5
			     )

test_batches = ImageDataGenerator(tf.keras.applications.resnet50.preprocess_input) \
	.flow_from_directory(
			      directory=test_path,
			      target_size=(224, 224),
			      classes=['Car','Truck','Bus','Van','Motorcycle'],
			      batch_size=5,
			      shuffle=False
			     )

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

""" For Debbuging
print(labels)
plotImages(imgs)
"""

# Train or load model
if 'resnet50.h5' in os.listdir('../saved_models/'):
	model = keras.models.load_model('resnet50.h5')
	model.summary()
else:
	# Download model
	resnet50_model = tf.keras.applications.resnet50.ResNet50()
	resnet50_model.summary()
	





















