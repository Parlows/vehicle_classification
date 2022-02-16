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

# Generate training and validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	directory="../partial_ds/train/train", # Directory of the dataset
	batch_size=32, # Batch size
	image_size=(224, 224), # Input image size
	seed=1337, # Random seed to generate the split
	validation_split=0.2, # 20% of set for validation
	subset="training", # This will be the training subset
)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
	directory="../partial_ds/train/train", # Directory of the dataset
	batch_size=32, # Batch size
	image_size=(224, 224), # Input image size
	seed=1337, # Random seed to generate the split
	validation_split=0.2, # 20% of set for validation
	subset="validation", # This will be the validating subset
)

# Resize the images
resizer = tf.keras.Sequential()
resizer.add(tf.keras.layers.Resizing(224, 224))

# Data augmentation
data_augmentation = tf.keras.Sequential()
data_augmentation.add(tf.keras.layers.RandomFlip("horizontal"))
data_augmentation.add(tf.keras.layers.RandomRotation(0.1))

# Build the model
model = keras.Sequential(
	[
		resizer,
		data_augmentation,
		tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
		tf.keras.layers.Dense(classes=5, activation='softmax', name='fc5')
	]
)




