"""
[EARLY STAGE]

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

"""

train_ds = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
	   .flow_from_directory(directory="../partial_ds/train/train",
	   			 target_size=(224,224),
		                classes=['Car','Truck','Van','Motorcycle','Bus'],
		                batch_size=32
		               )

"""

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	directory="../augmented_ds/train/train", # Directory of the dataset
	batch_size=32, # Batch size
	#image_size=(224, 224), # Input image size
	seed=1337, # Random seed to generate the split
	validation_split=0.2, # 20% of set for validation
	subset="training", # This will be the training subset
)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
	directory="../augmented_ds/train/train", # Directory of the dataset
	batch_size=32, # Batch size
	#image_size=(224, 224), # Input image size
	seed=1337, # Random seed to generate the split
	validation_split=0.2, # 20% of set for validation
	subset="validation", # This will be the validating subset
)

# Resize the images
size = (224, 224)
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))

batch_size = 32

# To optimize loading speed
train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)

base_model = tf.keras.applications.EfficientNetB3(
	weights="imagenet",
	input_shape=(224, 224, 3),
	include_top=False,
)
base_model.trainable = True

inputs = tf.keras.layers.Input(shape=(224, 224))

x = base_model(inputs, training=True)

outputs = tf.keras.layers.Dense(units=5, activation='softmax', name='fc5')(x)

model = tf.keras.Model(inputs, outputs)

model.summary()

epochs = 50

callbacks = [
	keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
	optimizer=keras.optimizers.Adam(1e-3),
	loss="binary_crossentropy",
	metrics=["accuracy"],
)
model.fit(
	train_ds, epochs=epochs, callbacks=callbacks, validation_data=validation_ds,
)
