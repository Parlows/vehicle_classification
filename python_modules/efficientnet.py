"""

This module implements a convolutional neural network with EfficientNet architecture,
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
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os

# Set the different datasets paths
train_path = '../augmented_ds/train'
valid_path = '../augmented_ds/valid'
test_path = '../augmented_ds/test'

# Load the sets
train_batches = ImageDataGenerator(tf.keras.applications.efficientnet.preprocess_input) \
	.flow_from_directory(
			      directory=train_path,
			      target_size=(300, 300),
			      classes=['Car','Truck','Bus','Van','Motorcycle'],
			      batch_size=5
			     )

valid_batches = ImageDataGenerator(tf.keras.applications.efficientnet.preprocess_input) \
	.flow_from_directory(
			      directory=valid_path,
			      target_size=(300, 300),
			      classes=['Car','Truck','Bus','Van','Motorcycle'],
			      batch_size=5
			     )

test_batches = ImageDataGenerator(tf.keras.applications.efficientnet.preprocess_input) \
	.flow_from_directory(
			      directory=test_path,
			      target_size=(300, 300),
			      classes=['Car','Truck','Bus','Van','Motorcycle'],
			      batch_size=5,
			      shuffle=False
			     )

imgs, labels = next(train_batches) # 5 images + 5 labels


"""
This function plots an array of images passed as a parameter.

"""
"""
def plotImages(images_arr):
	fig, axes = plt.subplots(1, 5, figsize=(20, 20))
	axes = axes.flatten()
	for img, ax in zip( images_arr, axes):
		ax.imshow(img)
		ax.axis('off')
	plt.tight_layout()
	plt.show(block=True)

"""
"""
print(labels)
plotImages(imgs)
"""

# Train or load model
if 'efficientnet.h5' in os.listdir('../saved_models/'):
	model = keras.models.load_model('efficientnet.h5')
	model.summary()
else:
	# Download model
	efficientnet_model = tf.keras.applications.EfficientNetB3(
							  include_top=False,
							  weights='imagenet',
							  input_shape=(300, 300, 3),
							  pooling='avg'
							 )
	efficientnet_model.summary()
	
	x = efficientnet_model.layers[-1].output
	x = Dropout(name='top_dropout', rate=0.3)(x)
	output = Dense(units=5, activation='softmax')(x)
	
	model = Model(inputs=efficientnet_model.input, outputs=output)

	for layer in model.layers[:-1]:
		layer.trainable = False

	model.summary()
	
	learning_rate = 0.0001
	optimizer = Adam(learning_rate=learning_rate)
	loss = 'categorical_crossentropy'
	epochs = 10
	
	model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
	
	checkpoint_path = "../saved_models/"
	callbacks = [
		ModelCheckpoint(checkpoint_path),
	]
	
	model.fit(x=train_batches, validation_data=valid_batches, epochs=epochs, callbacks=[callbacks], verbose=2)
	model.save('../saved_models/efficientnet.h5')

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

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

cm_plot_labels = ['Car', 'Truck', 'Bus', 'Van', 'Motorcycle']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')




