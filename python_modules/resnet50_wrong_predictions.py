"""

This module implements a test for a trained EfficientNet model, which
shows the wrong predicted images.

Author: Pablo Regodón Cerezo.
Date: March 2022.

"""

# Setup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import shutil

test_path = "../test_ds/"

# Load model
if 'resnet50_50epochs.h5' in os.listdir('../saved_models/'):
	model = keras.models.load_model('../saved_models/resnet50_50epochs.h5')
	
	categories = ['Car', 'Truck', 'Bus', 'Van', 'Motorcycle']
	wrong_predictions = []
	total = 0
	for categorie in os.listdir(test_path):
		for image_name in os.listdir(test_path + categorie):
			total += 1
			file_path = test_path + categorie + "/" + image_name
			img = load_img(file_path, target_size=(224,224))
			img_array = img_to_array(img)
			img_batch = np.expand_dims(img_array, axis=0)
			img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_batch)
			
			predictions = model.predict(x=img_preprocessed)
			
			accuracy = 0
			i = 0
			index = 0
			for accuracies in predictions[0]:
				if predictions[0][i] > accuracy:
					accuracy = predictions[0][i]
					index = i
				i += 1
			
			if categories[index] != categorie:
				wrong_predictions.append(file_path)
				print(f"Predicted {file_path} as {categories[index]}.")
				shutil.copy(file_path, '../predicted_wrong_ds/' + categorie + '/predicted_' + categories[index].lower())
			
	length = len(wrong_predictions)
	print(f"Total of images wrongly predicted: {length}/{total}")
	
	
	
"""
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
"""





