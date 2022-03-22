"""

This is a module that organizes the different sets in the dataset

Author: Pablo Regodón Cerezo
Date: February 2022

"""

# Setup
import os
import shutil
import random
import glob
from progress_bar import print_progress_bar

dataset_path = '../padded_ds/'

# Organize sets
os.makedirs(dataset_path + 'valid/Car')
os.makedirs(dataset_path + 'valid/Bus')
os.makedirs(dataset_path + 'valid/Truck')
os.makedirs(dataset_path + 'valid/Van')
os.makedirs(dataset_path + 'valid/Motorcycle')
os.makedirs(dataset_path + 'test/Car')
os.makedirs(dataset_path + 'test/Truck')
os.makedirs(dataset_path + 'test/Bus')
os.makedirs(dataset_path + 'test/Van')
os.makedirs(dataset_path + 'test/Motorcycle')

# Take a portion of original ds
for categorie in os.listdir(dataset_path + 'train/'):
	
	print(f"Building test ds for {categorie}")
	i = 0
	print_progress_bar(i, 50, prefix=' Building test ds:', suffix='complete', length=50)
	# 50 images for testing
	for image_file in random.sample(glob.glob(dataset_path + 'train/' + categorie + '/*'), 50):
		shutil.move(image_file, dataset_path + 'test/' + categorie + '/')
		i+=1
		print_progress_bar(i, 50, prefix=' Building test ds:', suffix='complete', length=50)
	
	# 20% of the rest for validation
	n_validation = int(len(os.listdir(dataset_path + 'train/' + categorie)) * 0.2)
	print(f"Building validation ds for {categorie}")
	i = 0
	print_progress_bar(i, n_validation, prefix=' Building validation ds:', suffix='complete', length=50)
	for image_file in random.sample(glob.glob(dataset_path + 'train/' + categorie + '/*'), n_validation):
		shutil.move(image_file, dataset_path + 'valid/' + categorie + '/')
		i+=1
		print_progress_bar(i, n_validation, prefix=' Building validation ds:', suffix='complete', length=50)

