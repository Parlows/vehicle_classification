"""

This module performs data augmentation for several classes in the vehicle dataset.
Author: Pablo Regodón Cerezo.
Date: Februrary 2022.

"""

# Setup
import os
import shutil
import random
from progress_bar import print_progress_bar
from PIL import Image, ImageFilter, ImageEnhance

# The pahts which will be used
input_path_preffix = "../partial_ds/train/"
output_path_preffix = "../augmented_ds/train/"

"""
This function performs the corresponding data augmentation given the category
name passed as a parameter.
"""
def augmentation(categorie):
	
	input_path = input_path_preffix + categorie + "/"
	output_path = output_path_preffix + categorie + "/"
	
	# Creates the directory
	if categorie in os.listdir(output_path_preffix):
		pass
	else:
		os.mkdir(output_path_preffix+categorie+"/")
	
	# Lists all the names of the images in the category
	image_list = os.listdir(input_path)
	
	# Shuffles the order
	random.shuffle(image_list)
	
	middle_index = len(image_list)//2
	
	# Prepares the sets that will be given each transformation
	if categorie == "Truck" or categorie == "Bus":
		gaussian_blur_set = image_list[:middle_index]
		brightness_set = image_list[middle_index:]
		random.shuffle(image_list)
		flip_set = image_list[:middle_index]
	elif categorie == "Motorcycle":
		gaussian_blur_set = image_list[:middle_index]
		brightness_set = image_list[middle_index:]
		flip_set = []
	elif categorie == "Car":
		gaussian_blur_set = []
		brightness_set = []
		flip_set = []
	elif categorie == "Van":
		gaussian_blur_set = image_list
		brightness_set = image_list
		flip_set = image_list
	
	
	print("--------------------------------------------------------------------------------")
	print(f" {categorie}:")
	
	print("     Augmenting data:")
	
	# Performs the augmentation
	i = 0
	print_progress_bar(i, len(gaussian_blur_set), prefix='     Gaussian blur:', suffix='complete', length=50)
	for fname in gaussian_blur_set:
		fpath = input_path + fname
		image = Image.open(fpath)
		image = image.filter(ImageFilter.GaussianBlur(radius=random.randint(2, 5)))
		image.save(output_path + "blur_" + fname)
		i += 1
		print_progress_bar(i, len(gaussian_blur_set), prefix='     Gaussian blur:', suffix='complete', length=50)
	
	i=0
	possible_values = (0.5, 0.6, 0.7, 1.3, 1.4, 1.5)
	print_progress_bar(i, len(brightness_set), prefix='     Brightness:', suffix='complete', length=50)
	for fname in brightness_set:
		fpath = input_path + fname
		image = Image.open(fpath)
		image = ImageEnhance.Brightness(image).enhance(possible_values[random.randint(0,5)])
		image.save(output_path + "bright_" + fname)
		i += 1
		print_progress_bar(i, len(brightness_set), prefix='     Brightness:', suffix='complete', length=50)
	
	random.shuffle(image_list)
	
	i=0
	print_progress_bar(i, len(flip_set), prefix='     Flip:', suffix='complete', length=50)
	for fname in flip_set:
		fpath = input_path + fname
		image = Image.open(fpath)
		image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
		image.save(output_path + "flipped_" + fname)
		i += 1
		print_progress_bar(i, len(flip_set), prefix='     Flip:', suffix='complete', length=50)
	
	print("     Copying original files:")
	
	# Copy the original images to the augmented data directories
	i = 0
	print_progress_bar(i, len(os.listdir(input_path)), prefix='     Copying:', suffix='complete', length=50)
	for fname in os.listdir(input_path):
		shutil.copy(input_path + fname, output_path + fname)
		i += 1
		print_progress_bar(i, len(os.listdir(input_path)), prefix='     Copying:', suffix='complete', length=50)
	print(" ")
	
	
augmentation("Car")
augmentation("Van")
augmentation("Truck")
augmentation("Bus")
augmentation("Motorcycle")

