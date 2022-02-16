"""

This module analyses the images from the dataset and extracts valuable data.
Author: Pablo Regodón Cerezo.
Date: February 2021.

"""

# Setup
import os
from PIL import Image

#ds_path = "../complete_ds/train/train/" # Absolute path to the complete dataset
ds_path = "../partial_ds/train/train/" # Absolute path to the partial dataset
#ds_path = "../augmented_ds/train/train/" # Absolute path to the augmented dataset

categories = 0 # Number of categories
image_number = 0 # Number of images
avg_image_size = [0, 0] # Average sizes of the images

cat_image_number = 0 # Number of images per category
cat_avg_image_size = [0, 0] # Average image size per category
sizes_dictionary = dict() # Dictionary that stores the number of occurrences of each image size

folders_list = os.listdir(ds_path) # List of categories
for folder_name in folders_list:
	folder_path = os.path.join(ds_path, folder_name) # Absolute path to the directory
	print("--------------------------------------------------------------------------------")
	print(f" {folder_name}:")
	
	categories += 1
	
	for fname in os.listdir(folder_path): # Goes through each image of the folder
		fpath = os.path.join(folder_path, fname) # Absolute path to the image
		im = Image.open(fpath) # Image descriptor
		
		cat_image_number += 1 # Count the image
		cat_avg_image_size[0] += im.size[0] # Add image size
		cat_avg_image_size[1] += im.size[1]
		if sizes_dictionary.get(im.size) == None: # Count occurrence of size
			sizes_dictionary[im.size] = 1
		else:
			sizes_dictionary[im.size] += 1
		# endif
	# endfor
	
	# Update global values
	image_number += cat_image_number
	avg_image_size[0] += cat_avg_image_size[0]
	avg_image_size[1] += cat_avg_image_size[1]
	cat_avg_image_size[0] = int(cat_avg_image_size[0] / cat_image_number)
	cat_avg_image_size[1] = int(cat_avg_image_size[1] / cat_image_number)
	
	print(f"     · Number of images: {cat_image_number}")
	cat_image_number = 0
	print(f"     · Average size: ({cat_avg_image_size[0]}, {cat_avg_image_size[1]})")
	cat_avg_image_size[0] = 0
	cat_avg_image_size[1] = 0
	#print("     · Detailed list of sizes:")
	#for key in sorted(sizes_dictionary.keys()):
	#	print(f"        {key}: {sizes_dictionary[key]}")
	sizes_dictionary = dict()
#endfor

# Calculates the average size of images
avg_image_size[0] = int(avg_image_size[0] / image_number)
avg_image_size[1] = int(avg_image_size[1] / image_number)

# Prints general information
print("--------------------------------------------------------------------------------")
print("--------------------------------------------------------------------------------")
print(" General:")
print(f"     · Number of categories: {categories}.")
print(f"     · Number of images: {image_number}.")
print(f"     · Average size of images: ({avg_image_size[0]}, {avg_image_size[1]})")
print("--------------------------------------------------------------------------------")

