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
from PIL import Image

input_path = "../augmented_ds/train/train/"
output_path = "../resized_ds/train/train/"

folder_list = os.listdir(input_path)

total_length = 0

for fname in folder_list:
	total_length += len(os.listdir(input_path+fname))

i = 0
print_progress_bar(i, total_length, prefix='Resizing:', suffix='complete', length=50)
for fname in folder_list:
	
	if fname in os.listdir(output_path):
		pass
	else:
		os.mkdir(output_path+fname)
	
	for image_name in os.listdir(input_path+fname):
		im = Image.open(input_path+fname+"/"+image_name)
		im = im.resize((300, 300))
		im.save(output_path+fname+"/"+image_name)
		i += 1
		print_progress_bar(i, total_length, prefix='Resizing:', suffix='complete', length=50)

