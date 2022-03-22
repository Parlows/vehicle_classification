
import PIL
from PIL import Image
import os
from progress_bar import print_progress_bar

path="../padded_ds/train/"

folder_list = os.listdir(path)
total_length = 0
for fname in folder_list:
	total_length += len(os.listdir(path+fname))

i = 0
print_progress_bar(i, total_length, prefix='Padding:', suffix='complete', length=50)
for fname in folder_list:
	
	for image_name in os.listdir(path+fname):
		im = PIL.Image.open(path+fname+"/"+image_name)
		max_size = 0
		if im.height > im.width:
			max_size = im.height
		else:
			max_size = im.width
		im = PIL.ImageOps.pad(im, size=(max_size, max_size))
		im.save(path+fname+"/"+image_name)
		i += 1
		print(path+fname+"/"+image_name+"                                                      ")
		print_progress_bar(i, total_length, prefix='Padding:', suffix='complete', length=50)


