"""

Este módulo analiza las imágenes del dataset y extrae datos de valor.
Autor: Pablo Regodón Cerezo.
Fecha: 9 de febrero de 2021.

"""

# Setup
import os # Para obtener información de los directorios
from PIL import Image # Para obtener información de las imágenes

#ds_path = "../complete_ds/train/train/" # Ruta absoluta al dataset completo
ds_path = "../partial_ds/train/train/" # Ruta absoluta al dataset parcial
#ds_path = "../augmented_ds/train/train/" # Ruta absoluta al dataset expandido

categories = 0 # Contará el número de categorías
image_number = 0 # Contará el número de imágenes
avg_image_size = [0, 0] # Se utilizará para calcular el tamaño medio de las imágenes

# Analizamos categoría a categoría
cat_image_number = 0
cat_avg_image_size = [0, 0]
sizes_dictionary = dict()

folders_list = os.listdir(ds_path) # Obtenemos la lista de categorías
for folder_name in folders_list: # Recorremos cada carpeta
	folder_path = os.path.join(ds_path, folder_name) # Ruta absoluta de la carpeta
	print("--------------------------------------------------------------------------------")
	print(f" {folder_name}:")
	
	categories += 1
	
	for fname in os.listdir(folder_path): # Recorremos cada imagen de la carpeta
		fpath = os.path.join(folder_path, fname) # Ruta absoluta de la imagen
		im = Image.open(fpath) # Descriptor de la imagen
		
		cat_image_number += 1 # Contamos la imagen
		cat_avg_image_size[0] += im.size[0] # Añadimos su tamaño
		cat_avg_image_size[1] += im.size[1]
		if sizes_dictionary.get(im.size) == None: # Añadimos su tamaño al diccionario
			sizes_dictionary[im.size] = 1
		else:
			sizes_dictionary[im.size] += 1
		# endif
	# endfor
	
	image_number += cat_image_number
	avg_image_size[0] += cat_avg_image_size[0]
	avg_image_size[1] += cat_avg_image_size[1]
	cat_avg_image_size[0] = int(cat_avg_image_size[0] / cat_image_number)
	cat_avg_image_size[1] = int(cat_avg_image_size[1] / cat_image_number)
	
	print(f"     · Número de imágenes: {cat_image_number}")
	cat_image_number = 0
	print(f"     · Tamaño medio: ({cat_avg_image_size[0]}, {cat_avg_image_size[1]})")
	cat_avg_image_size[0] = 0
	cat_avg_image_size[1] = 0
	#print("     · Lista detallada de tamaños:")
	#for key in sorted(sizes_dictionary.keys()):
	#	print(f"        {key}: {sizes_dictionary[key]}")
	sizes_dictionary = dict()
#endfor

# Calculamos el tamaño medio de las imágenes
avg_image_size[0] = int(avg_image_size[0] / image_number)
avg_image_size[1] = int(avg_image_size[1] / image_number)

# Imprimimos la información general
print("--------------------------------------------------------------------------------")
print("--------------------------------------------------------------------------------")
print(" General:")
print(f"     · Número de categorías: {categories}.")
print(f"     · Número total de imágenes: {image_number}.")
print(f"     · Tamaño medio de las imágenes: ({avg_image_size[0]}, {avg_image_size[1]})")
print("--------------------------------------------------------------------------------")

