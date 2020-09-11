import numpy as np
import cv2
import json
import os
from os import listdir
from os.path import isfile,join
import PIL
from PIL import Image
from numpy import asarray
image_path="...\\folder" #this must be the folder of image files
width_height_list=[]
batch_images=[]
batch_images_tensors=[]

def tensorize_image(image_path, output_shape):
    
    image_names=[f for f in listdir(image_path) if isfile(join(image_path, f))]
    for i in range(len(file_names)):
     image=Image.open(mypath+"\\"+image_names[i])
     image.thumbnail((128,128))
     batch_images.append(image)
     #batch_images_tensors=torch.
     #(width,height)=image.size
     #batch_images.append(tuple(width,height))
     


    return 


img.thumbnail((128,128))
img2=np.asarray(img)
img3=PIL.Image.fromarray(img2)
print(img3.size)