import numpy as np
import cv2
import json
import os
from os import listdir
from os.path import isfile,join
import PIL
from PIL import Image
from numpy import asarray
import torch
from sklearn.preprocessing import OneHotEncoder

image_path="...\\folder_image" #this must be the folder of image files
mask_path="...\\folder_mask"

batch_images=[]
batch_images_tensors=[]

batch_masks=[]
batch_masks_tensors=[]

def tensorize_image(image_path, output_shape):
    
    image_names=[f for f in listdir(image_path) if isfile(join(image_path, f))]
    for i in range(len(image_names)):
        image=Image.open(image_path+"\\"+image_names[i])
        image.resize((output_shape[0],output_shape[1]))
        batch_images.append(image)
        batch_images_tensor=torch.Tensor(batch_images)
    return batch_images_tensor
     
  

def tensorize_masks(mask_path,output_shape):
    mask_names=[f1 for f1 in listdir(mask_path) if isfile(join(mask_path, f1))]
    for y in range(len(mask_names)):
        img=cv2.imread((mask_path+"\\"+mask_names[y]),cv2.IMREAD_GRAYSCALE)
        img.resize((output_shape[0],output_shape[1]))
        enc=OneHotEncoder()
        enc.transform(np.asarray(img)).toarray()
        batch_masks.append(img)
        batch_masks_tensors=torch.Tensor(batch_masks)
    return batch_masks_tensors

 





