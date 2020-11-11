import numpy as np
import os
from os import listdir
from os.path import isfile,join
import PIL
from PIL import Image
from numpy import asarray
import torch
from sklearn.preprocessing import OneHotEncoder
import os.path
from sklearn.preprocessing import LabelBinarizer


image_path="...data\\images" #this must be the folder of image files
mask_path="...data\\masks"

batch_images=[]
batch_masks=[]



def tensorize_image(image_path, output_shape):
    
    image_names=[f for f in listdir(image_path) if isfile(join(image_path, f))]
    batch_size=len(image_names)
    for i in range(len(image_names)):
        image=Image.open(image_path+"\\"+image_names[i])
        image=image.resize((output_shape[0],output_shape[1]))
        batch_images.append(np.asarray(image))
    batch_images_tensor=torch.Tensor((batch_size,output_shape[0],output_shape[1],3))
    return batch_images_tensor
     
def tensorize_masks(mask_path,output_shape):
    mask_names=[f1 for f1 in listdir(mask_path) if isfile(join(mask_path, f1))]
    batch_size=len(mask_names)
    for y in range(len(mask_names)):
        img=Image.open(mask_path+"\\"+mask_names[y])
        img=img.resize((output_shape[0],output_shape[1]))
        one_hot_encoded=OneHotEncoder().fit_transform(img).toarray()
        batch_masks.append(one_hot_encoded)
    batch_mask_tensor=torch.Tensor((batch_size,output_shape[0],output_shape[1],2))
    return batch_mask_tensor
