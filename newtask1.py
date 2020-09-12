import numpy as np
import os
from os import listdir
from os.path import isfile,join
from numpy import asarray
import torch
from sklearn.preprocessing import OneHotEncoder
import os.path
from sklearn.preprocessing import LabelBinarizer
import cv2

image_path="...data\\images" #this must be the folder of image files
mask_path="...data\\masks"

batch_images=[]
batch_masks=[]



def tensorize_image(image_path, output_shape):
    
    image_names=[f for f in listdir(image_path) if isfile(join(image_path, f))]
    
    for i in range(len(image_names)):
        image=cv2.cv2.imread(image_path+"\\"+image_names[i],cv2.cv2.IMREAD_COLOR)
        image2=cv2.cv2.resize(image,(output_shape[0],output_shape[1]))
        batch_images.append(np.asarray(image2))
    batch_images_tensor=torch.Tensor(batch_images)
    return batch_images_tensor
     
def tensorize_masks(mask_path,output_shape):
    mask_names=[f1 for f1 in listdir(mask_path) if isfile(join(mask_path, f1))]
    for y in range(len(mask_names)):
        img=cv2.cv2.imread(mask_path+"\\"+mask_names[y],cv2.cv2.IMREAD_GRAYSCALE)
        img2=cv2.cv2.resize(img,(output_shape[0],output_shape[1]))
        one_hot_encoded=OneHotEncoder().fit_transform(img2).toarray()
        print(one_hot_encoded)
        print(one_hot_encoded.shape)
        batch_masks.append(one_hot_encoded)
    batch_mask_tensor=torch.Tensor(batch_masks)
        
    
    return batch_mask_tensor


