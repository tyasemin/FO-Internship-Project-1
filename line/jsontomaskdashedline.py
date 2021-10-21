import numpy as np
import cv2
import json
import os
#from skimage.draw import polygon
#from skimage import io
from os import listdir
from os.path import isfile,join

#read files in the folder
mypath="...\\json_folder" #this must be the folder of json files
file_names=[f for f in listdir(mypath) if isfile(join(mypath, f))]

dictio_list=[]
i=0
path_mask="...\\mask_folder" #this must be the folder of masks to save 

#open files in a loop
for i in range(len(file_names)):
    fileopen=open(mypath+"\\"+file_names[i]) 

    #return to the json object which is json dict 
    json_dict = json.load(fileopen)

    #add  objects to list 
    dictio_list.append(json_dict) 

    h_image=json_dict['size']['height']
    w_image=json_dict['size']['width']
    
    mask=np.zeros((h_image,w_image),dtype= np.uint8)

    #dictionary's object 
    json_objs=json_dict["objects"]  

    mask_name=(file_names[i][:-5])
    mask_path=path_mask+"\\"+mask_name
    for obj in json_objs:
        class_name=obj['classTitle'] #freespace required
        if class_name == "Dashed Line" :
            p=obj['points']          
            solidline_points=p['exterior'] #freespace's points x,y type
            mask=cv2.cv2.polylines(mask,np.array([solidline_points]),False,1,thickness=10)
            
    cv2.cv2.imwrite(mask_path,mask.astype(np.uint8))
