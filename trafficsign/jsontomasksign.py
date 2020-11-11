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

    
    json_objs=json_dict["objects"]  

    mask_name=(file_names[i][:-5])
    color=1
    mask_path=path_mask+"\\"+mask_name
    for obj in json_objs:
        class_name=obj['classTitle'] #trafficsign required
        if class_name == "Traffic Sign" :
            p=obj['points']          
            traffic_sign_points=p['exterior'] #trafficsign's points x,y type
            sign_x1=traffic_sign_points[0][0]
            sign_x2=traffic_sign_points[0][1]
            sign_y1=traffic_sign_points[1][0]
            sign_y2=traffic_sign_points[1][1]
            x=(sign_x1,sign_x2)
            y=(sign_y1,sign_y2)
            mask=cv2.cv2.rectangle(mask,x,y,1,thickness=8)
            
    cv2.cv2.imwrite(mask_path,mask.astype(np.uint8))