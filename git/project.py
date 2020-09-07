import numpy as np
import cv2
import json
import os
#from skimage.draw import polygon
#from skimage import io
from os import listdir
from os.path import isfile,join

#read files in the folder
mypath="...\\folder" #this must be the folder of json files
file_names=[f for f in listdir(mypath) if isfile(join(mypath, f))]

dictio_list=[]
i=0
path_mask="...\\folder_mask" #this must be the folder of masks to save 

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

    #only one image 
    for obj in json_objs:
        class_name=obj['classTitle'] #freespace required
        if class_name == "Freespace" :
            p=obj['points']          
            freespace_points=p['exterior'] #freespace's points x,y type
            image=cv2.fillPoly(mask,np.array([freespace_points]),1)
            #print(image)
            io.imsave(path_mask + "\\"+"test"+str(i)+ ".png", image) #shouldn't be the same name
            