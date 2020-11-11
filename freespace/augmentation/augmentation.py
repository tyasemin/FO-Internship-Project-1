import os, cv2, tqdm
import numpy as np
from skimage import io
from PIL import Image
from PIL import ImageOps

image_p = '...\\image_folder'

image_list = os.listdir(image_p)
for image_name in tqdm.tqdm(image_list):
    
    image_path     = '...\\image_folder\\'+image_name
    image_out_path = '...\\augmentationed_image\\f'+image_name
    
    image=Image.open(image_path)
    image=ImageOps.mirror(image)
    image.save(image_out_path)
    

  