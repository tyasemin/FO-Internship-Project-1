import os, cv2, tqdm
import numpy as np
from skimage import io
from PIL import Image,ImageEnhance

image_p = '...\\for_brightness'


image_list = os.listdir(image_p)
for image_name in tqdm.tqdm(image_list):
    
    image_path     = '...\\for_brightness\\'+image_name
    image_out_path = '...\\increase_brigthness\\f'+image_name

    image=Image.open(image_path)
    enchancer=ImageEnhance.Brightness(image)
    factor=1.5
    image_output=enchancer.enhance(factor)
    image_output.save(image_out_path)

    