import os, cv2, tqdm
import numpy as np
from skimage import io
mask_p  = '...\\mask_folder'
image_p = '...\\image_folder'

image_list = os.listdir(image_p)
for image_name in tqdm.tqdm(image_list):
    image_path     = '...\\image_folder\\'+image_name
    image_out_path = '...\\ground_truht\\'+image_name
    mask_name=image_name[:-4]
    mask_path=mask_p+"\\"+mask_name+".png"
    
    
    
    mask  = cv2.cv2.imread(mask_path, 0).astype(np.uint8)
    image = cv2.cv2.imread(image_path).astype(np.uint8)

    mask_ind   = mask == 1
    cpy_image  = image.copy()
    image[mask==1,:] = (0, 165, 255)
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)
    
    cv2.imwrite(image_out_path, opac_image)
