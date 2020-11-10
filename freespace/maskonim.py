import os, cv2, tqdm
import numpy as np
from skimage import io
IMAGE_DIR = '...\\test_data'

i=0
image_list = os.listdir(IMAGE_DIR)
for image_name in tqdm.tqdm(image_list):
    mask_path      = '...\\model_image\\'+str(i)+'.png'
    image_path     = '...\\test_data\\'+image_name
    image_out_path = '...\\mask_on_image\\'+image_name
    
    mask  = cv2.cv2.imread(mask_path, 0).astype(np.uint8)
    mask=cv2.cv2.resize(mask,(1920,1208))
    image = cv2.cv2.imread(image_path).astype(np.uint8)

    #mask_ind   = mask == 1
    cpy_image  = image.copy()
    image[mask>=127,:] = (0, 165, 255)
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)
    i+=1
    cv2.imwrite(image_out_path, opac_image)