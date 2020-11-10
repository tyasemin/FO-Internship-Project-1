import torch
import cv2
import glob
import torch.nn as nn
import torchvision
import PIL 
from skimage import io
input_shape = (224, 224)
n_classes=2
cuda=True
from preprocess import tensorize_image,tensorize_mask
model_path='...\\model.pt'
model=torch.load(model_path)
model.eval()

test=glob.glob('...\\test_data\\*')
test_mask=glob.glob('...\\mask_test_data\\*')


y=0
for test_img in test:
    test_img1 = tensorize_image([test_img], input_shape, cuda)
    output=model(test_img1)
    torchvision.utils.save_image(output,('...\\model_image\\'+str(y)+'.png'))
    y+=1
    