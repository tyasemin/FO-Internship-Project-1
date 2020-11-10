import cv2, torch
import numpy as np
import glob

def tensorize_image(image_path_list, output_shape, cuda=False):
    image_list = []
    for image_path in image_path_list:
        image = cv2.cv2.imread(image_path)
        image = cv2.cv2.resize(image, output_shape)
        torchlike_image = torchlike_data(image)
        image_list.append(torchlike_image)
        
    image_array = np.array(image_list, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()
    if cuda:
        torch_image = torch_image.cuda()
    return torch_image
    

def tensorize_mask(mask_path_list, output_shape, n_class, cuda=False):
    mask_list = []
    for mask_path in mask_path_list:
        mask = cv2.cv2.imread(mask_path, 0)
        mask = cv2.cv2.resize(mask, output_shape)
        mask = one_hot_encoder(mask, n_class)
        #####################################################
        torchlike_mask = torchlike_data(mask)
        #####################################################

        mask_list.append(torchlike_mask)

    mask_array = np.array(mask_list, dtype=np.int)
    torch_mask = torch.from_numpy(mask_array).float()
    if cuda:
        torch_mask = torch_mask.cuda()
    return torch_mask

def torchlike_data(data):
    n_channels = data.shape[2]
    torchlike_data = np.empty((n_channels, data.shape[0], data.shape[1]))
    for ch in range(n_channels):
        torchlike_data[ch] = data[:,:,ch]
    return torchlike_data


def one_hot_encoder(data, n_class):
    encoded_data = np.zeros((*data.shape, n_class), dtype=np.int) 

    encoded_labels = [[0,1], [1,0]]
    for lbl in range(n_class):
        encoded_label = encoded_labels[lbl]                      
        numerical_class_inds = data[:,:] == lbl                                     
        encoded_data[numerical_class_inds] = encoded_label                                                     
    return encoded_data

def image_mask_check(image_path_list, mask_path_list):
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        image_name = image_path.split('\\')[-1].split('.')[0]
        mask_name  = mask_path.split('\\')[-1].split('.')[0]
        assert image_name == mask_name, "Image and mask name does not match {} - {}".format(image_name, mask_name)



if __name__ == '__main__':
    
    

    image_list = glob.glob('...image_folder\\*')
    image_list.sort()
    batch_image_list = image_list[:10]
    batch_image_tensor = tensorize_image(batch_image_list, (224, 224))


    print(batch_image_list)
    print(batch_image_tensor.dtype)
    print(type(batch_image_tensor))
    print(batch_image_tensor.shape)


    mask_list = glob.glob('...mask_folder\\*')
    mask_list.sort()
    batch_mask_list = mask_list[:10]
    batch_mask_tensor = tensorize_mask(batch_mask_list, (224, 224), 2)


    print(batch_mask_list)
    print(batch_mask_tensor.dtype)
    print(type(batch_mask_tensor))
    print(batch_mask_tensor.shape)  