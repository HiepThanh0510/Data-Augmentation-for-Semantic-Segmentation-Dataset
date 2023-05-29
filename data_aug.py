# Created by Vo Hiep Thanh 29.05.2023

import os
import cv2
from tqdm import tqdm
from glob import glob
from albumentations import VerticalFlip, HorizontalFlip


def load_data(path):
    '''
    The load_data function takes the path of the dataset and gives you a list of images and masks path.
    '''
    images = sorted(glob(os.path.join(path, "input_crop/" + "*.png")))     
    masks = sorted(glob(os.path.join(path, "mask_crop/" + "*.png")))
    return images, masks

path = "/home/thanh/workspace/intern/Data-Augmentation-for-Semantic-Segmentation-Dataset/data/"

images, masks = load_data(path)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
new_data_path = "/home/thanh/workspace/intern/Data-Augmentation-for-Semantic-Segmentation-Dataset/new_data"
new_input_crop = "/home/thanh/workspace/intern/Data-Augmentation-for-Semantic-Segmentation-Dataset/new_data/input_crop"
new_mask_crop = "/home/thanh/workspace/intern/Data-Augmentation-for-Semantic-Segmentation-Dataset/new_data/mask_crop"

create_dir(new_data_path)
create_dir(new_input_crop)
create_dir(new_mask_crop)    

H = 512 
W = 512 
def augment_data(images, masks, augment, save_path):
    for x, y in zip(images, masks):
        """
        Extracting image and mask
        """    
        image = x.split("/")[-1].split(".")
        image_name = image[0] # subset0_0071_1020
        image_extn = image[1] # png
        
        mask = y.split("/")[-1].split(".") # ['subset0_0071_1020', 'png']
        mask_name = mask[0] # subset0_0071_1020
        mask_extn = mask[1] # png
        
        """
        Read image and mask
        """
        x = cv2.imread(x, cv2.IMREAD_COLOR) 
        y = cv2.imread(y, cv2.IMREAD_COLOR) 

        """
        Augmentation
        """
        if augment == True: 
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']
            
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']
            
            save_images = [x, x1, x2]
            save_masks  = [y, y1, y2]
                
        else: 
            save_images = [x]
            save_masks = [y]
        
        """
        Saving the image and mask.
        """
        idx = 0 
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (H, W))
            m = cv2.resize(m, (H, W))
            
            tmp_image_name = f"{image_name}_{idx}.{image_extn}"
            tmp_mask_name = f"{mask_name}_{idx}.{mask_extn}"
            
            image_path = os.path.join(save_path, "input_crop", tmp_image_name)
            mask_path = os.path.join(save_path, "mask_crop", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            
            idx += 1     

save_path = "/home/thanh/workspace/intern/Data-Augmentation-for-Semantic-Segmentation-Dataset/new_data"            
augment_data(images=images, masks=masks, augment=True, save_path=save_path)
        































