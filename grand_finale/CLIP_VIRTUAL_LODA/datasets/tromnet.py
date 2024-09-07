import os
import math
import random
from collections import defaultdict
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

import torchvision.datasets as datasets
from PIL import UnidentifiedImageError 
import cv2
from PIL import Image
imagenet_classes = ["injury","no_injury","injury_and_amputation"]

imagenet_templates = ["For the highlighted limb {} is present."]
class ImageFolderWithPaths(datasets.ImageFolder):

    def __init__(self, root, transform=None):
        super(ImageFolderWithPaths, self).__init__(root, transform)
    
    def __getitem__(self, index):
  
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        
        path = self.imgs[index][0]
    
        return (img, label ,path)
    
class DirectImageDataset(datasets.ImageFolder):
    def __init__(self, images_list, transform=None):
        self.images_list = images_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img = self.images_list[idx]
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image=Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        return image



class trom_net():

    dataset_dir = 'DATASET'

    def __init__(self, root, train_preprocess=None,test_preprocess=None):

        # self.dataset_dir = os.path.join(root, self.dataset_dir)
        # self.image_dir = os.path.join(self.dataset_dir, 'images')
        
        
        preprocess = transforms.Compose([ transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                                            ])
        
        # self.test=DirectImageDataset(test_imgs,transform=preprocess)
        self.test = ImageFolderWithPaths((root), transform=preprocess)
        
        self.template = imagenet_templates
        self.classnames = imagenet_classes
        
    def delete_images(self):
    # Iterate over dataset paths and delete images
        for _, _, path in self.test:
            if os.path.exists(path):
                os.remove(path)
            else:
                print(f"File {path} not found")
                
