from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import pdb

from xml.dom import minidom
from xml.etree.ElementTree import XML, fromstring

import os
import scipy.io as si
osp = os.path
osj = osp.join

class ImageNetLoader(Dataset):
    def __init__(self, path_images, csv_file, transform=None):
        self.path = path_images
        self.transform = transform
        data_obj = open(csv_file, 'r')
        self.listed_data = data_obj.readlines()[0:100]
        data_obj.close()
        
    def __getitem__(self, idx):
        image_name, label = self.listed_data[idx].strip().split(',')
        image_ori = Image.open(osp.join(self.path,
                             image_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image_ori)

        return image,  int(label), image_name

    def __len__(self):
        return len(self.listed_data)  


