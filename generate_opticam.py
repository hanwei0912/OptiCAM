import numpy as np
import cv2
import os
import time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import sys
sys.path.append("pytorch_grad_cam")
from image import show_cam_on_image

from imagenet_loader import ImageNetLoader
from util import *

from absl import flags, app

FLAGS = flags.FLAGS

def main(_):
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    

    model = models.resnet50(pretrained=True)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocess_layer = Preprocessing_Layer(mean,std)
    model = nn.Sequential(preprocess_layer, model)
    target_layers = model[1].layer4[-1]
    
    model.to(device)
    model.eval()
    
    valdir = './images/'
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #normalize
        ])
    
    transf = transforms.ToPILImage()
    
    val_loader = torch.utils.data.DataLoader(
            ImageNetLoader(valdir, './revisited_imagenet_2012_val.csv', transform),
            batch_size=5, shuffle=False,
            num_workers=1, pin_memory=True
            )
    
    
    save_dir = './results/'+ FLAGS.name_path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
   
    OptCAM = Basic_OptCAM(model, device, target_layer=target_layers, max_iter=FLAGS.max_iter, learning_rate=FLAGS.learning_rate, name_f=FLAGS.name_f, name_loss=FLAGS.name_loss, name_norm=FLAGS.name_norm, name_mode='resnet')
    
    
    for i, (images, labels, file_name) in enumerate(val_loader):
        start = time.time()
        saliency_map, masked_images = OptCAM(images, labels)
        end = time.time()
        saliency_map = saliency_map.data.cpu().detach().numpy()
        

        avg_time = end - start
    
        for j in range(len(file_name)):
            sal_map = np.transpose(saliency_map[j],(1,2,0))
    
            img = np.array(transf(images[j]))/255.0
    
            visualization = show_cam_on_image(img, sal_map)
            cv2.imwrite(save_dir+file_name[j]+'_'+str(labels[j])+'_Smap.png',np.uint8(visualization))
    
           

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", default = 50, help = "max iteration for optimizer ")
    parser.add_argument("--learning_rate", default = 0.1, help = "learning rate for optimizer ")
    parser.add_argument("--target_layer", default = '42', help = "target layer")
    parser.add_argument("--name_f", default = 'logit_predict', help = "name of f function ")
    parser.add_argument("--name_loss", default = 'plain', help = "name of loss function ")
    parser.add_argument("--name_norm", default = 'max_min', help = "name of normalization function ")
    parser.add_argument("--name_path", default = 'OptiCAM/', help = "path for saving data ")
    args = parser.parse_args()
    flags.DEFINE_integer(
            'max_iter', args.max_iter, 'scale')
    flags.DEFINE_float(
            'learning_rate', args.learning_rate, 'scale')
    flags.DEFINE_string(
            'target_layer', args.target_layer, 'string')
    flags.DEFINE_string(
            'name_f', args.name_f, 'string')
    flags.DEFINE_string(
            'name_loss', args.name_loss, 'string')
    flags.DEFINE_string(
            'name_norm', args.name_norm, 'string')
    flags.DEFINE_string(
            'name_path', args.name_path, 'string')
    app.run(main)
