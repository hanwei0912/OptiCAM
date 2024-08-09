import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import sys
sys.path.append("pytorch_grad_cam")
from activations_and_gradients import ActivationsAndGradients

import pdb

class Preprocessing_Layer(torch.nn.Module):
    def __init__(self, mean, std):
        super(Preprocessing_Layer, self).__init__()
        self.mean = mean
        self.std = std

    def preprocess(self, img, mean, std):
        img = img.clone()
        #img /= 255.0

        img[:,0,:,:] = (img[:,0,:,:] - mean[0]) / std[0]
        img[:,1,:,:] = (img[:,1,:,:] - mean[1]) / std[1]
        img[:,2,:,:] = (img[:,2,:,:] - mean[2]) / std[2]

        #img = img.transpose(1, 3).transpose(2, 3)
        return(img)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        res = self.preprocess(x, self.mean, self.std)
        return res

def check_bounding_box(bbox, sm, img):
    b, c, w, h = img.shape

    empty = np.zeros((w, h))
    for box in bbox:
        empty[box.xslice,box.yslice]=1

    sm = torch.from_numpy(np.tile(sm,(1,c,1,1)))
    empty = torch.from_numpy(np.tile(empty,(1,c,1,1)))

    bbox_sm = sm * empty
    out_bbox_sm = sm * (1-empty)

    out_bbox_img = img * (1-empty)
    bbox_img = img * empty

    sm_img = sm * img
    out_bbox_sm_img = out_bbox_sm * img
    bbox_sm_img = bbox_sm * img

    return bbox_img, out_bbox_img, sm_img, out_bbox_sm_img,  bbox_sm_img


def f_logit_predict(model, device, x, predict_labels):
    outputs = model(x).to(device)
    one_hot_labels = torch.eye(len(outputs[0]))[predict_labels].to(device)
    j = torch.masked_select(outputs, one_hot_labels.bool())
    return j

def f_logit_predict_max(model, device, x, predict_labels):
    outputs = model(x).to(device)
    one_hot_labels = torch.eye(len(outputs[0]))[predict_labels].to(device)
    i, _ = torch.max((1-one_hot_labels).to(device)*outputs, dim=1)
    j = torch.masked_select(outputs, one_hot_labels.bool())
    return j - i

def f_cross_entropy(model, device, x, predict_labels):
    logits = model(x).to(device)
    loss = torch.nn.CrossEntropyLoss()
    output = loss(logits, predict_labels)
    return output

def normlization_max_min(saliency_map):
    max_value = saliency_map.view(saliency_map.size(0),-1).max(dim=-1)[0]
    min_value = saliency_map.view(saliency_map.size(0),-1).min(dim=-1)[0]
    delta = max_value - min_value
    min_value = min_value.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    delta = delta.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    norm_saliency_map = (saliency_map - min_value) / delta
    return norm_saliency_map

def power_normalization(x, alpha):
    sign_term = torch.sign(x)
    abs_term = torch.abs(x)
    power_term = torch.pow(abs_term, alpha)
    norm_term = sign_term * power_term
    return norm_term

def normlization_max_min_power(saliency_map, power):
    max_value = saliency_map.view(saliency_map.size(0),-1).max(dim=-1)[0]
    min_value = saliency_map.view(saliency_map.size(0),-1).min(dim=-1)[0]
    delta = max_value - min_value 
    min_value = min_value.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    delta = delta.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    norm_saliency_map = (saliency_map - min_value) / delta
    norm_saliency_map = torch.pow(norm_saliency_map, power)
    return norm_saliency_map

def normlization_sigmoid(saliency_map):
    norm_saliency_map = 1/2*(nn.Tanh()(saliency_map/2)+1)
    return norm_saliency_map

def normlization_max(saliency_map):
    max_value = saliency_map.view(saliency_map.size(0),-1).max(dim=-1)[0]
    max_value = max_value.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    norm_saliency_map = saliency_map / max_value
    return norm_saliency_map

def normlization_max_power(saliency_map, power):
    max_value = saliency_map.view(saliency_map.size(0),-1).max(dim=-1)[0]
    max_value = max_value.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    norm_saliency_map = saliency_map / max_value
    norm_saliency_map = torch.pow(norm_saliency_map, power)
    return norm_saliency_map

def normlization_tanh(saliency_map):
    norm_saliency_map = nn.Tanh()(saliency_map)
    return norm_saliency_map

class Basic_OptCAM:
    def __init__(self,
            model,
            device,
            max_iter=100,
            learning_rate=0.01,
            target_layer = '51',
            name_f = 'logit_predict',
            name_loss = 'norm',
            name_norm = 'max_min',
            name_mode = 'vgg'
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fea_ext = ActivationsAndGradients(model, target_layer, None)
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm
        self.name_mode = name_mode

    def get_f(self, x, y):
        return f_logit_predict(self.model, self.device, x, y)


    def get_loss(self, new_images, predict_labels, f_images):
        if self.name_loss =='norm':
            #L2 = torch.nn.MSELoss()
            loss = torch.sum(torch.abs((f_images - self.get_f(new_images, predict_labels))))
            if self.name_f == 'logit_vector':
                L2 = torch.nn.MSELoss()
                loss = L2(f_images, self.get_f(new_images, predict_labels))
        elif self.name_loss == 'plain':
            loss = torch.sum(f_images - self.get_f(new_images, predict_labels))
        else:
            raise Exception("Not Implemented")
        return loss

    def normalization(self, saliency_map):
        if self.name_norm =='max_min':
            return normlization_max_min(saliency_map)
        if self.name_norm == 'sigmoid':
            return normlization_sigmoid(saliency_map)
        if self.name_norm == 'max':
            return normlization_max(saliency_map)
        else:
            raise Exception("Not Implemented")


    def combine_activations(self, feature, w, images):
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images

    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        labels = labels.to(self.device)
        if self.name_mode == 'vgg' or self.name_mode == 'vgg_norm':
            feature = relu(self.fea_ext(images)[0])
        else:
            output = self.fea_ext(images)
            feature = relu(self.fea_ext.activations[0]).to(self.device)
        w = Variable(0.5*torch.ones((feature.shape[0],feature.shape[1],1,1),dtype=torch.float), requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        prev = 1e10
        predict_labels = labels 
        #predict_labels = self.model(images).argmax(axis=1).to(self.device)
        f_images = self.get_f(images, predict_labels)

        for step in range(self.max_iter):
            norm_saliency_map, new_images = self.combine_activations(feature, w, images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Optimization stopped due to convergence...')
                    return norm_saliency_map, new_images
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images = self.combine_activations(feature, w, images)
        return norm_saliency_map, new_images


    def __call__(self,
                images,
                labels
                ):
        return self.forward(images, labels)

    def __enter__(self):
        return self

