# -*- coding: utf-8 -*-


from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import scipy.io as sio
import h5py

class MyDataset(data.Dataset):
    def __init__(self, path='load_test.mat',method = 'h',lens = 15):
        if method=='h':
            data = h5py.File(path)
            image,label = data['image'],data['label']
            image = np.transpose(image)
            label = np.transpose(label)
            self.images = torch.from_numpy(image)
            self.images =  self.images[:,:,:,:,:]
            self.labels = torch.from_numpy(label).float()

        elif method=='nmnist_r':
            data = sio.loadmat(path)
            self.images = torch.from_numpy(data['image'])
            self.labels = torch.from_numpy(data['label']).float()
            self.images = self.images.permute(0,3,1,2,4)


        elif method=='nmnist_h':
            data = h5py.File(path)
            image, label = data['image'], data['label']
            image = np.transpose(image)
            label = np.transpose(label)
            self.images = torch.from_numpy(image)
            self.images = self.images[:, :, :, :, :]
            self.labels = torch.from_numpy(label).float()
            self.images = self.images.permute(0, 3, 1, 2, 4)

        else:
            data = sio.loadmat(path)
            self.images = torch.from_numpy(data['image'])
            self.labels = torch.from_numpy(data['label']).float()
        self.num_sample = int((len(self.images)//100)*100)
        print(self.images.size(),self.labels.size())

    def __getitem__(self, index):#返回的是tensor
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return self.num_sample