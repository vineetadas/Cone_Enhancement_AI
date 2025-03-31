import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import load_img,Augment_RGB_torch #, Augment_RGB_torch
import torch.nn.functional as F
import random
import pdb

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

class DataLoaderTrain_RF(Dataset):
    def __init__(self, rgb_dir, file_name, img_options=None, target_transform=None):
        super(DataLoaderTrain_RF, self).__init__()

        self.target_transform = target_transform
        self.rgb_dir = rgb_dir
        self.file_name = file_name

        
        self.gt_filenames = [os.path.join(self.rgb_dir, 'gt', x) for x in self.file_name]
        self.input_filenames = [os.path.join(self.rgb_dir, 'input', x) for x in self.file_name]
        
        self.img_options=img_options

        self.tar_size = len(self.gt_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        gt = torch.from_numpy(np.float32(load_img(self.gt_filenames[tar_index])))
        input1 = torch.from_numpy(np.float32(load_img(self.input_filenames[tar_index])))
        
        gt = gt.permute(2,0,1)
        input1 = input1.permute(2,0,1)

        gt_filename = os.path.split(self.gt_filenames[tar_index])[-1]
        input_filename = os.path.split(self.input_filenames[tar_index])[-1]      

        return gt, input1, gt_filename, input_filename
  

  
def get_training_data_RF(rgb_dir,fname):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain_RF(rgb_dir,fname)

