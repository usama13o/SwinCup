import torch.utils.data as data
import re
import numpy as np
import datetime

from os import listdir
from os.path import isdir
from os.path import join
from os.path import basename
from .utils import load_nifti_img, check_exceptions, is_image_file, open_image_np,open_target_np, open_target_np_glas, open_target_np_peso;                   
import random

class cc_dataset_test(data.Dataset):
    def find_in_y(self,x):
        match = [y for y in self.target_filenames if x.split("\\")[-1][:-4] in y if x.split("\\")[-2] in y]

        return match[0]

    def __init__(self, root_dir, split, transform=None, preload_data=False,train_pct=0.8,balance=True):
        super( cc_dataset_test, self).__init__()
        img_dir= root_dir
        # targets are a comob of two dirs 1- normal 1024 patches 2- Tum 1024

        self.image_filenames  =([join(root_dir, x) for x in listdir(root_dir) if is_image_file(x)])
        self.image_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

        # report the number of images in the dataset
        print('Number of {0} images: {1} patches'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [open_image_np(ii)[0] for ii in self.image_filenames]
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            input  = open_image_np(self.image_filenames[index])
        else:
            input = np.copy(self.raw_images[index])

        # handle exceptions
        # check_exceptions(input, target)
        if self.transform:
            input= self.transform(input)
        return input

    def __len__(self):
        return len(self.image_filenames)