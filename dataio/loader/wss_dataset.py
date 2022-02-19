import torch.utils.data as data
import numpy as np
import datetime

from os import listdir
from os.path import join
from os.path import basename
from .utils import load_nifti_img, check_exceptions, is_image_file, open_image_np,open_target_np, open_target_np_glas, open_target_np_peso, open_wss_target;                   
import random

class wss_dataset(data.Dataset):
    def find_in_y(self,x):
        if "test" in x:
            x = basename(x)
            match = [y for y in self.target_filenames if x in y if "test" in y]
        else:
            x = basename(x)
            match = [y for y in self.target_filenames if x in y]
        return match[0]
    def __init__(self, root_dir, split, transform=None, preload_data=False,train_pct=0.8,balance=True):
        super(wss_dataset, self).__init__()
        #train dir 
        img_dir = join(root_dir,"1.training")
        img_dir = join(root_dir,"2.validation","img")
        mask_dir= join(root_dir,"2.validation","mask")
        test_mask_dir = join(root_dir,"2.validation","3.testing","mask")
        test_dir = join(root_dir,"2.validation","3.testing","img")

        self.image_filenames  = sorted([join(img_dir, x) for x in listdir(img_dir) if is_image_file(x)])
        self.image_filenames.extend(sorted([join(test_dir, x) for x in listdir(test_dir) if is_image_file(x)]))
        self.target_filenames  = sorted([join(mask_dir, x) for x in listdir(mask_dir) if is_image_file(x)])
        self.target_filenames.extend(sorted([join(test_mask_dir, x) for x in listdir(test_mask_dir) if is_image_file(x)]))
        # self.target_filenames = [list(map(int,[x.split('-')[-1][:-4][1],x.split('-')[-1][:-4][4],x.split('-')[-1][:-4][7]])) for x in self.image_filenames]
        sp= self.target_filenames.__len__()
        sp= int(train_pct *sp)
        random.shuffle(self.image_filenames)
        if split == 'train':
            self.image_filenames = self.image_filenames[:sp]
        elif split =='all':
            self.image_filenames = self.image_filenames
        else:
            self.image_filenames = self.image_filenames[sp:]
            # find the mask for the image
        self.target_filenames = [ self.find_in_y((x)) for x in self.image_filenames]
        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} patches'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [open_image_np(ii)[0] for ii in self.image_filenames]
            self.raw_labels = [open_target_np_glas(ii)[0] for ii in self.target_filenames]
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            input  = open_image_np(self.image_filenames[index])
            target  = open_wss_target(self.target_filenames[index])
        else:
            input = np.copy(self.raw_images[index])
            target = self.target_filenames[index]

        # handle exceptions
        if self.transform:
            input, target = self.transform(input, target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)