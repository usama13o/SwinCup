import torch.utils.data as data
import numpy as np
import datetime

from os import listdir
from os.path import isdir
from os.path import join
from os.path import basename
from .utils import load_nifti_img, check_exceptions, is_image_file, open_image_np,open_target_np, open_target_np_glas, open_target_np_peso;                   
import random

class asdc_dataset(data.Dataset):
    def find_in_y(self,x):
        x = basename(x)
        match = [y for y in self.target_filenames if x.split(".")[0] in y]
        assert len(match) == 1 , f"Found more than one target for image: {x}"

        return match[0]

    def img_gt(self,a,dir,parent):
        return [join(dir,parent+"\\"+x) for x in a if "frame" in x if "gt" not in x],[join(dir,parent+"\\"+x) for x in a if "gt" in x]
    def __init__(self, root_dir, split, transform=None, preload_data=False,train_pct=0.8,balance=True):
        super( asdc_dataset, self).__init__()

        dir=root_dir
        a = [self.img_gt(listdir(join(dir,x)),dir,x) for x in listdir(dir) if isdir(join(dir,x))]
        a=np.array(a)
        aa = a.reshape(100,-1)
        self.image_filenames = np.array([x[:2] for x in aa]).reshape(-1).tolist()
        self.target_filenames = np.array([x[2:] for x in aa]).reshape(-1).tolist()
        sp= self.target_filenames.__len__()
        sp= int(train_pct *sp)

        random.shuffle(self.image_filenames)
        if split == 'train':
            self.image_filenames = self.image_filenames[:sp]
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
            self.raw_images = [load_nifti_img(ii, dtype=np.int16)[0] for ii in self.image_filenames]
            self.raw_labels = [load_nifti_img(ii,dtype=np.int16)[0] for ii in self.target_filenames]
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            input ,_ =load_nifti_img(self.image_filenames[index],dtype=np.int16)
            target,_  =load_nifti_img(self.target_filenames[index],dtype=np.int16)
        else:
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])

        # handle exceptions
        check_exceptions(input, target)
        if self.transform:
            input, target = self.transform(input, target)
        # check_exceptions(input, target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)