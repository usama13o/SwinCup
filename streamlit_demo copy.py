import numpy as np
from utils.util import json_file_to_dict_args, json_file_to_pyobj,get_tags
from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from sklearn.metrics import f1_score, precision_score, recall_score
import streamlit as st
import os 
from PIL import Image
from torchvision import transforms
import torch
import pywick.transforms.tensor_transforms as ts
from torch.utils.tensorboard._utils import make_grid

from pathlib import Path
#crag_dataset
import argparse

parser = argparse.ArgumentParser(description='CNN Seg Training Function')
parser.add_argument('-c', '--config',  help='training config file', required=False,type=str,default="/home/uz1/projects/codeServerEPI/configs/config_SwinT_v2_decoderCup.json")

parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
parser.add_argument('-a', '--arch_type',   help='wich architecture type')
parser.add_argument('-wandb', '--use_wandb',   help='use wandb to log the training',type=bool)
parser.add_argument('-cont', '--continue_train',   help='Should contine training?',type=bool)
parser.add_argument('-seed',help='Use the same seed>',type=bool)
parser.add_argument('-wep', '--which_epoch',   help='which epoch to continue training from?',type=int)
parser.add_argument('-maxlr', '--max_lr',   help='maximum learning rate for cyclic learning',  type=float)
parser.add_argument('-bs', '--batchSize',   help='batch size',type=int)
parser.add_argument('-cv', '--cross_val',   help='Cross validation folds',type=int,default=1)
parser.add_argument('-ep', '--n_epochs',   help='number of epochs', type=int)
parser.add_argument('-img', '--img_size',   help='number of epochs', type=int)
parser.add_argument('-out', '--output_nc',   help='Number of output classes', type=int)
parser.add_argument('-pretrain', '--path_pre_trained_model',   help='path to pre trained model', type=str,default="/home/uz1/checkpoints/swin transformer decodercup pretrained on pesol/140_net_S.pth")
parser.add_argument('-tag',   help='tags passed to wandb' , type=str,default="")
parser.add_argument('--gpu_ids',   help='gpu id to use for trianing' , type=int,default=0)

def load_image(train_dataset):
    #randomly pick an image from dataset 
    x,y = train_dataset[np.random.randint(0,len(train_dataset))]
    return x,y,Image.fromarray(x.numpy().squeeze().astype(np.uint8)),Image.fromarray(y.numpy().squeeze().astype(np.uint8)*255)
    

def load_model(args):
    
    from models import get_model
    json_opts = json_file_to_pyobj(args.config,args)

    model = get_model(json_opts.model)
    return model



def predict(model,image,y):
    model.set_input(image,y)
    model.validate()
    seg_img= np.transpose(model.pred_seg.cpu().numpy().astype(np.uint8),(1,2,3,0)).squeeze()
    st.image(seg_img*255,width=300,caption="Predicted")

    return None

def main(args):
    st.title('AI Model for Histology segemtnation')

    json_opts = json_file_to_pyobj(args.config,args)
    arch_type = json_opts.training.arch_type
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)
    ds_class = get_dataset(arch_type)
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    train_dataset = ds_class(ds_path, split='all',      transform=ds_transform['train'], balance=False)


    model = load_model(args)
    st.write("Model loaded . . .")
    x,y,image,target = load_image(train_dataset)
    # st.write(x.shape,y.shape)
    st.image([image,target],width=300)
    
    # dataset =  load_dataset(p)
    # st.write("Dataset loaded . . .")
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        im = predict(model, x,y)
    #     st.write("f1 score",f1)
    #     st.write("precision",precision)
    #     st.write("recall",recall)



if __name__ == '__main__':
    
    args = parser.parse_args()
    main(args)