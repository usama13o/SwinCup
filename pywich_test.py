import os
from pytorch_lightning.utilities.seed import seed_everything

from torch.utils.data.dataloader import T
import numpy as np
# TODO fix mkl problem 
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pywick 
from pywick.models.segmentation import deeplab_v3_plus
from torch.utils.data.sampler import SubsetRandomSampler
from utils.util import json_file_to_dict_args, json_file_to_pyobj,get_tags
from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from torch.utils.data import DataLoader, dataset
from pywick.modules import ModuleTrainer
import pywick.metrics as pwm
from utils.error_logger import ErrorLogger
import torch
import numpy
from tqdm import tqdm
import json
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from models import get_model


def train(args):
    # if args.seed:
    #     seed_everything(5)

    # Parse input arguments
    # json_filename ="configs\config_SwinT.json"
    # json_filename ="configs\config_SwinT_unet.json"
    # json_filename ="configs\config_SwinT_v2_decoderCup.json"
    # json_filename ="configs\config_TransUnet.json"
    # json_filename ="configs\config_TransUnet_AG.json"
    # json_filename ="configs\config_deeplab.json"
    # json_filename ="configs\config_unet_epi_multi_att_dsv.json"

    # Load options
    json_opts = json_file_to_pyobj(args.config,args)
    wanb_config= get_tags(json_file_to_dict_args(args.config,args),args)

    train_opts = json_opts.training

    # Architecture type
    arch_type =  train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup Data Loader

    seed_everything(5)
    from sklearn.model_selection import KFold
    train_dataset = ds_class(ds_path, split='all',      transform=ds_transform['train'], preload_data=train_opts.preloadData,balance=False)

    splits=KFold(n_splits=args.cross_val,shuffle=True,random_state=int(os.environ.get("PL_GLOBAL_SEED")))



    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
        # Add device to json_opts.model so it's passed through get_model -> ModelOpts -> model.initialize
        json_opts.model.device = device
        json_opts.model.isTrain = True # Ensure isTrain is set for model initialization
        
        # Determine save directory based on config and fold
        # This will be used by model.initialize (via ModelOpts) and Visualiser
        base_save_dir = getattr(json_opts.model, 'save_dir_prefix', "./checkpoints")
        experiment_name = getattr(json_opts.model, 'name', 'unnamed_experiment')
        # save_dir for model checkpoints, specific to this fold
        fold_save_dir = os.path.join(base_save_dir, experiment_name, f"fold_{fold}")
        mkdir(fold_save_dir)
        json_opts.model.save_dir = fold_save_dir # This will be picked up by ModelOpts

        # get_model will call model.initialize(model_opts) internally.
        # model_opts will now contain the device.
        model = get_model(json_opts.model, device) # Pass device to get_model
        
        # The following model.initialize call is REMOVED as get_model handles it.
        # model.initialize(opt=json_opts.model, device=device) 
        
        train_sampler = SubsetRandomSampler(train_idx)
        
        test_sampler = SubsetRandomSampler(val_idx)
        # Note: pin_memory=True is fine with CPU, but most beneficial with CUDA.
        # num_workers > 0 can cause issues with CUDA in scripts/debuggers on some systems.
        # Keep as is for now.
        train_loader = DataLoader(dataset=train_dataset, num_workers=8, batch_size= train_opts.batchSize, shuffle=False,pin_memory=True,persistent_workers=False,sampler=train_sampler)
        valid_loader= DataLoader(dataset=train_dataset, num_workers=8, batch_size= train_opts.batchSize, shuffle=False,pin_memory=True,persistent_workers=False,sampler=test_sampler)
        
        wanb_config['tags'] = wanb_config['tags']+'_fold_{}'.format(fold)

        # Visualiser needs to know the save_dir, ensure it's consistent with model's save_dir
        # model.save_dir is now set before initialize, visualizer can use it.
        visualizer_save_path = os.path.join(json_opts.model.save_dir, json_opts.visualisation.name) # Visualiser might create its own subfolder
        mkdir(visualizer_save_path)
        visualizer = Visualiser(json_opts.visualisation, save_dir=visualizer_save_path, resume= False if json_opts.model.continue_train else False, config=wanb_config)
        
        error_logger = ErrorLogger()
        start_epoch = False if json_opts.training.n_epochs < json_opts.model.which_epoch else json_opts.model.continue_train
        if train_opts.lr_policy == "one_cycle":
            model.set_scheduler(train_opts,len_train=len(train_loader),max_lr=json_opts.model.max_lr,division_factor=json_opts.model.division_factor,last_epoch=json_opts.model.which_epoch * len(train_loader) if start_epoch else -1)
        else:
            model.set_scheduler(train_opts)
        frozen=False
        # model.freeze()
        for epoch in range(model.which_epoch, train_opts.n_epochs):
            print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))
            #if epoch % 10 == 0:
                #if frozen:
                    #model.unfreeze()
                    #frozen = False
                #else: 
                    #print("freezing model")
                    #model.freeze()
                    #frozen=True

            # Training Iterations

            for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
                # Make a training update
                model.set_input(images, labels)
                # with torch.autograd.set_detect_anomaly(True):
                model.optimize_parameters()
                # model.optimize_parameters_accumulate_grd(epoch_iter)
                lr = model.update_learning_rate()
                
                
                lr = {"lr":lr}
                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats,**lr}, split='train')
                visualizer.plot_current_errors(epoch, error_logger.get_errors('train'), split_name='train')


            # Validation and Testing Iterations
            for loader, split in zip([valid_loader], ['validation']):
                for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):

                    # Make a forward pass with the model
                    model.set_input(images, labels)
                    model.validate()

                    # Error visualisation
                    errors = model.get_current_errors()
                    stats = model.get_segmentation_stats()
                    error_logger.update({**errors, **stats}, split=split)

                    # Visualise predictions
                    visuals = model.get_current_visuals()
                    visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

            # Update the plots
            for split in ['validation']:
                visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
                visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            error_logger.reset()
            visualizer.upload_limit=45

            # Save the model parameters
            if epoch % train_opts.save_epoch_freq == 0:
                model.save(epoch) # model.save internally calls save_network which is now device aware
                visualizer.save_model(epoch) # This should ideally use model.save_dir or a related path

            # Update the model learning rate
            # model.update_learning_rate(errors['Seg_Loss'])
        visualizer.finish()
        del train_loader,valid_loader, model # Explicitly delete model
        if device.type == 'cuda':
            torch.cuda.empty_cache() # Clear CUDA cache after each fold

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')
    parser.add_argument('-c', '--config',  help='training config file', required=False,type=str)

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
    parser.add_argument('-pretrain', '--path_pre_trained_model',   help='path to pre trained model', type=str)
    parser.add_argument('-tag',   help='tags passed to wandb' , type=str,default="")
    # parser.add_argument('--gpu_ids',   help='gpu id to use for trianing' , type=int,default=0) # Deprecated
    parser.add_argument('--device', help='Device for training (e.g., "cpu", "cuda", "cuda:0")', type=str, default=None)
    args = parser.parse_args()

    # Determine device
    if args.device:
        selected_device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            selected_device = torch.device("cuda")
            print("CUDA is available. Defaulting to GPU.")
        else:
            selected_device = torch.device("cpu")
            print("CUDA not available. Defaulting to CPU.")
    
    args.device = selected_device # Store the torch.device object in args for convenience

    train(args, selected_device) # Pass the determined torch.device object
