import numpy as np
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

import os

from utils.util import json_file_to_dict_args, json_file_to_pyobj,get_tags
from pytorch_lightning.utilities.seed import seed_everything
from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
from models.networks_other import adjust_learning_rate

from torch.utils.data.sampler import SubsetRandomSampler
from models import get_model


class StratifiedSampler(object):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.class_vector = class_vector
        self.batch_size = batch_size
        self.num_iter = len(class_vector) // 52
        self.n_class = 14
        self.sample_n = 2
        # create pool of each vectors
        indices = {}
        for i in range(self.n_class):
            indices[i] = np.where(self.class_vector == i)[0]

        self.indices = indices
        self.background_index = np.argmax([ len(indices[i]) for i in range(self.n_class)])


    def gen_sample_array(self):
        # sample 2 from each class
        sample_array = []
        for i in range(self.num_iter):
            arrs = []
            for i in range(self.n_class):
                n = self.sample_n
                if i == self.background_index:
                    n = self.sample_n * (self.n_class-1)
                arr = np.random.choice(self.indices[i], n)
                arrs.append(arr)

            sample_array.append(np.hstack(arrs))
        return np.hstack(sample_array)

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


# Not using anymore
def check_warm_start(epoch, model, train_opts):
    if hasattr(train_opts, "warm_start_epoch"):
        if epoch < train_opts.warm_start_epoch:
            print('... warm_start: lr={}'.format(train_opts.warm_start_lr))
            adjust_learning_rate(model.optimizers[0], train_opts.warm_start_lr)
        elif epoch == train_opts.warm_start_epoch:
            print('... warm_start ended: lr={}'.format(model.opts.lr_rate))
            adjust_learning_rate(model.optimizers[0], model.opts.lr_rate)


def train(arguments):

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
        #re-init model each fold 
        model = get_model(json_opts.model)

        
        train_sampler = SubsetRandomSampler(train_idx)
        
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset=train_dataset, num_workers=24, batch_size= train_opts.batchSize, shuffle=False,pin_memory=True,persistent_workers=False,sampler=train_sampler)
        valid_loader= DataLoader(dataset=train_dataset, num_workers=24, batch_size= train_opts.batchSize, shuffle=False,pin_memory=True,persistent_workers=False,sampler=test_sampler)
        wanb_config['tags'] = wanb_config['tags']+'_fold_{}'.format(fold)

        visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir,resume= False if json_opts.model.continue_train else False,config=wanb_config)
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
                # stats = model.get_classification_stats()
                error_logger.update({**errors,**lr}, split='train')
                visualizer.plot_current_errors(epoch, error_logger.get_errors('train'), split_name='train')


            # Validation and Testing Iterations
            for loader, split in zip([valid_loader], ['validation']):
                for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):

                    # Make a forward pass with the model
                    model.set_input(images, labels)
                    model.validate()

                    # Error visualisation
                    errors = model.get_current_errors()
                    stats = model.get_classification_stats()
                    error_logger.update({**errors, **stats}, split=split)

                    # Visualise predictions
                    visuals = model.get_current_visuals()
                    # visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

            # Update the plots
            for split in ['validation']:
                visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
                visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            error_logger.reset()
            visualizer.upload_limit=45

            # Save the model parameters
            if epoch % train_opts.save_epoch_freq == 0:
                model.save(epoch)
                visualizer.save_model(epoch)

            # Update the model learning rate
            # model.update_learning_rate(errors['Seg_Loss'])
        visualizer.finish()
        del train_loader,valid_loader


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
    parser.add_argument('--gpu_ids',   help='gpu id to use for trianing' , type=int,default=0)
    args = parser.parse_args()

    train(args)