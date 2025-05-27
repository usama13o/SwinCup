from models.utils import load_byol_checkpoint, load_moco_checkpoint
from models.utils import load_checkpoint
import os
import numpy
import torch
from utils.util import mkdir
from .networks_other import get_n_parameters

class BaseModel():
    def __init__(self):
        self.input = None
        self.net = None
        self.isTrain = False
        self.device = None # Will be set in initialize
        self.schedulers = []
        self.optimizers = []
        self.save_dir = None
        # self.gpu_ids = [1] # Replaced by self.device
        self.which_epoch = int(0)
        self.path_pre_trained_model = None

    def name(self):
        return 'BaseModel'

    def initialize(self, opt, **kwargs): # device argument removed, opt now is ModelOpts
        self.device = opt.device # opt is ModelOpts, which has device set
        self.isTrain = opt.isTrain
        # self.gpu_ids = opt.gpu_ids # This was already commented out / replaced by device logic
        
        # Tensors will be moved to device in set_input or when created by the model itself.
        # self.ImgTensor and self.LblTensor are no longer defined here.
        
        self.save_dir = opt.save_dir; mkdir(self.save_dir)
        # Moving self.net to device will be handled by subclasses after net instantiation.
        # if self.net is not None:
        #     self.net.to(self.device) # This line is removed

    def set_input(self, images, labels=None): # Modified to move data to device
        self.input_images = images.to(self.device)
        if labels is not None:
            self.input_labels = labels.to(self.device)
        else:
            self.input_labels = None

    def set_scheduler(self, train_opt):
        pass

    def forward(self, split):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self): # Assuming visuals are derived from input_images/input_labels
        visuals = {'input_image': self.input_images}
        if hasattr(self, 'output_segmentation'): # Or however output is stored
            visuals['output_segmentation'] = self.output_segmentation 
        return visuals

    def get_current_errors(self):
        return {}

    def get_input_size(self):
        return self.input_images.size() if self.input_images is not None else None

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label): # Removed gpu_ids argument
        print('Saving the model {0} at the end of epoch {1}'.format(network_label, epoch_label))
        save_filename = '{0:03d}_net_{1}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        original_device = next(network.parameters()).device # Store original device
        torch.save(network.cpu().state_dict(), save_path)
        network.to(original_device) # Move back to original device

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        print('Loading the model {0} - epoch {1}'.format(network_label, epoch_label))
        save_filename = '{0:03d}_net_{1}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        # Load state dict to CPU first, then move model to target device
        network.load_state_dict(torch.load(save_path, map_location='cpu'))
        network.to(self.device)

    def load_network_from_path(self, network, network_filepath, strict):
        network_label = os.path.basename(network_filepath)
        epoch_label = network_label.split('_')[0]
        print('Loading the model {0} - epoch {1}'.format(network_label, epoch_label))
        
        # Common loading logic, ensure model is on CPU first, then moved to device
        # The specific load_X_checkpoint functions might need to be aware of map_location or model moved afterwards
        
        original_device = next(network.parameters()).device if len(list(network.parameters())) > 0 else self.device
        # Try to load to CPU first to avoid issues if checkpoint was saved on a different GPU
        try:
            if 'ep=' in network_label:
                # load_byol_checkpoint internally uses map_location='cpu'
                load_byol_checkpoint(network, network_filepath)
            elif "swin" in network_label:
                # load_checkpoint accepts map_location
                load_checkpoint(network, filename=network_filepath, map_location='cpu')
            elif 'epoch' in network_label:
                # load_moco_checkpoint internally uses map_location='cpu'
                load_moco_checkpoint(network, network_filepath)
            else:
                network.load_state_dict(torch.load(network_filepath, map_location='cpu'), strict=strict)
            network.to(self.device) # Move to target device
        except Exception as e:
            print(f"Error loading network: {e}. Attempting to load on original_device: {original_device}")
            # Fallback or re-raise. Ensure these calls also don't pass map_location if not accepted.
            if 'ep=' in network_label:
                load_byol_checkpoint(network,network_filepath) # No map_location
            elif "swin" in network_label:
                # load_checkpoint accepts map_location, but if error was due to map_location, try default
                load_checkpoint(network,filename=network_filepath) 
            elif 'epoch' in network_label:
                load_moco_checkpoint(network,network_filepath) # No map_location
            else:
                network.load_state_dict(torch.load(network_filepath, map_location=original_device), strict=strict)
            network.to(self.device)


    # update learning rate (called once every epoch)
    def update_learning_rate(self, metric=None, epoch=None):
        for scheduler in self.schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics=metric)
            else:
                scheduler.step()
            lr = self.optimizers[0].param_groups[0]['lr']
        print('current learning rate = %.7f' % lr)
        return lr

    # returns the number of trainable parameters
    def get_number_parameters(self):
        return get_n_parameters(self.net)

    # clean up the GPU memory
    def destructor(self):
        del self.net
        del self.input_images
        if self.input_labels is not None:
            del self.input_labels
