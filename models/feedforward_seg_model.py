import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim

from collections import OrderedDict
import utils.util as util
from .base_model import BaseModel
from .networks import get_network
from .layers.loss import *
from .networks_other import get_scheduler, print_network, benchmark_fp_bp_time
from .utils import segmentation_stats, get_optimizer, get_criterion
from .networks.utils import HookBasedFeatureExtractor


class FeedForwardSegmentation(BaseModel):

    def name(self):
        return 'FeedForwardSegmentation'
    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
    def initialize(self, opts, **kwargs):
        # Call super().initialize(opts) which sets self.device from opts.device
        super().initialize(opts) # **kwargs removed, assuming not used by BaseModel.initialize
        self.isTrain = opts.isTrain # opts is ModelOpts, isTrain is an attribute

        # define network input and output pars
        # self.input and self.target will be set by set_input
        self.tensor_dim = opts.tensor_dim
        # self.gpu_ids=1 # This is deprecated, use self.device
        print(f"Model initialized on device: {self.device}")

        # load/define networks
        self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                               in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                               tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                               attention_dsample=opts.attention_dsample,img_size=opts.img_size,config=opts)
        
        if self.net is not None:
            self.net.to(self.device) # Move the network to the configured device
        else:
            raise RuntimeError("self.net was not initialized by get_network.")
            
        self.net.apply_argmax_softmax = self.apply_argmax_softmax
        # The old try-except block for moving to CUDA is removed. self.net.to(self.device) handles it.

        # load the model if a path is specified or it is in inference mode
        # BaseModel's load_network_from_path and load_network now handle moving to self.device
        if not self.isTrain or opts.continue_train:
            self.path_pre_trained_model = opts.path_pre_trained_model
            if self.path_pre_trained_model:
                self.load_network_from_path(self.net, self.path_pre_trained_model, strict=False)
                self.which_epoch = self.which_epoch
            else:
                self.which_epoch = opts.which_epoch
                self.load_network(self.net, 'S', self.which_epoch)

        # training objective
        if self.isTrain:
            self.criterion = get_criterion(opts)
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_S = get_optimizer(opts, self.net.parameters())
            self.optimizers.append(self.optimizer_S)

            # print the network details
            # print the network details
            if kwargs.get('verbose', True):
                print('Network is initialized')
                # print_network(self.net)

    def set_scheduler(self, train_opt, **kwargs):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opt,**kwargs))
            print('Scheduler is added for optimiser {0}'.format(optimizer))

    def set_input(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = current_input.size() # Use current_input consistently
            if (len(bs) == 4):
                # This permute seems specific, ensure it's (B, C, H, W) for typical PyTorch
                # If original is (B, H, W, C), then (0,3,1,2) makes it (B, C, H, W)
                current_input = current_input.permute(0,3,1,2) 
            if (len(bs) > 4): # Assuming 5D like (B, Z, C, H, W) or (B, C, Z, H, W)
                # This permute (0,1,4,2,3) and view suggests (B, C, Z, H, W) -> (B*Z, C, H, W)
                # If it's (B, C, Z, H, W), then permute(0,2,1,3,4) -> (B, Z, C, H, W)
                # then view(bs[0]*bs[2], bs[1], bs[3], bs[4])
                # Original: (B,C,H,W,Z) -> permute(0,1,4,2,3) -> (B,C,Z,H,W) -> view(B*Z,C,H,W) ???
                # Let's assume the original permute was for a specific input format e.g. (B, C, Depth, H, W)
                # And it wants to make it (B*Depth, C, H, W)
                # If input is (B, C, D, H, W), permute (0,2,1,3,4) -> (B, D, C, H, W)
                # then view(bs[0]*bs[1], bs[4], bs[2], bs[3]) is view(B*D, C, H, W)
                # This seems more standard if Z is depth.
                # The original permute(0,1,4,2,3) on (B,C,H,W,Z) -> (B,C,Z,H,W)
                # then view(bs[0]*bs[1], bs[4], bs[2], bs[3]) means (B*C, Z, H, W) - this is unlikely.
                # Let's trust the original permute for now, assuming it handles a specific data format.
                current_input = current_input.permute(0,1,4,2,3).contiguous().view(bs[0]*bs[1], bs[4], bs[2], bs[3])


            # Move to device, this replaces the BaseModel set_input essentially
            if idx == 0:
                self.input_images = current_input.to(self.device) # Use self.device
            elif idx == 1:
                self.input_labels = Variable(current_input.to(self.device)) # Use self.device
                # Ensure input and target dimensions match after potential permutes if necessary for loss
                # This assert might need adjustment based on the loss function and expected shapes
                # For instance, if target is (N, H, W) and input is (N, C, H, W) for CrossEntropyLoss
                # assert self.input_images.size()[2:] == self.input_labels.size()[1:] # Example check
                # The original assert was self.input.size(2) == self.target.size(2)

    def forward(self, split):
        # Use self.input_images as set by set_input (and moved to device by BaseModel's set_input)
        # Or, if set_input is overridden as here, use self.input_images
        if split == 'train':
            self.prediction = self.net(Variable(self.input_images))
        elif split == 'test':
            with torch.no_grad():
                self.prediction = self.net(Variable(self.input_images))
                # Apply a softmax and return a segmentation map
                self.logits = self.net.apply_argmax_softmax(self.prediction)
                self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1)
            
    def backward(self):
        self.loss_S = self.criterion(self.prediction, self.target)
        self.loss_S.backward()

    def optimize_parameters(self):
        self.net.train()
        self.forward(split='train')

        self.optimizer_S.zero_grad()
        self.backward()
        self.optimizer_S.step()

    # This function updates the network parameters every "accumulate_iters"
    def optimize_parameters_accumulate_grd(self, iteration):
        accumulate_iters = int(4) # how many iters to wait before backprop
        if iteration == 0: self.optimizer_S.zero_grad()
        self.net.train()
        self.forward(split='train')
        self.backward()

        if iteration % accumulate_iters == 0:
            self.optimizer_S.step()
            self.optimizer_S.zero_grad()

    def test(self):
        self.net.eval()
        self.forward(split='test')

    def validate(self):
        self.net.eval()
        self.forward(split='test')
        self.loss_S = self.criterion(self.prediction, self.target)

    def get_segmentation_stats(self):
        self.seg_scores, self.dice_score,self.precision,self.recall = segmentation_stats(self.prediction, self.target)
        seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']), ('Mean_IOU', self.seg_scores['mean_iou'])]
        for class_id in range(self.dice_score.size):
            seg_stats.append(('Class_dice_{}'.format(class_id), self.dice_score[class_id]))
        
        for class_id in range(self.precision.size):
            seg_stats.append(('Class_precision_{}'.format(class_id), self.precision[class_id]))
        # print(self.precision)
        # print(self.recall)
        for class_id in range(self.recall.size):
            seg_stats.append(('Class_recall_{}'.format(class_id), self.recall[class_id]))
        return OrderedDict(seg_stats)

    def get_current_errors(self):
        return OrderedDict([('Seg_Loss', self.loss_S.data.item())
                            ])

    def get_current_visuals(self):
        # Assumes self.input_images, self.pred_seg, self.input_labels are on the correct device
        # .cpu() is needed for numpy conversion
        inp_img_np = self.input_images.cpu().numpy().astype(np.uint8)
        # Adjust transpose based on expected input image format (e.g., CHW or HWC)
        # Assuming input is (B,C,H,W), visualizer might expect (B,H,W,C) or list of HWC
        if inp_img_np.ndim == 4 and inp_img_np.shape[1] in [1, 3]: # B,C,H,W
             inp_img_np = np.transpose(inp_img_np, (0,2,3,1))

        pred_seg_np = self.pred_seg.cpu().numpy().astype(np.uint8)
        if pred_seg_np.ndim == 4 and pred_seg_np.shape[1] == 1 : # B,1,H,W
            pred_seg_np = np.transpose(pred_seg_np, (0,2,3,1)).squeeze(3) # B,H,W
        elif pred_seg_np.ndim == 3: # B,H,W
            pass # Already in desired shape for typical single channel label map display

        true_seg_np = self.input_labels.cpu().numpy().astype(np.uint8)
        if true_seg_np.ndim == 4 and true_seg_np.shape[1] == 1: # B,1,H,W
            true_seg_np = np.transpose(true_seg_np, (0,2,3,1)).squeeze(3) # B,H,W
        elif true_seg_np.ndim == 3: # B,H,W
             pass # Already in desired shape

        return OrderedDict([('out_S', pred_seg_np), ('inp_S', inp_img_np),('true_S', true_seg_np)])

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))

    # returns the fp/bp times of the model
    def get_fp_bp_time (self, size=None):
        if size is None:
            # Adjust default size if necessary, ensure it matches tensor_dim (2D/3D)
            # Example for 2D: (1, self.net.in_channels, height, width)
            # Example for 3D: (1, self.net.in_channels, depth, height, width)
            # This needs to be consistent with the model's expected input
            if self.tensor_dim == '2D':
                size = (1, opts.input_nc if hasattr(self,'opts') else 1, 224, 224) # opts might not be here
            else: # 3D
                size = (1, opts.input_nc if hasattr(self,'opts') else 1, 96, 160, 160) # D, H, W or H, W, D depends on model

        inp_array = Variable(torch.zeros(*size)).to(self.device) # Use self.device
        out_array = Variable(torch.zeros(*size)).to(self.device) # Use self.device, though out_array might not be needed for benchmark
        # Ensure self.net is on self.device before benchmark
        # self.net.to(self.device) # Already done in initialize

        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array) # out_array might be unused by benchmark

        bsize = size[0]
        return fp/float(bsize), bp/float(bsize)

    def save(self, epoch_label):
        # self.gpu_ids is deprecated. save_network in BaseModel now handles device correctly.
        self.save_network(self.net, 'S', epoch_label) 
    def freeze(self,layername="Transformer"):
        print("Freezing ")
        for layer in self.net.children():
            for parameter in layer.parameters():
                if layer._get_name() in ['PatchEmbed','Dropout','ModuleList']:
                    parameter.requires_grad = False
    def unfreeze(self):
        print("unfreezing . . . ")
        for layer in self.net.children():
            for parameter in layer.parameters():
                parameter.requires_grad = True
