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
        BaseModel.initialize(self, opts, **kwargs)
        self.isTrain = opts.isTrain

        # define network input and output pars
        self.input = None
        self.target = None
        self.tensor_dim = opts.tensor_dim
        self.gpu_ids=1
        print("USING CUDA !!cuda:{}".format(self.gpu_ids)) if self.use_cuda else print("NOT USING CUDA !!!!!")
        # load/define networks
        self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                               in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                               tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                               attention_dsample=opts.attention_dsample,img_size=opts.img_size,config=opts)
        self.net.apply_argmax_softmax = self.apply_argmax_softmax
        try:
            if self.use_cuda: self.net = self.net.to(device='cuda:{}'.format(self.gpu_ids))
        except:
            self.use_cuda = False

        # load the model if a path is specified or it is in inference mode
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
            bs = _input.size()
            if (len(bs) == 4):
                _input = _input.permute(0,3,1,2)
            if (len(bs) > 4):
                _input = _input.permute(0,1,4,2,3).contiguous().view(bs[0]*bs[1], bs[4], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = _input.to(device='cuda:{}'.format(self.gpu_ids)) if self.use_cuda else _input
            elif idx == 1:

                self.target = Variable(_input.to(device='cuda:{}'.format(self.gpu_ids))) if self.use_cuda else Variable(_input)
                assert self.input.size(2) == self.target.size(2)


    def forward(self, split):
        if split == 'train':
            self.prediction = self.net(Variable(self.input))
        elif split == 'test':
            with torch.no_grad():
                self.prediction = self.net(Variable(self.input))
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
        # inp_img = util.tensor2im(self.input, 'img')
        # seg_img = util.tensor2im(self.pred_seg, 'lbl')
        # ture_seg = util.tensor2im(self.target, 'gt')
        inp_img = np.transpose(self.input.cpu().numpy().astype(np.uint8),(0,2,3,1))
        seg_img= np.transpose(self.pred_seg.cpu().numpy().astype(np.uint8),(0,2,3,1)).squeeze(3)
        true_seg= np.transpose(self.target.cpu().numpy().astype(np.uint8),(0,2,3,1)).squeeze(3)
        return OrderedDict([('out_S', seg_img), ('inp_S', inp_img),('true_S', true_seg)])

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))

    # returns the fp/bp times of the model
    def get_fp_bp_time (self, size=None):
        if size is None:
            size = (1, 1, 160, 160, 96)

        inp_array = Variable(torch.zeros(*size)).cuda()
        out_array = Variable(torch.zeros(*size)).cuda()
        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

        bsize = size[0]
        return fp/float(bsize), bp/float(bsize)

    def save(self, epoch_label):
        self.save_network(self.net, 'S', epoch_label, self.gpu_ids)
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
