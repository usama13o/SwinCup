# Abstract level model definition
# Returns the model class for specified network type
import os


class ModelOpts:
    def __init__(self):
        self.gpu_ids = [0]
        self.isTrain = True
        self.continue_train = False
        self.which_epoch = int(0)
        self.save_dir = './checkpoints/default'
        self.model_type = 'unet'
        self.input_nc = 1
        self.output_nc = 4
        self.lr_rate = 1e-12
        self.l2_reg_weight = 0.0
        self.feature_scale = 4
        self.tensor_dim = '2D'
        self.path_pre_trained_model = None
        self.criterion = 'cross_entropy'
        self.type = 'seg'

        # Attention
        self.nonlocal_mode = 'concatenation'
        self.attention_dsample = (2,2)

        # Attention Classifier
        self.aggregation_mode = 'concatenation'


    def initialise(self, json_opts):
        opts = json_opts

        self.raw = json_opts
        self.gpu_ids = opts.gpu_ids
        self.isTrain = opts.isTrain
        self.save_dir = os.path.join(opts.checkpoints_dir, opts.experiment_name)
        self.model_type = opts.model_type
        self.input_nc = opts.input_nc
        self.output_nc = opts.output_nc
        self.continue_train = opts.continue_train
        self.which_epoch = opts.which_epoch
        self.img_size=224

        if hasattr(opts, 'type'): self.type = opts.type
        if hasattr(opts, 'l2_reg_weight'): self.l2_reg_weight = opts.l2_reg_weight
        if hasattr(opts, 'lr_rate'):       self.lr_rate = opts.lr_rate
        if hasattr(opts, 'feature_scale'): self.feature_scale = opts.feature_scale
        if hasattr(opts, 'tensor_dim'):    self.tensor_dim = opts.tensor_dim

        if hasattr(opts, 'path_pre_trained_model'): self.path_pre_trained_model = opts.path_pre_trained_model
        if hasattr(opts, 'criterion'):              self.criterion = opts.criterion

        if hasattr(opts, 'nonlocal_mode'): self.nonlocal_mode = opts.nonlocal_mode
        if hasattr(opts, 'attention_dsample'): self.attention_dsample = opts.attention_dsample
        # Classifier
        if hasattr(opts, 'aggregation_mode'): self.aggregation_mode = opts.aggregation_mode

        if hasattr(opts, 'optim'): self.optim= opts.optim
        if hasattr(opts, 'img_size'): self.img_size= opts.img_size
        if hasattr(opts, 'patch_size'): self.patch_size= opts.patch_size
        if hasattr(opts, 'embed_dim'): self.embed_dim= opts.embed_dim
        if hasattr(opts, 'embed_dims'): self.embed_dims= opts.embed_dims
        if hasattr(opts, 'num_levels'): self.num_levels= opts.num_levels
        if hasattr(opts, 'depth'): self.depth = opts.depth
        if hasattr(opts, 'depths'): self.depths = opts.depths
        if hasattr(opts, 'num_heads'): self.num_heads = opts.num_heads
        if hasattr(opts, 'window_size'): self.window_size = opts.window_size
        if hasattr(opts, 'device'): self.device = opts.device # Store device if passed in json_opts

    def initialise(self, json_opts, device=None): # Added device argument
        opts = json_opts

        self.raw = json_opts
        # Prefer device passed directly, else from json_opts, else default to None (or handle error)
        self.device = device if device is not None else getattr(opts, 'device', None)
        self.gpu_ids = opts.gpu_ids # This might be redundant if device is the primary controller
        self.isTrain = opts.isTrain
        # self.save_dir = os.path.join(opts.checkpoints_dir, opts.experiment_name) # This was original
        # save_dir is now set in pywich_test.py and passed via json_opts.save_dir
        self.save_dir = opts.save_dir if hasattr(opts, 'save_dir') else './checkpoints/default_init'

        self.model_type = opts.model_type
        self.input_nc = opts.input_nc
        self.output_nc = opts.output_nc
        self.continue_train = opts.continue_train
        self.which_epoch = opts.which_epoch
        self.img_size=224

        if hasattr(opts, 'type'): self.type = opts.type
        if hasattr(opts, 'l2_reg_weight'): self.l2_reg_weight = opts.l2_reg_weight
        if hasattr(opts, 'lr_rate'):       self.lr_rate = opts.lr_rate
        if hasattr(opts, 'feature_scale'): self.feature_scale = opts.feature_scale
        if hasattr(opts, 'tensor_dim'):    self.tensor_dim = opts.tensor_dim

        if hasattr(opts, 'path_pre_trained_model'): self.path_pre_trained_model = opts.path_pre_trained_model
        if hasattr(opts, 'criterion'):              self.criterion = opts.criterion

        if hasattr(opts, 'nonlocal_mode'): self.nonlocal_mode = opts.nonlocal_mode
        if hasattr(opts, 'attention_dsample'): self.attention_dsample = opts.attention_dsample
        # Classifier
        if hasattr(opts, 'aggregation_mode'): self.aggregation_mode = opts.aggregation_mode

        if hasattr(opts, 'optim'): self.optim= opts.optim
        if hasattr(opts, 'img_size'): self.img_size= opts.img_size
        if hasattr(opts, 'patch_size'): self.patch_size= opts.patch_size
        if hasattr(opts, 'embed_dim'): self.embed_dim= opts.embed_dim
        if hasattr(opts, 'embed_dims'): self.embed_dims= opts.embed_dims
        if hasattr(opts, 'num_levels'): self.num_levels= opts.num_levels
        if hasattr(opts, 'depth'): self.depth = opts.depth
        if hasattr(opts, 'depths'): self.depths = opts.depths
        if hasattr(opts, 'num_heads'): self.num_heads = opts.num_heads
        if hasattr(opts, 'window_size'): self.window_size = opts.window_size


def get_model(json_opts, device): # Added device argument

    # Neural Network Model Initialisation
    model = None
    model_opts = ModelOpts()
    # Pass both json_opts (which is json_opts.model from pywich_test) and device
    model_opts.initialise(json_opts, device) 

    # Print the model type
    print('\nInitialising model {}'.format(model_opts.model_type))

    model_type = model_opts.type
    if model_type == 'seg':
        # Return the model type
        from .feedforward_seg_model import FeedForwardSegmentation
        model = FeedForwardSegmentation()

    elif model_type == 'classifier':
        # Return the model type
        from .feedforward_classifier import FeedForwardClassifier
        model = FeedForwardClassifier()

    elif model_type == 'aggregated_classifier':
        # Return the model type
        from .aggregated_classifier import AggregatedClassifier
        model = AggregatedClassifier()


    # Initialise the created model
    model.initialize(model_opts)
    print("Model [%s] is created" % (model.name()))

    return model
