import unittest
import torch

# Add project root to sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.networks.unet_2D import Unet2D

class TestModels(unittest.TestCase):

    def _run_unet2d_test(self, device_str):
        device = torch.device(device_str)
        
        # Minimal configuration for Unet2D
        # Typically, these would come from a loaded JSON config (via ModelOpts in the project)
        # For Unet2D, key params are: feature_scale, n_classes, is_deconv, in_channels, is_batchnorm
        config = {
            'feature_scale': 1, # To keep model small
            'n_classes': 3,
            'is_deconv': False,
            'in_channels': 1,
            'is_batchnorm': True
        }

        model = Unet2D(
            feature_scale=config['feature_scale'],
            n_classes=config['n_classes'],
            is_deconv=config['is_deconv'],
            in_channels=config['in_channels'],
            is_batchnorm=config['is_batchnorm']
        ).to(device)
        
        model.eval() # Set to evaluation mode

        # Dummy input tensor (Batch, Channels, Height, Width)
        batch_size = 2
        img_height = 64 # Must be divisible by 2^N where N is num pool layers (e.g. 4 for Unet2D default)
        img_width = 64  # Or ensure model handles padding
        dummy_input = torch.randn(batch_size, config['in_channels'], img_height, img_width).to(device)

        with torch.no_grad():
            output = model(dummy_input)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.device.type, device.type)
        
        # Expected output shape: (Batch, NumClasses, Height, Width)
        # Unet2D output height/width should match input if padding is handled correctly (which it usually is)
        self.assertEqual(output.shape, (batch_size, config['n_classes'], img_height, img_width))
        self.assertEqual(output.dtype, torch.float32) # Default for nn.Conv2d weights

    def test_unet2d_cpu(self):
        print("Running Unet2D test on CPU...")
        self._run_unet2d_test("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_unet2d_gpu(self):
        print("Running Unet2D test on GPU...")
        self._run_unet2d_test("cuda")

    # TODO: Add test for SwinUnet if time permits and complexity is manageable for unit test setup

if __name__ == '__main__':
    unittest.main()
