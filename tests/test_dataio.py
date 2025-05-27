import unittest
import os
import shutil
import numpy as np
from PIL import Image
import torch

# Add project root to sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataio.loader.glas_dataset import glas_dataset
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj # If needed for transform opts

# Helper function to create a dummy BMP image
def create_dummy_bmp(filepath, size=(64, 64), color=(0,0,0)):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img = Image.new('RGB', size, color)
    img.save(filepath, 'BMP')

class TestDataIO(unittest.TestCase):

    def setUp(self):
        self.test_dir = "temp_test_data_dir"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create dummy image files for glas_dataset
        self.glas_img_dir = os.path.join(self.test_dir, "glas_data")
        os.makedirs(self.glas_img_dir, exist_ok=True)
        
        create_dummy_bmp(os.path.join(self.glas_img_dir, "test1.bmp"), color=(255,0,0))
        create_dummy_bmp(os.path.join(self.glas_img_dir, "test1_anno.bmp"), color=(1,1,1)) # Mask
        create_dummy_bmp(os.path.join(self.glas_img_dir, "test2.bmp"), color=(0,255,0))
        create_dummy_bmp(os.path.join(self.glas_img_dir, "test2_anno.bmp"), color=(1,1,1)) # Mask
        create_dummy_bmp(os.path.join(self.glas_img_dir, "test3.bmp"), color=(0,0,255))
        create_dummy_bmp(os.path.join(self.glas_img_dir, "test3_anno.bmp"), color=(0,0,0)) # Different mask

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_glas_dataset_loading(self):
        # Test with split='all' to load all dummy images
        dataset = glas_dataset(root_dir=self.glas_img_dir, split='all', preload_data=False)
        self.assertEqual(len(dataset), 3)

        # Test __getitem__
        img, mask = dataset[0] # Input is HxWxC np array, target is HxW np array
        
        self.assertIsInstance(img, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(img.shape, (64, 64, 3)) # From open_image_np
        self.assertEqual(mask.shape, (64, 64))   # From open_target_np_glas (squeezes last dim if 1)

        # Test with preload_data=True
        dataset_preload = glas_dataset(root_dir=self.glas_img_dir, split='all', preload_data=True)
        self.assertEqual(len(dataset_preload), 3)
        img_pre, mask_pre = dataset_preload[0]
        self.assertIsInstance(img_pre, np.ndarray)
        self.assertIsInstance(mask_pre, np.ndarray)

    def test_epi_transform_on_dummy_data(self):
        # Create dummy augmentation options (usually from a config file)
        # For 'epi' transform, scale_size and patch_size are key if not using defaults
        class MockAugOpts:
            class EpiOpts:
                scale_size = (32, 32) # Smaller for testing
                # patch_size = (24, 24) # Not used in valid transform for epi
                # other opts like shift, rotate, etc., are mostly for train
            epi = EpiOpts()

        mock_aug_opts = MockAugOpts()
        
        # Get the 'valid' part of 'epi_transform'
        # The 'name' for get_dataset_transformation is like 'epi', 'glas', etc.
        transforms_dict = get_dataset_transformation('epi', opts=mock_aug_opts)
        self.assertIn('valid', transforms_dict)
        valid_transform = transforms_dict['valid']

        # Create dummy input data (numpy arrays, as output by dataset's __getitem__ before transform)
        dummy_img_np = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        dummy_mask_np = np.random.randint(0, 2, size=(64, 64), dtype=np.uint8) # Binary mask

        # The transform expects a dictionary {'image': img, 'label': mask}
        # However, the glas_dataset __getitem__ applies transform like: input, target = self.transform(input, target)
        # This suggests the transform callable itself should handle separate img, mask args,
        # OR the Compose object from pywick handles dicts internally.
        # Let's check pywick.transforms.Compose or the individual transform types.
        # The `ts.Compose` in `transforms.py` takes a list of transforms.
        # Many pywick transforms (like RandomFlip, RandomAffine) are designed to work on (image, mask) tuples.
        # Resize from myImageTransformations is the first one. Let's assume it takes (img, mask).

        # The current structure of epi_transform in transforms.py seems to be a Compose of transforms
        # that expect the (input, target) tuple directly, not a dictionary.
        # Example: ts.RandomFlip(h=True, v=True, p=...)
        # Resize(size=self.scale_size) is the first. If it expects (img, label), it's fine.

        transformed_img, transformed_mask = valid_transform(dummy_img_np, dummy_mask_np)
        
        self.assertIsInstance(transformed_img, torch.Tensor)
        self.assertIsInstance(transformed_mask, torch.Tensor)

        # Expected shape after epi_transform's validation part:
        # Resize to (32,32) -> ToTensor -> ChannelsFirst (C,H,W) -> TypeCast -> ChannelsLast (H,W,C) -> AddChannel (1,H,W,C) -> TypeCast
        # Image: (1, 32, 32, 3) (after ChannelsLast, then AddChannel adds batch-like dim for image)
        # Mask:  (1, 32, 32)   (after ChannelsLast, then AddChannel adds batch-like dim for mask, assuming mask becomes H,W before AddChannel)
        # Let's re-check epi_transform valid:
        # Resize(size=(32,32)) -> out H,W,C (img), H,W (mask)
        # ToTensor() -> converts H,W,C to C,H,W (img), H,W to 1,H,W or H,W (mask)
        # ChannelsFirst() -> no-op if already C,H,W
        # TypeCast()
        # ChannelsLast() -> C,H,W to H,W,C (img), 1,H,W to H,W,1 (mask) or H,W to H,W (mask)
        # AddChannel(axis=0) -> 1,H,W,C (img), 1,H,W,1 or 1,H,W (mask)
        # TypeCast()
        
        # Based on pywick's design, ToTensor usually makes image C,H,W and mask H,W or 1,H,W.
        # ChannelsFirst/Last and AddChannel manipulate this.
        # If AddChannel(axis=0) is the last spatial-like op:
        # Image: (1, num_channels, H, W) or (1, H, W, num_channels)
        # Mask: (1, H, W) or (1, 1, H, W)
        # The `epi_transform` ends with `ts.ChannelsLast(), ts.AddChannel(axis=0), ts.TypeCast(['float', 'long'])`
        # So, img: (1, 32, 32, 3), mask: (1, 32, 32) (if mask was H,W after ToTensor) or (1,32,32,1)

        self.assertEqual(transformed_img.ndim, 4) # Expecting something like (1, H, W, C) due to ChannelsLast then AddChannel
        self.assertEqual(transformed_img.shape[1], 32) # H
        self.assertEqual(transformed_img.shape[2], 32) # W
        self.assertEqual(transformed_img.shape[3], 3)  # C
        
        self.assertEqual(transformed_mask.ndim, 3) # Expecting (1, H, W)
        self.assertEqual(transformed_mask.shape[1], 32) # H
        self.assertEqual(transformed_mask.shape[2], 32) # W

        self.assertEqual(transformed_img.dtype, torch.float32)
        self.assertEqual(transformed_mask.dtype, torch.long)


if __name__ == '__main__':
    unittest.main()
