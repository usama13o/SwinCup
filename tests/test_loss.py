import unittest
import torch

# Add project root to sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.layers.loss import SoftDiceLoss, ce_loss, IoU_loss

class TestLossFunctions(unittest.TestCase):

    def _run_soft_dice_loss_test(self, device_str):
        device = torch.device(device_str)
        n_classes = 3
        batch_size = 2
        height, width = 32, 32

        # Dummy model output (logits)
        # Shape: (Batch, NumClasses, Height, Width)
        dummy_pred_logits = torch.randn(batch_size, n_classes, height, width).to(device)

        # Dummy target (ground truth class indices)
        # Shape: (Batch, Height, Width) or (Batch, 1, Height, Width)
        dummy_target_indices = torch.randint(0, n_classes, (batch_size, height, width)).to(device)
        # SoftDiceLoss (and ce_loss it uses) expects target for ce_loss to be (B, H, W)
        # and for one_hot_encoder it will be (B,1,H,W) or (B,H,W) which is handled by One_Hot's view and squeeze.

        # Instantiate loss
        soft_dice_criterion = SoftDiceLoss(n_classes=n_classes).to(device)
        
        loss = soft_dice_criterion(dummy_pred_logits, dummy_target_indices)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0) # Scalar loss
        self.assertTrue(loss.item() >= 0) # Loss should be non-negative (or close to it)
        self.assertEqual(loss.device.type, device.type)

    def test_soft_dice_loss_cpu(self):
        print("Running SoftDiceLoss test on CPU...")
        self._run_soft_dice_loss_test("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_soft_dice_loss_gpu(self):
        print("Running SoftDiceLoss test on GPU...")
        self._run_soft_dice_loss_test("cuda")

    def _run_ce_loss_test(self, device_str):
        device = torch.device(device_str)
        n_classes = 3
        batch_size = 2
        height, width = 32, 32

        dummy_pred_logits = torch.randn(batch_size, n_classes, height, width).to(device)
        # ce_loss expects target of shape [B, 1, H, W] or [B, H, W] containing class indices
        dummy_target_indices = torch.randint(0, n_classes, (batch_size, 1, height, width)).to(device)
        
        # Test without weights
        loss_no_weights = ce_loss(dummy_pred_logits, dummy_target_indices)
        self.assertIsInstance(loss_no_weights, torch.Tensor)
        self.assertEqual(loss_no_weights.ndim, 0)
        self.assertTrue(loss_no_weights.item() >= 0)
        self.assertEqual(loss_no_weights.device.type, device.type)

        # Test with weights
        class_weights = torch.rand(n_classes).to(device)
        loss_with_weights = ce_loss(dummy_pred_logits, dummy_target_indices, weights=class_weights)
        self.assertIsInstance(loss_with_weights, torch.Tensor)
        self.assertEqual(loss_with_weights.ndim, 0)
        self.assertTrue(loss_with_weights.item() >= 0)

    def test_ce_loss_cpu(self):
        print("Running ce_loss test on CPU...")
        self._run_ce_loss_test("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ce_loss_gpu(self):
        print("Running ce_loss test on GPU...")
        self._run_ce_loss_test("cuda")


    def _run_iou_loss_test(self, device_str):
        device = torch.device(device_str)
        n_classes = 3
        batch_size = 2
        height, width = 32, 32

        dummy_pred_logits = torch.randn(batch_size, n_classes, height, width).to(device)
        # IoU_loss expects target of shape [B, H, W] or [B, 1, H, W] containing class indices
        dummy_target_indices = torch.randint(0, n_classes, (batch_size, height, width)).to(device)

        iou_criterion = IoU_loss(n_classes=n_classes).to(device)
        loss = iou_criterion(dummy_pred_logits, dummy_target_indices)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(loss.item() >= 0) # (1-iou) + ce_loss, should be >=0
        self.assertEqual(loss.device.type, device.type)

    def test_iou_loss_cpu(self):
        print("Running IoU_loss test on CPU...")
        self._run_iou_loss_test("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_iou_loss_gpu(self):
        print("Running IoU_loss test on GPU...")
        self._run_iou_loss_test("cuda")


if __name__ == '__main__':
    unittest.main()
