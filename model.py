"""
Models for spine landmark detection using coordinate regression.

This module contains the main network architecture for landmark detection
along with the loss function and visualization utilities.
"""
import torch
import torch.nn as nn
import math
import dsntnn
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from monai.networks.nets import UNet, HighResNet
from backbones import hg1, hg2, hg8


def init_weights(model: nn.Module) -> None:
	"""
	Initialize model weights using Kaiming initialization.

	Args:
		model: PyTorch model whose weights need initialization
	"""
	for m in model.modules():
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
			
			if m.bias is not None:
				fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
				bound = 1 / math.sqrt(fan_out)
				nn.init.normal_(m.bias, -bound, bound)


class CoordRegressionNetwork(nn.Module):
	"""
	Coordinate regression network for spine landmark detection.

	This network uses a backbone feature extractor followed by a heatmap
	regression approach to predict landmark coordinates.
	"""
	
	def __init__(self, arch: str, n_locations_global: int, n_ch: int, n_blocks: int):
		"""
		Initialize the coordinate regression network.

		Args:
			arch: Backbone architecture ('hg1', 'hg2', 'hg8', 'unet', 'hrnet')
			n_locations_global: Number of landmark points to detect
			n_ch: Number of input channels
			n_blocks: Number of residual blocks in hourglass backbone
		"""
		super().__init__()
		
		self.arch = arch
		
		# Initialize backbone based on architecture choice
		if self.arch == 'hg1':
			self.global_model = hg1(pretrained=False, num_classes=n_locations_global * 2,
			                        num_blocks=n_blocks, n_ch=n_ch)
		elif self.arch == 'hg2':
			self.global_model = hg2(pretrained=False, num_classes=n_locations_global * 2,
			                        num_blocks=n_blocks, n_ch=n_ch)
		elif self.arch == 'hg8':
			self.global_model = hg8(pretrained=False, num_classes=n_locations_global * 2,
			                        num_blocks=n_blocks, n_ch=n_ch)
		elif self.arch == 'unet':
			self.global_model = UNet(spatial_dims=2, in_channels=1, out_channels=n_locations_global * 2,
			                         channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
			                         num_res_units=2)
		elif self.arch == 'hrnet':
			self.global_model = HighResNet(spatial_dims=2, in_channels=1, out_channels=n_locations_global * 2)
		
		# DSNT layer for converting heatmaps to coordinates
		self.hm_conv = nn.Conv2d(in_channels=n_locations_global * 2, out_channels=n_locations_global,
		                         kernel_size=1, bias=False)
	
	def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Forward pass through the network.

		Args:
			img: Input image batch [B, C, H, W]

		Returns:
			Tuple containing:
				- global_coords: Normalized landmark coordinates [B, N, 2]
				- global_heatmaps: Detection heatmaps [B, N, H, W]
		"""
		# Extract features from backbone
		global_features = self.global_model(img)
		
		# For hourglass networks, take the last output from the stack
		features = global_features[-1] if self.arch.startswith('hg') else global_features
		
		# Convert to unnormalized heatmaps
		unnormalized_heatmaps = self.hm_conv(features)
		
		# Apply softmax to get proper heatmaps
		global_heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
		
		# Convert heatmaps to normalized coordinates using DSNT
		global_coords = dsntnn.dsnt(global_heatmaps)
		
		return global_coords, global_heatmaps


class LandmarkLoss(nn.Module):
	"""
	Multi-component loss function for landmark detection.

	Combines coordinate regression loss with optional spatial constraints.
	"""
	
	def __init__(self, sigma_t: float = 1.0):
		"""
		Initialize the landmark loss function.

		Args:
			sigma_t: Temperature parameter for Jensen-Shannon loss
			smoothness_weight: Weight for spatial smoothness constraint
			anatomical_weight: Weight for anatomical consistency constraint
		"""
		super().__init__()
		self.sigma_t = sigma_t
	
	
	def forward(self, outputs: Tuple[torch.Tensor, torch.Tensor],
	            targets: torch.Tensor) -> Dict[str, torch.Tensor]:
		"""
		Compute the loss between predicted and target landmark coordinates.

		Args:
			outputs: Tuple of (coords, heatmaps) from the model
			targets: Target landmark coordinates [batch_size, n_points, 2]

		Returns:
			Dictionary of loss components
		"""
		coords, heatmaps = outputs
		
		# Global detection losses
		reg_losses = dsntnn.js_reg_losses(heatmaps, targets, sigma_t=self.sigma_t)
		euc_losses = dsntnn.euclidean_losses(coords, targets)
		loss = dsntnn.average_loss(euc_losses + reg_losses)
		
		return {
			'total_loss': loss,
		}


def visualize_predictions(imgs: torch.Tensor,
                          ground_truth: torch.Tensor,
                          outputs: Tuple[torch.Tensor, torch.Tensor],
                          epoch: int,
                          batch_idx: int = 0,
                          save_path: Optional[str] = None):
	"""
	Visualize predictions against ground truth landmarks.

	Args:
		imgs: Input image batch [B, C, H, W]
		ground_truth: Ground truth landmarks [B, N, 2]
		outputs: Model predictions (coords, heatmaps)
		epoch: Current training epoch
		batch_idx: Index of sample to visualize from the batch
		save_path: Optional path to save the visualization
	"""
	with torch.no_grad():
		plt.figure(figsize=(10, 10))
		
		# Display input image
		plt.imshow(imgs[batch_idx, 0].cpu().detach(), cmap='gray')
		
		# Convert normalized coordinates to pixel coordinates
		coords_true = dsntnn.normalized_to_pixel_coordinates(ground_truth, size=imgs.shape[2:])
		
		coords, _ = outputs
		coords_pred = dsntnn.normalized_to_pixel_coordinates(coords, size=imgs.shape[2:])
		
		# Plot predicted landmarks
		plt.scatter(coords_pred[batch_idx, :, 0].cpu(),
		            coords_pred[batch_idx, :, 1].cpu(),
		            c='b', label='Predicted', s=5)
		
		# Plot ground truth landmarks
		plt.scatter(coords_true[batch_idx, :, 0].cpu(),
		            coords_true[batch_idx, :, 1].cpu(),
		            s=5, c='y', label='Ground Truth')
		
		plt.legend()
		plt.title(f'Epoch {epoch}')
		
		if save_path:
			plt.savefig(save_path)
		
		plt.close()