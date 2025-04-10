import torch
import torch.nn as nn
from backbones import hg1, hg2, hg8
import math
import dsntnn
import matplotlib.pyplot as plt
from monai.networks.nets import UNet, HighResNet


def _init_weights(model):
	"""
	Initialize the weights of convolutional layers in the model using Kaiming initialization.

	Args:
		model (nn.Module): PyTorch model whose weights need to be initialized
	"""
	for m in model.modules():
		if type(m) in {
			nn.Conv2d,
		}:
			nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
			
			if m.bias is not None:
				fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
				bound = 1 / math.sqrt(fan_out)
				nn.init.normal_(m.bias, -bound, bound)


class CoordRegressionNetwork(torch.nn.Module):
	"""
	A two-stage coordinate regression network for spine landmark detection.

	The network first performs global landmark detection on a downsampled image,

	Args:
		arch (str): Backbone architecture ('hg1', 'hg2', 'hg8', 'unet', 'hrnet')
		n_locations_global (int): Number of landmark points to detect globally
		n_ch (int): Number of input channels
		n_blocks (int): Number of residual blocks in backbone for hg
	"""
	
	def __init__(self, arch, n_locations_global, n_ch, n_blocks):
		super().__init__()
		
		self.arch = arch
		
		# Global model for initial landmark detection
		if self.arch == 'hg1':
			self.global_model = hg1(pretrained=False, num_classes=n_locations_global * 2, num_blocks=n_blocks, n_ch=n_ch)
		elif self.arch == 'hg2':
			self.global_model = hg2(pretrained=False, num_classes=n_locations_global * 2, num_blocks=n_blocks, n_ch=n_ch)
		elif self.arch == 'hg8':
			self.global_model = hg8(pretrained=False, num_classes=n_locations_global * 2, num_blocks=n_blocks, n_ch=n_ch)
		elif self.arch == 'unet':
			self.global_model = UNet(spatial_dims=2, in_channels=1, out_channels=n_locations_global * 2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
		elif self.arch == 'hrnet':
			self.global_model = HighResNet(spatial_dims=2, in_channels=1, out_channels=n_locations_global * 2)
			
		# Initialize all weights
		#_init_weights(self.global_model)
		# DSNT layer
		self.hm_conv = nn.Conv2d(in_channels=n_locations_global * 2, out_channels = n_locations_global, kernel_size=1, bias=False)
		
		
	
	def forward(self, img):
		"""
		Forward pass through the network.

		Args:
			img (torch.Tensor): Input image batch [B, C, H, W]

		Returns:
				- global_coords: Global landmark coordinates [B, N, 2]
				- global_heatmaps: Global detection heatmaps
		"""
		# Global prediction
		# x = nn.functional.avg_pool2d(img, self.kernel_size)
		global_features = self.global_model(img)
		# 2. Use a 1x1 conv to get one unnormalized heatmap per location
		unnormalized_heatmaps = self.hm_conv(global_features[-1] if self.arch.startswith('hg') else global_features)
		global_heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
		global_coords = dsntnn.dsnt(global_heatmaps)
		
		return global_coords, global_heatmaps


class LandmarkLoss(nn.Module):
	"""
	Multi-component loss function for landmark detection training.

	Combines global detection loss, local refinement loss, and optional spatial smoothness loss.
	During warmup epochs, only global loss is used. After warmup, all components are active.

	Args:
		warmup_epoch (int): Number of epochs to train with global loss only
		sigma_t (float, optional): Temperature parameter for Jensen-Shannon loss. Defaults to 1.0
		smoothness_weight (float, optional): Weight for smoothness loss term. Defaults to 0.1
	"""
	
	def __init__(self, sigma_t=1.0,
	             smoothness_weight=0.0, anatomical_weight=0.0
	             ):
		super().__init__()
		self.sigma_t = sigma_t
		self.smoothness_weight = smoothness_weight
		self.anatomical_weight = anatomical_weight
	
	def compute_smoothness_loss(self, coords):
		"""
		Calculate spatial smoothness loss between adjacent vertebrae landmarks.
		Encourages consistent spacing and alignment between vertebrae.

		Args:
			coords (torch.Tensor): Predicted landmark coordinates [batch_size, n_points, 2]

		Returns:
			torch.Tensor: Scalar smoothness loss value
		"""
		# Extract vertebrae coordinates (excluding sacrum/femur)
		vertebrae_coords = coords[:, 4:].view(coords.shape[0], 6, 4, 2)
		
		# Calculate differences between adjacent vertebrae positions
		vertebrae_diffs = vertebrae_coords[:, 1:] - vertebrae_coords[:, :-1]
		
		# Compute variance of differences as smoothness measure
		smoothness_loss = torch.var(vertebrae_diffs, dim=1).mean()
		
		return smoothness_loss
	
	def compute_anatomical_consistency_loss(self, coords):
		"""
		Enhanced anatomical consistency loss that considers vertebrae spacing,
		alignment, and relative sizes.
		"""
		# Reshape to separate vertebrae (excluding sacrum/femur)
		vertebrae_coords = coords[:, 4:].view(coords.shape[0], 6, 4, 2)
		
		# 1. Inter-vertebral spacing consistency
		centroids = vertebrae_coords.mean(dim=2)  # [batch, 6, 2]
		spacing = centroids[:, 1:] - centroids[:, :-1]  # [batch, 5, 2]
		spacing_variance = torch.var(torch.norm(spacing, dim=-1), dim=-1).mean()
		
		# 2. Vertebrae size consistency
		sizes = torch.norm(vertebrae_coords[:, :, [0, 2], :] -
		                   vertebrae_coords[:, :, [1, 3], :], dim=-1)  # Width and height
		size_variance = torch.var(sizes, dim=1).mean()
		
		# 3. Alignment consistency (vertebrae should be roughly aligned)
		centroids_x = centroids[..., 0]  # x-coordinates of centroids
		alignment_variance = torch.var(centroids_x, dim=-1).mean()
		
		return spacing_variance + size_variance + 0.5 * alignment_variance
	
	def forward(self, outputs, targets):
		coords, heatmaps = outputs
		# Global detection losses
		reg_losses = dsntnn.js_reg_losses(heatmaps.cpu(), targets.cpu(), sigma_t=self.sigma_t)
		euc_losses = dsntnn.euclidean_losses(coords.cpu(), targets.cpu())
		loss = dsntnn.average_loss(euc_losses + reg_losses)
		
		# Add anatomical constraints during warmup
		smoothness_loss = self.compute_smoothness_loss(coords.cpu())
		anatomical_loss = self.compute_anatomical_consistency_loss(coords.cpu())
		
		total_loss = (loss +
		              self.smoothness_weight * smoothness_loss +
		              self.anatomical_weight * anatomical_loss)
			
		return {
			'total_loss': total_loss,
			'global_loss': loss,
			'smoothness_loss': smoothness_loss,
			'anatomical_loss': anatomical_loss
		}


def debug_overfit_training(model, imgs, lands, num_epochs=100, warmup_epoch=10, plot_interval=5):
	"""
	Debug model training by overfitting on a single batch with visualization.
	Useful for verifying model capacity and loss function behavior.

	Args:
		model (nn.Module): The spine landmark detection model
		imgs (torch.Tensor): Single batch of training images [batch_size, channels, height, width]
		lands (torch.Tensor): Ground truth landmarks for the batch [batch_size, n_points, 2]
		num_epochs (int, optional): Number of training epochs. Defaults to 100
		warmup_epoch (int, optional): Epochs of global-only training. Defaults to 10
		plot_interval (int, optional): Frequency of visualization updates. Defaults to 5
	"""
	criterion = LandmarkLoss(
		sigma_t=1.0,
		smoothness_weight=0.05,
		anatomical_weight=0.05
	)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	
	model.train()
	num_batch = 0  # Index for visualization
	
	for epoch in range(num_epochs):
		optimizer.zero_grad()
		
		# Forward pass and loss computation
		with torch.set_grad_enabled(True):
			outputs = model(imgs, epoch)
			losses = criterion(outputs, lands)
			
			# Backward pass
			losses['total_loss'].backward()
			optimizer.step()
		
		# Print training progress
		print(f'\nEpoch {epoch}:')
		for loss_name, loss_value in losses.items():
			print(f'{loss_name}: {loss_value.item():.6f}')
		
		# Periodic visualization
		if epoch % plot_interval == 0:
			visualize_predictions(imgs, lands, outputs, epoch, num_batch, warmup_epoch)


def visualize_predictions(imgs, lands, outputs, epoch, num_batch, warmup_epoch):
	"""
	Create visualization comparing model predictions with ground truth.
	Shows both global and local predictions (when available) against ground truth.

	Args:
		imgs (torch.Tensor): Input image batch
		lands (torch.Tensor): Ground truth landmarks
		outputs (tuple): Model predictions (format varies by training phase)
		epoch (int): Current training epoch
		num_batch (int): Index of batch to visualize
		warmup_epoch (int): Number of warmup epochs
	"""
	with torch.no_grad():
		plt.figure(figsize=(10, 10))
		
		# Display input image
		plt.imshow(imgs[num_batch, 0].cpu().detach(), cmap='gray')
		
		# Convert ground truth to pixel coordinates
		coords_true_red = dsntnn.normalized_to_pixel_coordinates(lands, size=imgs.shape[2:])
		
		coords, _ = outputs
		coords_rec_global = dsntnn.normalized_to_pixel_coordinates(coords, size=imgs.shape[2:])
			
		plt.scatter(coords_rec_global[num_batch, :, 0].cpu(),
		            coords_rec_global[num_batch, :, 1].cpu(),
		            c='b', label='global', s=5)
		# Plot ground truth landmarks
		plt.scatter(coords_true_red[num_batch, :, 0].cpu(),
		            coords_true_red[num_batch, :, 1].cpu(),
		            s=5, c='y', label='GT')
		
		plt.legend()
		plt.title(f'Epoch {epoch}')
		plt.show()
		plt.close()


# Main execution block for testing the model
if __name__ == '__main__':
	# Set model configuration parameters
	arch = 'hg2'
	
	# Initialize the coordinate regression model
	model = CoordRegressionNetwork(
		arch=arch,
		n_locations_global=28,
		n_ch=1,  # Single channel (grayscale) input
		n_blocks=3,  # Number of hourglass blocks
	)
	
	# Dummy input
	dummy_input = torch.randn(4, 1, 1024, 1024)
	
	out = model.global_model(dummy_input)
	
	hm = model.hm_conv(out[-1] if arch.startswith('hg') else out)
	
	co, hms = model(dummy_input)
	
	
	
	# Initialize loss function and optimizer
	criterion = LandmarkLoss(
		sigma_t=1.0,
		smoothness_weight=0.05,
		anatomical_weight=0.05
	)
	
	