import torch
from tqdm import tqdm
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from model import LandmarkLoss
import gc
from config import Config


class TrainingMetrics:
	"""
	Class to store and manage training metrics during model training.
	"""
	
	def __init__(self):
		self.train_losses = []
		self.val_losses = []
		self.best_val_loss = float('inf')
	
	def update_train(self, total_loss):
		"""Append training loss for the current epoch."""
		self.train_losses.append(total_loss)
	
	def update_val(self, total_loss):
		"""
		Append validation loss for the current epoch and check if it is the best so far.

		Returns:
			True if this is the best validation loss so far; otherwise, False.
		"""
		self.val_losses.append(total_loss)
		if total_loss < self.best_val_loss:
			self.best_val_loss = total_loss
			return True
		return False
	
	def save_metrics(self, save_dir):
		"""
		Save training metrics to a JSON file in the specified directory.

		Args:
			save_dir: Directory where the metrics file should be saved.
		"""
		metrics_dict = {
			'train_losses': self.train_losses,
			'val_losses': self.val_losses,
			'best_val_loss': self.best_val_loss
		}
		with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
			json.dump(metrics_dict, f)
	
	def plot_losses(self, save_dir):
		"""
		Plot training and validation loss curves and save the plot to a file.

		Args:
			save_dir: Directory where the plot should be saved.
		"""
		plt.figure(figsize=(12, 8))
		plt.subplot(1, 1, 1)
		plt.plot(self.train_losses, label='Train Loss')
		plt.plot(self.val_losses, label='Validation Loss')
		plt.title('Total Loss vs Epoch')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		plt.savefig(os.path.join(save_dir, 'loss_plots.png'))
		plt.close()


def train(net, trainloader, lr, device, epochs: int, verbose=False, global_params=None, proximal_mu=0.0):
	"""
	Train the network on the training set with an optional FedProx proximal term.

	Args:
		net: The PyTorch model.
		trainloader: DataLoader for training data.
		lr: Learning rate.
		device: Device to train on.
		epochs: Number of training epochs.
		verbose: If True, print loss information.
		global_params: If provided, contains the global parameters for FedProx.
		proximal_mu: Hyperparameter for the FedProx proximal term.
	"""
	criterion = LandmarkLoss()
	optimizer = torch.optim.Adam(net.parameters(), lr)
	net.train()
	
	for epoch in tqdm(range(epochs), desc='Training local model...'):
		epoch_loss = 0.0
		for batch in trainloader:
			images, landmarks = batch["image"].to(device), batch["landmarks_norm"].to(device)
			optimizer.zero_grad()
			outputs = net(images)
			losses = criterion(outputs, landmarks)
			loss_value = losses['total_loss']
			
			# If using FedProx, add the proximal term.
			if global_params is not None:
				proximal_term = 0.0
				for local_param, global_param in zip(net.parameters(), global_params):
					proximal_term += torch.square((local_param - global_param).norm(2))
				loss_value = loss_value + (proximal_mu / 2) * proximal_term
			
			loss_value.backward()
			optimizer.step()
			epoch_loss += loss_value.item() * images.shape[0]
		
		epoch_loss /= len(trainloader.dataset)
		gc.collect()
		torch.cuda.empty_cache()
		if verbose:
			print(f"Epoch {epoch + 1}: train loss {epoch_loss}")


def test(net, testloader, device):
	"""
	Evaluate the network on the test/validation set.

	Args:
		net: The PyTorch model.
		testloader: DataLoader for test data.
		device: Device to evaluate on.

	Returns:
		The average loss over the dataset.
	"""
	criterion = LandmarkLoss()
	loss = 0.0
	net.eval()
	with torch.no_grad():
		for batch in testloader:
			images, landmarks = batch["image"].to(device), batch["landmarks_norm"].to(device)
			outputs = net(images)
			losses = criterion(outputs, landmarks)
			loss += losses['total_loss'].item() * images.shape[0]
	loss /= len(testloader.dataset)
	return loss


def train_epoch(model, train_loader, criterion, optimizer, epoch, device):
	"""
	Train the model for one epoch.

	Args:
		model: The PyTorch model.
		train_loader: DataLoader for training data.
		criterion: Loss function.
		optimizer: Optimizer.
		epoch: Current epoch number.
		device: Device for training.

	Returns:
		The average training loss for the epoch.
	"""
	model.train()
	total_loss = 0
	progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
	for batch in progress_bar:
		imgs = batch['image'].to(device)
		landmarks = batch['landmarks_norm'].to(device)
		optimizer.zero_grad()
		outputs = model(imgs)
		losses = criterion(outputs, landmarks)
		losses['total_loss'].backward()
		optimizer.step()
		total_loss += losses['total_loss'].item() * imgs.shape[0]
		progress_bar.set_postfix({'loss': f"{losses['total_loss'].item():.4f}"})
	avg_loss = total_loss / len(train_loader.dataset)
	return avg_loss


def validate_epoch(model, val_loader, criterion, epoch, device):
	"""
	Validate the model for one epoch.

	Args:
		model: The PyTorch model.
		val_loader: DataLoader for validation data.
		criterion: Loss function.
		epoch: Current epoch number.
		device: Device for validation.

	Returns:
		The average validation loss for the epoch.
	"""
	model.eval()
	total_loss = 0
	with torch.no_grad():
		progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
		for batch in progress_bar:
			imgs = batch['image'].to(device)
			landmarks = batch['landmarks_norm'].to(device)
			outputs = model(imgs)
			losses = criterion(outputs, landmarks)
			total_loss += losses['total_loss'].item() * imgs.shape[0]
			progress_bar.set_postfix({'val_loss': f"{losses['total_loss'].item():.4f}"})
	avg_loss = total_loss / len(val_loader.dataset)
	return avg_loss


def train_centralised_and_local(model, train_loader, val_loader, criterion, optimizer,
                                num_epochs, device, save_dir, checkpoint_interval=5,
                                model_type='centralized', hospital=None, logger=None):
	"""
	Main training loop for centralized and local models.

	Args:
		model: The PyTorch model.
		train_loader: DataLoader for training data.
		val_loader: DataLoader for validation data.
		criterion: Loss function.
		optimizer: Optimizer.
		num_epochs: Number of epochs to train.
		device: Device for training.
		save_dir: Directory to save checkpoints and metrics.
		checkpoint_interval: Interval for saving model checkpoints.
		model_type: String indicating the type of model ('centralized' or 'local').
		hospital: (Optional) Hospital name for local models.
		logger: (Optional) Logger instance for logging training info.
	"""
	save_dir = Path(save_dir)
	save_dir.mkdir(parents=True, exist_ok=True)
	metrics = TrainingMetrics()
	model = model.to(device)
	initial_lr = optimizer.param_groups[0]['lr']
	
	for epoch in range(num_epochs):
		print(f"\nEpoch {epoch + 1}/{num_epochs}")
		if epoch == 30:
			for param_group in optimizer.param_groups:
				param_group['lr'] = initial_lr / 10
			print(f"Reducing learning rate to {initial_lr / 10}")
		
		train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, device)
		val_loss = validate_epoch(model, val_loader, criterion, epoch, device)
		metrics.update_train(train_loss)
		is_best = metrics.update_val(val_loss)
		metrics.save_metrics(save_dir)
		metrics.plot_losses(save_dir)
		
		if is_best:
			if logger is not None:
				logger.log_best_loss(model_type=model_type, loss=val_loss, epoch=epoch + 1, hospital=hospital)
			print(f"New best model with validation loss: {val_loss:.4f}")
			save_checkpoint(model, save_dir, name='best_model.pth')


def save_checkpoint(model, save_dir, name):
	"""
	Save the model checkpoint.

	Args:
		model: The PyTorch model.
		save_dir: Directory where to save the checkpoint.
		name: Filename for the checkpoint.
	"""
	checkpoint = {
		'model_state_dict': model.state_dict(),
	}
	torch.save(checkpoint, os.path.join(save_dir, name))
