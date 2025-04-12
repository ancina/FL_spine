"""
Federated learning client implementation for spine landmark detection.

This module defines the Flower client that trains local models at each hospital/client.
"""
import torch
import flwr as fl
import numpy as np
from typing import List, Dict, Tuple

from flwr.common import Context, Scalar
from torch.utils.data import DataLoader

from dataset import SpineDataset, create_train_val_test_split
from model import CoordRegressionNetwork
from config import Config
from training_utils import train, test
from strategy import set_weights

# Create data splits once for all clients
SPLITS = create_train_val_test_split(Config.DATA_DIR, debug=Config.DEBUG, train_ratio=0.8, val_ratio=0.1, seed=42)


def get_parameters(net: torch.nn.Module) -> List[np.ndarray]:
	"""
	Convert PyTorch model state to a list of NumPy arrays.

	Args:
		net: PyTorch model

	Returns:
		List of model parameters as NumPy arrays
	"""
	return [val.cpu().numpy() for _, val in net.state_dict().items()]


class FlowerClient(fl.client.NumPyClient):
	"""
	Flower client for spine landmark detection federated learning.

	Implements the NumPyClient interface for training and evaluating
	local models at each hospital/node.
	"""
	
	def __init__(self, pid: str, net: torch.nn.Module, trainloader: DataLoader, valloader: DataLoader):
		"""
		Initialize the Flower client.

		Args:
			pid: Partition/client ID
			net: PyTorch model
			trainloader: DataLoader for training data
			valloader: DataLoader for validation data
		"""
		self.pid = pid
		self.net = net
		self.trainloader = trainloader
		self.valloader = valloader
		
		# Set device for computation
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.net.to(self.device)
		
		print(f"Client {pid} initialized. Using device: {self.device}")
	
	def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
		"""
		Get model parameters.

		Args:
			config: Configuration from server (not used)

		Returns:
			List of model parameters as NumPy arrays
		"""
		return get_parameters(self.net)
	
	def fit(
			self,
			parameters: List[np.ndarray],
			config: Dict[str, Scalar]
	) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
		"""
		Train the model using parameters received from the server.

		Args:
			parameters: Global model parameters
			config: Configuration dictionary with training settings:
				- server_round: Current round number
				- local_epochs: Number of local training epochs
				- lr: Learning rate
				- use_fedprox: Whether to use FedProx
				- proximal_mu: FedProx regularization strength

		Returns:
			Tuple of (updated parameters, number of training samples, metrics)
		"""
		# Extract configuration
		server_round = config["server_round"]
		local_epochs = config["local_epochs"]
		current_lr = config["lr"]
		use_fedprox = config["use_fedprox"]
		proximal_mu = config["proximal_mu"]
		
		print(f"[round {server_round}] Client {self.pid} starting training")
		print(f"Learning rate: {current_lr}, Local epochs: {local_epochs}")
		print(f"FedProx: {use_fedprox}, mu: {proximal_mu}")
		
		# Update local model with global parameters
		set_weights(self.net, parameters)
		
		# If using FedProx, keep a copy of global parameters
		global_params = None
		if use_fedprox:
			print('Using FedProx: copying global parameters')
			global_params = [val.detach().clone() for val in self.net.parameters()]
		
		# Train the local model
		train(
			net=self.net,
			trainloader=self.trainloader,
			lr=current_lr,
			device=self.device,
			epochs=local_epochs,
			global_params=global_params,
			proximal_mu=proximal_mu,
			verbose=True
		)
		
		# Return updated parameters and dataset size
		updated_parameters = get_parameters(self.net)
		num_examples = len(self.trainloader.dataset)
		
		return updated_parameters, num_examples, {}
	
	def evaluate(
			self,
			parameters: List[np.ndarray],
			config: Dict[str, Scalar]
	) -> Tuple[float, int, Dict[str, Scalar]]:
		"""
		Evaluate the model using parameters received from the server.

		Args:
			parameters: Global model parameters
			config: Configuration dictionary

		Returns:
			Tuple of (loss, number of validation samples, metrics)
		"""
		# Update local model with global parameters
		set_weights(self.net, parameters)
		
		# Evaluate the model
		loss = test(self.net, self.valloader, device=self.device)
		num_examples = len(self.valloader.dataset)
		
		# Return loss, dataset size, and metrics
		return float(loss), num_examples, {'loss': loss}


def client_fn(context: Context) -> fl.client.Client:
	"""
	Create a Flower client for a specific hospital.

	Args:
		context: Flower context with client configuration

	Returns:
		Configured Flower client
	"""
	# Extract partition ID from context
	partition_id = context.node_config["partition-id"]
	partition_id_int = int(partition_id)
	
	# Map partition ID to hospital code
	hospital_codes = {
		0: 'MAD', 1: 'BCN', 2: 'BOR', 3: 'IST', 4: 'ZUR'
	}
	hospital = hospital_codes[partition_id_int]
	
	print(f"Client {partition_id} assigned to hospital: {hospital}")
	
	# Initialize model architecture
	model = CoordRegressionNetwork(
		arch=Config.ARCH,
		n_locations_global=28,
		n_ch=1,
		n_blocks=Config.N_BLOCKS
	)
	
	# Create hospital-specific datasets
	train_data = SpineDataset(
		Config.DATA_DIR,
		SPLITS['by_hospital'][hospital]['train'],
		mode='train'
	)
	
	val_data = SpineDataset(
		Config.DATA_DIR,
		SPLITS['by_hospital'][hospital]['val'],
		mode='val'
	)
	
	# Create data loaders
	train_loader = torch.utils.data.DataLoader(
		train_data,
		batch_size=Config.BATCH_SIZE,
		shuffle=True,
		num_workers=Config.NUM_WORKERS
	)
	
	val_loader = torch.utils.data.DataLoader(
		val_data,
		batch_size=Config.BATCH_SIZE,
		shuffle=False,
		num_workers=Config.NUM_WORKERS
	)
	
	# Create and return client
	client = FlowerClient(partition_id, model, train_loader, val_loader)
	return client.to_client()