import torch
import flwr as fl
from dataset import SpineDataset
from model import CoordRegressionNetwork
from config import Config
from typing import List
from flwr.common import Context
import numpy as np
from training_utils import train, test
from dataset import create_train_val_test_split
from strategy import set_weights


# Create data splits once for all clients.
SPLITS = create_train_val_test_split(Config.DATA_DIR, debug=Config.DEBUG, train_ratio=0.8, val_ratio=0.1, seed=42)


def get_parameters(net) -> List[np.ndarray]:
	"""
	Convert the model's state dictionary to a list of NumPy arrays.

	Args:
		net: The PyTorch model.

	Returns:
		List of parameters as NumPy arrays.
	"""
	return [val.cpu().numpy() for _, val in net.state_dict().items()]


class FlowerClient(fl.client.NumPyClient):
	"""
	A Flower client that wraps a PyTorch model along with its data loaders.
	"""
	
	def __init__(self, pid, net, trainloader, valloader):
		"""
		Initialize the Flower client.

		Args:
			pid: Partition/client identifier.
			net: The PyTorch model.
			trainloader: DataLoader for training data.
			valloader: DataLoader for validation data.
		"""
		self.pid = pid
		self.net = net
		self.trainloader = trainloader
		self.valloader = valloader
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.net.to(self.device)
	
	def get_parameters(self, config):
		"""Return the model parameters."""
		return get_parameters(self.net)
	
	def fit(self, parameters, config):
		"""
		Train the model using parameters sent from the server.

		The configuration may include FedProx settings. If so, a deep copy of
		the global parameters is made and passed to the training function.

		Args:
			parameters: Global model parameters received from the server.
			config: Dictionary with training configuration, including keys:
				- "server_round": current round number.
				- "local_epochs": number of epochs to train locally.
				- "lr": learning rate.
				- "use_fedprox": flag indicating if FedProx is used.
				- "proximal_mu": FedProx hyperparameter.

		Returns:
			A tuple of (updated model parameters, number of training samples, empty dict).
		"""
		server_round = config["server_round"]
		local_epochs = config["local_epochs"]
		current_lr = config["lr"]
		print(f'Current LR is {current_lr}')
		print(f"[round {server_round}] fit, config: {config}")
		
		# Update the local model with the global parameters.
		set_weights(self.net, parameters)
		
		# If FedProx is used, copy the global parameters before training.
		global_params = None
		if config["use_fedprox"]:
			print('Using FedProx: copying global parameters.')
			global_params = [val.detach().clone() for val in self.net.parameters()]
		else:
			print(f'Not using FedProx. Global params is {global_params}')

		# Train the local model with or without the FedProx proximal term.
		train(self.net, self.trainloader, lr=current_lr, device=self.device,
		      epochs=local_epochs,
		      global_params=global_params,
		      proximal_mu=config["proximal_mu"])
		
		# Return updated parameters along with dataset size.
		return get_parameters(self.net), len(self.trainloader.dataset), {}
	
	def evaluate(self, parameters, config):
		"""
		Evaluate the model on the local validation dataset.

		Args:
			parameters: Global model parameters received from the server.
			config: Configuration dictionary (not used here).

		Returns:
			Tuple containing loss (as float), number of validation samples, and a dict with additional metrics.
		"""
		set_weights(self.net, parameters)
		loss = test(self.net, self.valloader, device=self.device)
		return float(loss), len(self.valloader.dataset), {'loss': loss}


def client_fn(context: Context) -> fl.client.Client:
	"""
	Create a Flower client representing a single hospital based on the context.

	The context provides a "partition-id" used to select the hospital-specific data.

	Args:
		context: Flower context containing node configuration.

	Returns:
		An instance of FlowerClient converted to a Flower client.
	"""
	# Extract partition ID and map to hospital code.
	partition_id = context.node_config["partition-id"]
	hospital_codes = {
		0: 'MAD', 1: 'BCN', 2: 'BOR', 3: 'IST'
	}
	hospital = hospital_codes[int(partition_id)]
	
	print(f"Client assigned to hospital: {hospital}")
	print(f"Partition id is {partition_id}")
	
	# Instantiate the model.
	model = CoordRegressionNetwork(
		arch=Config.ARCH,
		n_locations_global=28,
		n_ch=1,
		n_blocks=Config.N_BLOCKS
	)
	
	# Create hospital-specific datasets.
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
	
	# Wrap datasets in DataLoaders.
	train_loader = torch.utils.data.DataLoader(
		train_data, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS
	)
	val_loader = torch.utils.data.DataLoader(
		val_data, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
	)
	
	# Return the Flower client.
	return FlowerClient(partition_id, model, train_loader, val_loader).to_client()
