"""
Federated learning server implementation for spine landmark detection.

This module defines the server-side components for federated learning, including
evaluation functions and server configuration.
"""
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Callable, Any

import flwr as fl
from flwr.common import ndarrays_to_parameters, Scalar, NDArrays

from model import CoordRegressionNetwork
from config import Config
from dataset import SpineDataset, create_train_val_test_split
from client_app import get_parameters
from training_utils import test
from strategy import set_weights

# Create dataset splits once
SPLITS = create_train_val_test_split(Config.DATA_DIR, debug=Config.DEBUG, train_ratio=0.8, val_ratio=0.1, seed=42)


def get_evaluate_fn(
		testloader: DataLoader,
		device: torch.device
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]]]:
	"""
	Create a function for centralized evaluation of the global model.

	Args:
		testloader: DataLoader for validation/test data
		device: Device to run evaluation on

	Returns:
		Function that evaluates the global model and returns loss/metrics
	"""
	
	def evaluate(
			server_round: int,
			parameters_ndarrays: NDArrays,
			config: Dict[str, Scalar]
	) -> Tuple[float, Dict[str, Scalar]]:
		"""
		Evaluate the global model on a validation dataset.

		Args:
			server_round: Current federated round
			parameters_ndarrays: Global model parameters as NumPy arrays
			config: Configuration dictionary

		Returns:
			Tuple of (loss, metrics dictionary)
		"""
		# Initialize the model with the same architecture
		model = CoordRegressionNetwork(
			arch=Config.ARCH,
			n_locations_global=28,
			n_ch=1,
			n_blocks=Config.N_BLOCKS
		)
		
		# Set model weights from parameters
		set_weights(model, parameters_ndarrays)
		model.to(device)
		
		# Evaluate the model
		loss = test(model, testloader, device)
		
		return loss, {'centralized_loss': loss}
	
	return evaluate


def weighted_average(
		metrics: List[Tuple[int, Dict[str, float]]]
) -> Dict[str, float]:
	"""
	Aggregate evaluation metrics from clients using weighted average.

	Args:
		metrics: List of tuples (num_examples, metrics_dict)

	Returns:
		Dictionary with aggregated metrics
	"""
	# Extract values for weighted averaging
	losses = [num_examples * m["loss"] for num_examples, m in metrics]
	examples = [num_examples for num_examples, _ in metrics]
	
	# Calculate weighted average
	avg_loss = sum(losses) / sum(examples) if examples else 0.0
	
	return {"federated_evaluate_loss": avg_loss}


def make_fit_config(run_config: Dict[str, Any]) -> Callable[[int], Dict[str, Scalar]]:
	"""
	Create a function that returns training configuration for each round.

	Args:
		run_config: Experiment-level configuration

	Returns:
		Function that generates per-round configuration
	"""
	
	def fit_config(server_round: int) -> Dict[str, Scalar]:
		"""
		Generate configuration for client training in the given round.

		Args:
			server_round: Current federated round

		Returns:
			Configuration dictionary for client training
		"""
		# Common configuration for all rounds
		config = {"server_round": server_round}
		
		# Different settings based on federated approach
		fl_approach = run_config.get("fl_approach", "FedAvg")
		
		if fl_approach == "FedAvg":
			# Possibly reduce local epochs in later rounds
			config["local_epochs"] = (
				Config.LOCAL_EPOCHS if server_round < 30
				else int(Config.LOCAL_EPOCHS / 2)
			)
			
			# Learning rate decay
			config["lr"] = (
				Config.LEARNING_RATE if server_round <= 20
				else Config.LEARNING_RATE / (server_round ** 0.5)
			)
		else:
			# Base settings for other approaches (FedOpt, FedProx)
			base_lr = 1e-4
			config["local_epochs"] = 3 if fl_approach == "FedOpt" else 8
			config["lr"] = base_lr / (server_round ** 0.5) if server_round > 1 else base_lr
		
		# FedProx settings
		config["use_fedprox"] = run_config.get("use_fedprox", False)
		config["proximal_mu"] = run_config.get("proximal_mu", 0.0)
		
		return config
	
	return fit_config


def get_server_fn(
		strategy_class: Any,
		save_suffix: str,
		fl_approach: str
) -> Callable[[Dict[str, Any]], fl.server.ServerAppComponents]:
	"""
	Create a server function that configures the FL server for a specific strategy.

	Args:
		strategy_class: Class for the federated learning strategy
		save_suffix: Suffix for saving results
		fl_approach: String identifier for the federated approach

	Returns:
		Function that creates server components
	"""
	
	def server_fn(context: Dict[str, Any]) -> fl.server.ServerAppComponents:
		"""
		Configure the server components for federated learning.

		Args:
			context: Context with server configuration

		Returns:
			ServerAppComponents with strategy and configuration
		"""
		# Get and update experiment-level configuration
		run_config = context.run_config.copy()
		run_config["save_suffix"] = save_suffix
		run_config["fl_approach"] = fl_approach
		
		# Configure FedProx settings
		if fl_approach == 'FedProx':
			run_config["use_fedprox"] = True
			run_config["proximal_mu"] = 0.1
		else:
			run_config["use_fedprox"] = False
			run_config["proximal_mu"] = 0.0
		
		# Create fit config function for the specific approach
		fit_config_fn = make_fit_config(run_config)
		
		# Initialize global model
		net = CoordRegressionNetwork(
			arch=Config.ARCH,
			n_locations_global=28,
			n_ch=1,
			n_blocks=Config.N_BLOCKS
		)
		
		# Get initial parameters
		ndarrays = get_parameters(net)
		parameters = ndarrays_to_parameters(ndarrays)
		
		# Create validation data loader for evaluation
		val_data = SpineDataset(
			Config.DATA_DIR,
			SPLITS['val'],
			mode='val'
		)
		val_loader = torch.utils.data.DataLoader(
			val_data,
			batch_size=Config.BATCH_SIZE,
			shuffle=False,
			num_workers=Config.NUM_WORKERS
		)
		
		# Determine device for evaluation
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# Create evaluation function
		evaluate_fn = get_evaluate_fn(testloader=val_loader, device=device)
		
		# Configure strategy based on approach
		strategy_params = {
			"run_config": run_config,
			"fraction_fit": 1.0,
			"fraction_evaluate": 1.0,
			"min_fit_clients": 4,
			"min_evaluate_clients": 4,
			"min_available_clients": 4,
			"on_fit_config_fn": fit_config_fn,
			"initial_parameters": parameters,
			"evaluate_fn": evaluate_fn,
			"evaluate_metrics_aggregation_fn": weighted_average,
		}
		
		# Add strategy-specific parameters
		if fl_approach == 'FedProx':
			strategy_params["proximal_mu"] = 0.1
		elif fl_approach == 'FedOpt':
			strategy_params["eta"] = 1e-4
			strategy_params["eta_l"] = 1e-4
			strategy_params["tau"] = 1e-9
		
		# Create the strategy
		strategy = strategy_class(**strategy_params)
		
		# Set number of rounds based on approach
		if fl_approach == "FedAvg":
			num_rounds = Config.NUM_ROUNDS_FL
		else:
			num_rounds = Config.NUM_ROUNDS_FL_OPT
			run_config["num_rounds_fedopt"] = num_rounds
		
		# Create server configuration
		config_server = fl.server.ServerConfig(num_rounds=num_rounds)
		
		return fl.server.ServerAppComponents(strategy=strategy, config=config_server)
	
	return server_fn