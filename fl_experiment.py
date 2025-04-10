"""
Federated learning experiment runner for spine landmark detection.

This module provides functionality to run federated learning simulations
using Flower.
"""
import torch

import flwr as fl
from flwr.client import ClientApp
from flwr.server import ServerApp


def run_federated_experiment(
		client_fn,
		server_fn,
		device: torch.device,
		num_clients: int = 5,
		verbose: bool = True
) -> None:
	"""
	Run a federated learning experiment simulation.

	Args:
		client_fn: Function that creates Flower clients
		server_fn: Function that configures the Flower server
		device: Computation device (CPU or GPU)
		num_clients: Number of clients/hospitals to simulate
		verbose: Whether to enable verbose logging
	"""
	print("\n" + "=" * 50)
	print(f"Starting federated learning experiment with {num_clients} clients")
	print(f"Using device: {device}")
	print("=" * 50 + "\n")
	
	# Create client and server applications
	client = ClientApp(client_fn=client_fn)
	server = ServerApp(server_fn=server_fn)
	
	# Configure resource allocation
	if device.type == "cuda":
		# When using GPU, assign GPU resources to clients
		backend_config = {
			"client_resources": {
				"num_cpus": 1,
				"num_gpus": 1.0
			}
		}
	else:
		# CPU-only configuration
		backend_config = {
			"client_resources": {
				"num_cpus": 1,
				"num_gpus": 0.0
			}
		}
		print("Running on CPU")
	
	# Run the federated learning simulation
	fl.simulation.run_simulation(
		server_app=server,
		client_app=client,
		num_supernodes=num_clients,
		backend_config=backend_config,
		verbose_logging=verbose
	)
	
	print("\n" + "=" * 50)
	print("Federated learning experiment completed")
	print("=" * 50 + "\n")