import flwr as fl
import torch


def run_federated_experiment(client_fn, server_fn, device):
    # Create the ClientApp
	client = fl.client.ClientApp(client_fn=client_fn)
    # Create the ServerApp
	server = fl.server.ServerApp(server_fn=server_fn)
	
	# Specify the resources each of your clients need
	# By default, each client will be allocated 1x CPU and 0x GPUs
	backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
	
	# When running on GPU, assign an entire GPU for each client
	if device == torch.device("cuda"):
		backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
	# Refer to our Flower framework documentation for more details about Flower simulations
	# and how to set up the `backend_config`
	
	# Run simulation
	fl.simulation.run_simulation(
		server_app=server,
		client_app=client,
		num_supernodes=4,
		backend_config=backend_config,
		verbose_logging = True
	)
	