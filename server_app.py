from flwr.common import ndarrays_to_parameters
from model import CoordRegressionNetwork, _init_weights
from config import Config
from client_app import get_parameters
import flwr as fl
import torch
from dataset import create_train_val_test_split
from strategy import set_weights
from training_utils import test
from dataset import SpineDataset

# Define number of federated rounds and create dataset splits.
ROUNDS = Config.NUM_ROUNDS_FL
SPLITS = create_train_val_test_split(Config.DATA_DIR, debug=Config.DEBUG, train_ratio=0.8, val_ratio=0.1, seed=42)


def get_evaluate_fn(testloader, device):
	"""
	Return a callback function for evaluating the global model on a test set.

	Args:
		testloader: DataLoader for the validation/test set.
		device: The device to run evaluation on.

	Returns:
		A function that evaluates the global model and returns loss and metrics.
	"""
	
	def evaluate(server_round, parameters_ndarrays, config):
		"""
		Evaluate the global model for the given round.

		Args:
			server_round: The current federated round.
			parameters_ndarrays: Global model parameters.
			config: Configuration dictionary (unused).

		Returns:
			Tuple of loss and a dictionary with additional metrics.
		"""
		# Instantiate and set up the model.
		model = CoordRegressionNetwork(arch=Config.ARCH,
		                               n_locations_global=28,
		                               n_ch=1,
		                               n_blocks=Config.N_BLOCKS).to(device)
		set_weights(model, parameters_ndarrays)
		model.to(device)
		loss = test(model, testloader, device)
		return loss, {'centralized_loss': loss}
	
	return evaluate

def make_fit_config(run_config):
    """
    Create a fit_config function that returns a configuration dictionary for each round,
    incorporating values from the provided run_config.
    """
    def fit_config(server_round: int):
        config = {"server_round": server_round}
        # Use different settings depending on the federated approach.
        if run_config["fl_approach"] == "FedAvg":
            # Retrieve the total rounds for FedOpt from run_config
            #num_rounds = run_config["num_rounds_fedopt"]
            config["local_epochs"] = Config.LOCAL_EPOCHS if server_round < 30 else int(Config.LOCAL_EPOCHS / 2)
            config["lr"] = Config.LEARNING_RATE if server_round <= 20 else Config.LEARNING_RATE / (server_round ** 0.5)
            
        else:
	        base_lr = 1e-4
	        config["local_epochs"] = 3 if run_config["fl_approach"] == "FedOpt" else 8
	        config["lr"] = base_lr / (server_round ** 0.5) if server_round > 1 else base_lr
         

        # Propagate FedProx settings from run_config.
        config["use_fedprox"] = run_config["use_fedprox"]
        config["proximal_mu"] = run_config["proximal_mu"]
        return config
    return fit_config


def weighted_average(metrics):
	"""
	Aggregate client metrics using a weighted average.

	Args:
		metrics: List of tuples, each containing the number of examples and a dict with 'loss'.

	Returns:
		A dictionary with the aggregated 'federated_evaluate_loss'.
	"""
	losses = [num_examples * m["loss"] for num_examples, m in metrics]
	examples = [num_examples for num_examples, _ in metrics]
	return {"federated_evaluate_loss": sum(losses) / sum(examples)}


def get_server_fn(strategy_class, save_suffix, fl_approach):
	"""
	Create a server function (to be used in run_federated_experiment) based on the provided strategy.

	This function sets experiment-level configuration (including FedProx settings) and returns a
	server function that instantiates the chosen strategy.

	Args:
		strategy_class: The custom federated strategy class (e.g., CustomFedAvg, CustomFedProx).
		save_suffix: Unique suffix for saving results.
		fl_approach: A string representing the federated approach (e.g., 'FedProx').

	Returns:
		A server function that returns ServerAppComponents.
	"""
	
	def server_fn(context):
		# Copy and update the experiment-level configuration.
		run_config = context.run_config.copy()
		run_config["save_suffix"] = save_suffix
		run_config["fl_approach"] = fl_approach
		# Set FedProx flags based on the chosen approach.
		if fl_approach == 'FedProx':
			run_config["use_fedprox"] = True
			run_config["proximal_mu"] = 0.1  # Adjust as needed.
		else:
			run_config["use_fedprox"] = False
			run_config["proximal_mu"] = 0.0
		
		# Create a fit_config function that incorporates run_config values.
		fit_config_fn = make_fit_config(run_config)
		
		# Instantiate the global model and prepare initial parameters.
		net = CoordRegressionNetwork(
			arch=Config.ARCH,
			n_locations_global=28,
			n_ch=1,
			n_blocks=Config.N_BLOCKS
		)
		_init_weights(net)
		ndarrays = get_parameters(net)
		parameters = ndarrays_to_parameters(ndarrays)
		
		# Prepare validation data.
		val_data = SpineDataset(
			Config.DATA_DIR,
			SPLITS['val'],
			mode='val'
		)
		val_loader = torch.utils.data.DataLoader(
			val_data, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
		)
		
		# Instantiate the chosen federated strategy.
		if fl_approach == 'FedProx':
			strategy = strategy_class(
				run_config=run_config,
				fraction_fit=1.0,
				fraction_evaluate=1.0,
				min_fit_clients=4,
				min_evaluate_clients=4,
				min_available_clients=4,
				on_fit_config_fn=fit_config_fn,
				initial_parameters=parameters,
				evaluate_fn=get_evaluate_fn(testloader = val_loader,
				                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")),
				evaluate_metrics_aggregation_fn=weighted_average,
				proximal_mu=0.1  # Additional parameter for FedProx.
			)
		elif fl_approach == 'FedAvg':
			strategy = strategy_class(
				run_config=run_config,
				fraction_fit=1.0,
				fraction_evaluate=1.0,
				min_fit_clients=4,
				min_evaluate_clients=4,
				min_available_clients=4,
				on_fit_config_fn=fit_config_fn,
				initial_parameters=parameters,
				evaluate_fn=get_evaluate_fn(testloader = val_loader,
											device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
				evaluate_metrics_aggregation_fn=weighted_average
			)
		else:
			strategy = strategy_class(
				run_config=run_config,
				fraction_fit=1.0,
				fraction_evaluate=1.0,
				min_fit_clients=4,
				min_evaluate_clients=4,
				min_available_clients=4,
				on_fit_config_fn=fit_config_fn,
				initial_parameters=parameters,
				evaluate_fn=get_evaluate_fn(testloader=val_loader,
				                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
				evaluate_metrics_aggregation_fn=weighted_average,
				eta = 1e-4,
				eta_l = 1e-4,
				tau = 1e-9
			)
		# Set the total number of FL rounds based on the approach.
		if run_config["fl_approach"] == "FedAvg":
			num_rounds = Config.NUM_ROUNDS_FL
		else:
			num_rounds = Config.NUM_ROUNDS_FL_OPT
			run_config["num_rounds_fedopt"] = num_rounds
			
		
		# Create server configuration.
		config_server = fl.server.ServerConfig(num_rounds=num_rounds)
		return fl.server.ServerAppComponents(strategy=strategy, config=config_server)
	
	return server_fn


# def server_fn(context: Context) -> fl.server.ServerAppComponents:
#     """Construct components that set the ServerApp behaviour.
#
#     You can use the settings in `context.run_config` to parameterize the
#     construction of all elements (e.g the strategy or the number of rounds)
#     wrapped in the returned ServerAppComponents object.
#     """
#
#     net = CoordRegressionNetwork(
# 			arch=Config.ARCH,
# 			n_locations_global=28,
# 			n_ch=1,
# 			n_blocks=Config.N_BLOCKS
# 		)
#
#     _init_weights(net)
#
#     ndarrays = get_parameters(net)
#
#     parameters = ndarrays_to_parameters(ndarrays)
#
#     val_data = SpineDataset(
# 	    Config.DATA_DIR,
# 	    SPLITS['val'],
# 	    mode='val'
#     )
#
#     val_loader = torch.utils.data.DataLoader(
# 	    val_data, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
#     )
#
#     # Define strategy
#     strategy = CustomFedAvg(
# 	    run_config=context.run_config,
# 	    fraction_fit=1.0,
# 	    fraction_evaluate=1.0,
# 	    min_fit_clients=2,
# 	    min_evaluate_clients=2,
# 	    min_available_clients=4,
# 	    on_fit_config_fn=fit_config,
# 	    initial_parameters=parameters,
#         evaluate_fn=get_evaluate_fn(val_loader,
#         device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
#         evaluate_metrics_aggregation_fn=weighted_average
# 	    #evaluate_fn=evaluate_centr
# 	    #logger = my_logger
# 	    # evaluate_metrics_aggregation_fn=weighted_average,
#     )
#
#     # Configure the server for N rounds of training
#     config = fl.server.ServerConfig(num_rounds=ROUNDS)
#
#     return fl.server.ServerAppComponents(strategy=strategy, config=config)
