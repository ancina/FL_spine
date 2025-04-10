import json
from logging import INFO
from config import Config
import torch
from model import CoordRegressionNetwork
from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg, FedOpt, FedProx
from pathlib import Path
import numpy as np
from collections import OrderedDict
import os


def set_weights(net, parameters):
	"""
	Copy parameters (as NumPy arrays) onto the model.

	Args:
		net: The PyTorch model.
		parameters: List of NumPy arrays.
	"""
	params_dict = zip(net.state_dict().keys(), parameters)
	state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
	net.load_state_dict(state_dict, strict=True)


def create_run_dir(config: UserConfig):
	"""
	Create a directory to save results from this run.

	The directory name incorporates a timestamp and optional suffix from the config.

	Args:
		config: Experiment-level configuration.

	Returns:
		A tuple (save_path, run_dir) where save_path is the directory and run_dir is its name.
	"""
	run_dir = f"experiment_{Config.TIMESTAMP}"
	run_suffix = config.get("save_suffix", "")
	# Create a save path that includes the suffix.
	save_path = Path(f"{Config.SAVE_DIR}/federated_{run_suffix}")
	save_path.mkdir(parents=True, exist_ok=False)
	
	# Save the run configuration to a JSON file.
	with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
		json.dump(config, fp)
	
	return save_path, run_dir

# Create a mixin class that contains all the common functionality
class ResultSavingAndCheckpointingMixin:
    """
    Mixin class that adds result saving and checkpointing functionality to federated learning strategies.
    """
    
    def __init__(self, run_config: UserConfig, *args, **kwargs):
        # Call the parent class's __init__
        super().__init__(*args, **kwargs)
        # Create a directory to store results
        self.save_path, self.run_dir = create_run_dir(run_config)
        # Track best loss so far
        self.best_loss_so_far = np.inf
        # Dictionary to store results
        self.results = {}
    
    def aggregate_fit(self, server_round: int, results, failures):
        """
        Aggregate model updates, save the global model checkpoint, and log metrics.
        """
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        # Save the global model checkpoint
        ndarrays = parameters_to_ndarrays(parameters_aggregated)
        model = CoordRegressionNetwork(
            arch=Config.ARCH,
            n_locations_global=28,
            n_ch=1,
            n_blocks=Config.N_BLOCKS
        )
        set_weights(model, ndarrays)
        torch.save(model.state_dict(), os.path.join(self.save_path, f"global_model_round_{server_round}"))
        return parameters_aggregated, metrics_aggregated
    
    def _store_results(self, tag: str, results_dict):
        """
        Store results in a dictionary and save as JSON.
        
        Args:
            tag: Label for the result (e.g., "centralized_evaluate").
            results_dict: Dictionary of metrics/results.
        """
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)
    
    def _update_best_loss(self, round, loss, parameters):
        """
        Update the best loss if the current loss is lower, and save the model checkpoint.
        
        Args:
            round: Current federated round.
            loss: Loss value.
            parameters: Model parameters.
        """
        if loss < self.best_loss_so_far:
            self.best_loss_so_far = loss
            logger.log(INFO, f"💡 New best global model found for round {round}: %f", loss)
            ndarrays = parameters_to_ndarrays(parameters)
            model = CoordRegressionNetwork(
                arch=Config.ARCH,
                n_locations_global=28,
                n_ch=1,
                n_blocks=Config.N_BLOCKS
            )
            set_weights(model, ndarrays)
            file_name = f"best_model.pth"
            torch.save(model.state_dict(), self.save_path / file_name)
    
    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """
        Helper method to store and log results.
        
        Args:
            server_round: Current federated round.
            tag: Label for the results.
            results_dict: Dictionary containing the results.
        """
        self._store_results(tag=tag, results_dict={"round": server_round, **results_dict})
    
    def evaluate(self, server_round, parameters):
        """
        Run centralized evaluation and save the model if a new best is found.
        """
        loss, metrics = super().evaluate(server_round, parameters)
        self._update_best_loss(server_round, metrics["centralized_loss"], parameters)
        self.store_results_and_log(
            server_round=server_round, 
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics}
        )
        return loss, metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggregate evaluation metrics from clients.
        """
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self.store_results_and_log(
            server_round=server_round, 
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics}
        )
        return loss, metrics


# Now create the custom strategy classes using multiple inheritance
# The mixin class comes first in the inheritance order to ensure its __init__ is called first
class CustomFedAvg(ResultSavingAndCheckpointingMixin, FedAvg):
    """
    Custom FedAvg strategy that adds result saving and checkpointing functionality.
    """
    pass


class CustomFedOpt(ResultSavingAndCheckpointingMixin, FedOpt):
    """
    Custom FedOpt strategy that adds result saving and checkpointing functionality.
    """
    pass


class CustomFedProx(ResultSavingAndCheckpointingMixin, FedProx):
    """
    Custom FedProx strategy that adds additional functionality for saving results and checkpointing.
    """
    pass

# class CustomFedAdam(ResultSavingAndCheckpointingMixin, FedAdam):
#     """
#     Custom FedAdam strategy with additional functionality for saving results and checkpoints.
#     """
#     pass
