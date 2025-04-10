"""
Federated learning strategies for spine landmark detection.

This module defines custom federated learning strategies that extend
the base strategies from Flower with additional functionality for
saving results and model checkpoints.
"""
import json
import os
from logging import INFO
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

import torch
import torch.nn as nn
from collections import OrderedDict

from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import Parameters, Scalar, FitRes, EvaluateRes, UserConfig
from flwr.server.strategy import FedAvg, FedOpt, FedProx
from flwr.server.client_proxy import ClientProxy

from model import CoordRegressionNetwork
from config import Config


def set_weights(net: nn.Module, parameters: List[np.ndarray]) -> None:
	"""
	Set model weights from a list of NumPy arrays.

	Args:
		net: PyTorch model
		parameters: List of NumPy arrays containing model parameters
	"""
	params_dict = zip(net.state_dict().keys(), parameters)
	state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
	net.load_state_dict(state_dict, strict=True)


def create_run_dir(config: UserConfig) -> Tuple[Path, str]:
	"""
	Create a directory for saving federated learning experiment results.

	Args:
		config: Configuration dictionary for the experiment

	Returns:
		Tuple of (directory path, run name)
	"""
	# Create run name with timestamp and optional suffix
	run_dir = f"experiment_{Config.TIMESTAMP}"
	run_suffix = config.get("save_suffix", "")
	
	# Create save path
	save_path = Path(f"{Config.SAVE_DIR}/federated_{run_suffix}")
	save_path.mkdir(parents=True, exist_ok=True)
	
	# Save configuration for reproducibility
	with open(save_path / "run_config.json", "w", encoding="utf-8") as fp:
		json.dump(config, fp, indent=4)
	
	return save_path, run_dir


class ResultSavingAndCheckpointingMixin:
	"""
	Mixin class that adds result saving and model checkpointing to federated strategies.

	This mixin provides functionality to:
	- Save experiment results to JSON
	- Track and save best performing models
	- Save regular checkpoints of the global model
	"""
	
	def __init__(self, run_config: UserConfig, *args, **kwargs):
		"""
		Initialize the mixin.

		Args:
			run_config: Configuration for the experiment
			*args, **kwargs: Arguments for the parent class
		"""
		# Call parent class constructor
		super().__init__(*args, **kwargs)
		
		# Create directory for saving results
		self.save_path, self.run_dir = create_run_dir(run_config)
		
		# Track best performance
		self.best_loss_so_far = float('inf')
		
		# Storage for results
		self.results = {}
	
	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
	) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
		"""
		Aggregate model updates and save global model checkpoint.

		Args:
			server_round: Current federated round
			results: List of client training results
			failures: List of client failures

		Returns:
			Tuple of (aggregated parameters, metrics dict)
		"""
		# Call parent class method to aggregate parameters
		parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
		
		# Save the global model checkpoint if parameters were successfully aggregated
		if parameters_aggregated is not None:
			self._save_model_checkpoint(parameters_aggregated, server_round)
		
		return parameters_aggregated, metrics_aggregated
	
	def _save_model_checkpoint(self, parameters: Parameters, server_round: int) -> None:
		"""
		Save a checkpoint of the global model.

		Args:
			parameters: Model parameters
			server_round: Current federated round
		"""
		# Convert parameters to NumPy arrays
		ndarrays = parameters_to_ndarrays(parameters)
		
		# Initialize model with the same architecture
		model = CoordRegressionNetwork(
			arch=Config.ARCH,
			n_locations_global=28,
			n_ch=1,
			n_blocks=Config.N_BLOCKS
		)
		
		# Set model weights from parameters
		set_weights(model, ndarrays)
		
		# Save model
		checkpoint_path = self.save_path / f"global_model_round_{server_round}.pth"
		torch.save(model.state_dict(), checkpoint_path)
		logger.log(INFO, f"Saved global model checkpoint for round {server_round}")
	
	def _store_results(self, tag: str, results_dict: Dict[str, Any]) -> None:
		"""
		Store results in a dictionary and save as JSON.

		Args:
			tag: Category of results (e.g., 'centralized_evaluate')
			results_dict: Dictionary of metrics/results
		"""
		# Add results to the appropriate category
		if tag in self.results:
			self.results[tag].append(results_dict)
		else:
			self.results[tag] = [results_dict]
		
		# Save updated results to JSON
		with open(self.save_path / "results.json", "w", encoding="utf-8") as fp:
			json.dump(self.results, fp, indent=4)
	
	def _update_best_loss(self, round: int, loss: float, parameters: Parameters) -> None:
		"""
		Update the best model if current loss is lower.

		Args:
			round: Current federated round
			loss: Loss value
			parameters: Model parameters
		"""
		if loss < self.best_loss_so_far:
			self.best_loss_so_far = loss
			logger.log(INFO, f"💡 New best global model found for round {round}: {loss:.6f}")
			
			# Save best model
			ndarrays = parameters_to_ndarrays(parameters)
			model = CoordRegressionNetwork(
				arch=Config.ARCH,
				n_locations_global=28,
				n_ch=1,
				n_blocks=Config.N_BLOCKS
			)
			set_weights(model, ndarrays)
			
			best_model_path = self.save_path / "best_model.pth"
			torch.save(model.state_dict(), best_model_path)
	
	def store_results_and_log(self, server_round: int, tag: str, results_dict: Dict[str, Any]) -> None:
		"""
		Store results and log them.

		Args:
			server_round: Current federated round
			tag: Category of results
			results_dict: Dictionary of metrics/results
		"""
		# Add round number to results
		results_with_round = {"round": server_round, **results_dict}
		
		# Store and log
		self._store_results(tag=tag, results_dict=results_with_round)
		logger.log(INFO, f"Round {server_round} {tag}: {results_dict}")
	
	def evaluate(self, server_round: int, parameters: Parameters) -> Tuple[float, Dict[str, Scalar]]:
		"""
		Run centralized evaluation and save results.

		Args:
			server_round: Current federated round
			parameters: Global model parameters

		Returns:
			Tuple of (loss, metrics dict)
		"""
		# Call parent class method for evaluation
		loss, metrics = super().evaluate(server_round, parameters)
		
		# Check if this is the best model so far
		if "centralized_loss" in metrics:
			self._update_best_loss(server_round, metrics["centralized_loss"], parameters)
		
		# Store and log results
		self.store_results_and_log(
			server_round=server_round,
			tag="centralized_evaluate",
			results_dict={"centralized_loss": loss, **metrics}
		)
		
		return loss, metrics
	
	def aggregate_evaluate(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, EvaluateRes]],
			failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
	) -> Tuple[Optional[float], Dict[str, Scalar]]:
		"""
		Aggregate client evaluation results and save them.

		Args:
			server_round: Current federated round
			results: List of client evaluation results
			failures: List of client failures

		Returns:
			Tuple of (aggregated loss, metrics dict)
		"""
		# Call parent class method to aggregate results
		loss, metrics = super().aggregate_evaluate(server_round, results, failures)
		
		# Store and log results
		if loss is not None:
			self.store_results_and_log(
				server_round=server_round,
				tag="federated_evaluate",
				results_dict={"federated_evaluate_loss": loss, **metrics}
			)
		
		return loss, metrics


# Define custom federated learning strategies using the mixin
class CustomFedAvg(ResultSavingAndCheckpointingMixin, FedAvg):
	"""
	Custom Federated Averaging strategy with result saving and checkpointing.

	Extends the standard FedAvg algorithm from Flower with additional
	functionality for tracking and saving results.
	"""
	pass


class CustomFedOpt(ResultSavingAndCheckpointingMixin, FedOpt):
	"""
	Custom Federated Optimization strategy with result saving and checkpointing.

	Extends the standard FedOpt algorithm from Flower with additional
	functionality for tracking and saving results.
	"""
	pass


class CustomFedProx(ResultSavingAndCheckpointingMixin, FedProx):
	"""
	Custom FedProx strategy with result saving and checkpointing.

	Extends the standard FedProx algorithm from Flower with additional
	functionality for tracking and saving results.
	"""
	pass