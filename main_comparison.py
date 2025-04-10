import torch
import json
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import SpineDataset, create_train_val_test_split
from model import CoordRegressionNetwork, LandmarkLoss, _init_weights
from training_utils import train_centralised_and_local
from config import Config
from fl_experiment import run_federated_experiment
from logger_utils import initialize_logger
from client_app import client_fn
from server_app import get_server_fn#server_fn
from fl_experiment import run_federated_experiment
from strategy import CustomFedAvg, CustomFedOpt, CustomFedProx

def train_all_approaches():
	"""Train and compare centralized, local, and federated learning approaches.

	    This function orchestrates the training of three different approaches:
	    1. Centralized: Single model trained on all data
	    2. Local: Independent models for each hospital
	    3. Federated: Collaborative learning across hospitals

	    The function handles data splitting, model creation, training, and logging
	    of results for each approach.
	"""
	# Set up device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	
	# Create dataset splits
	splits = create_train_val_test_split(Config.DATA_DIR, Config.DEBUG, train_ratio=0.8, val_ratio=0.1, seed=42)
	
	# Save splits for future reference
	# timestamp = Config.TIMESTAMP
	base_save_dir = Path(Config.SAVE_DIR)
	base_save_dir.mkdir(parents=True, exist_ok=True)
	# global logger_my
	# # Initialize logger
	# logger_my = initialize_logger(base_save_dir)
	# #
	# with open(base_save_dir / 'splits.json', 'w') as f:
	# 	json.dump(splits, f)
	#
	# def create_model():
	# 	return CoordRegressionNetwork(
	# 		arch=Config.ARCH,
	# 		n_locations_global=28,
	# 		n_ch=1,
	# 		n_blocks=Config.N_BLOCKS
	# 	)
	#
	# criterion = LandmarkLoss(
	# 	sigma_t=Config.SIGMA_T,
	# 	smoothness_weight=Config.SMOOTHNESS_WEIGHT
	# )
	#
	# # 1. Train Centralized Model
	# logger_my.log_training_start('centralized')
	# print("\nTraining Centralized Model...")
	# centralized_save_dir = Path(f"{Config.SAVE_DIR}/centralized_model")
	# centralized_save_dir.mkdir(parents=True, exist_ok=True)
	#
	# # Create complete training dataset
	# train_dataset = SpineDataset(Config.DATA_DIR, splits['train'], mode='train')
	# train_loader = DataLoader(
	# 	train_dataset,
	# 	batch_size=Config.BATCH_SIZE,
	# 	shuffle=True,
	# 	num_workers=Config.NUM_WORKERS,
	# 	pin_memory=True
	# )
	#
	# # Create pooled validation dataset
	# val_dataset = SpineDataset(Config.DATA_DIR, splits['val'], mode='val')
	# val_loader = DataLoader(
	# 	val_dataset,
	# 	batch_size=Config.BATCH_SIZE,
	# 	shuffle=False,
	# 	num_workers=Config.NUM_WORKERS,
	# 	pin_memory=True
	# )
	#
	# # Train centralized model
	# model = create_model()
	# _init_weights(model)
	# optimizer = torch.optim.Adam(
	# 	model.parameters(),
	# 	lr=Config.LEARNING_RATE
	# )
	#
	# train_centralised_and_local(
	# 	model=model,
	# 	train_loader=train_loader,
	# 	val_loader=val_loader,
	# 	criterion=criterion,
	# 	optimizer=optimizer,
	# 	num_epochs=Config.NUM_EPOCHS,
	# 	device=device,
	# 	save_dir=centralized_save_dir,
	# 	checkpoint_interval=Config.MODEL_CHECKPOINT_INTERVAL,
	# 	logger = logger_my
	# )
	#
	# # 2. Train Local Models (Independent)
	# print("\nTraining Local Models...")
	# local_save_dir = Path(f"{Config.SAVE_DIR}/local_models")
	# local_save_dir.mkdir(parents=True, exist_ok=True)
	#
	# for center in ['MAD', 'BCN', 'BOR', 'IST']:
	# 	logger_my.log_training_start('local', hospital=center)
	# 	print(f"\nTraining {center} Model...")
	# 	center_save_dir = local_save_dir/center
	#
	# 	# Create center-specific datasets
	# 	center_train_dataset = SpineDataset(
	# 		Config.DATA_DIR,
	# 		splits['by_hospital'][center]['train'],
	# 		mode='train'
	# 	)
	# 	center_val_dataset = SpineDataset(
	# 		Config.DATA_DIR,
	# 		splits['by_hospital'][center]['val'],
	# 		mode='val'
	# 	)
	#
	# 	# Create center-specific dataloaders
	# 	center_train_loader = DataLoader(
	# 		center_train_dataset,
	# 		batch_size=Config.BATCH_SIZE,
	# 		shuffle=True,
	# 		num_workers=Config.NUM_WORKERS,
	# 		pin_memory=True
	# 	)
	#
	# 	center_val_loader = DataLoader(
	# 		center_val_dataset,
	# 		batch_size=Config.BATCH_SIZE,
	# 		shuffle=False,
	# 		num_workers=Config.NUM_WORKERS,
	# 		pin_memory=True
	# 	)
	#
	# 	# Train center-specific model
	# 	model = create_model()
	# 	_init_weights(model)
	# 	optimizer = torch.optim.Adam(
	# 		model.parameters(),
	# 		lr=Config.LEARNING_RATE
	# 	)
	#
	# 	train_centralised_and_local(
	# 		model=model,
	# 		train_loader=center_train_loader,
	# 		val_loader=center_val_loader,
	# 		criterion=criterion,
	# 		optimizer=optimizer,
	# 		num_epochs=Config.NUM_EPOCHS,
	# 		device=device,
	# 		save_dir=center_save_dir,
	# 		checkpoint_interval=Config.MODEL_CHECKPOINT_INTERVAL,
	# 		model_type='local',  # Added parameter for logging
	# 		hospital=center,
	# 		logger = logger_my
	# 	)
	
	# 3. Train Federated Models
	strategy_map = {
		#"FedOpt": CustomFedOpt,
		"FedProx": CustomFedProx,
		#"FedAvg": CustomFedAvg,
		#"FedAdam": CustomFedAdam,
		
		
	}
	#logger_my.log_training_start('federated')
	print("\nTraining Federated Model...")
	# run_federated_experiment(client_fn=client_fn, server_fn=server_fn, device=device)
	for name, strat_class in strategy_map.items():
		print(f"Running federated experiment with {name}...")
		# Create a server_fn for the current strategy with a unique suffix
		server_fn = get_server_fn(strat_class, save_suffix=name, fl_approach=name)
		# Run the federated experiment; assuming run_federated_experiment doesn't change.
		run_federated_experiment(client_fn=client_fn, server_fn=server_fn, device=device)


if __name__ == '__main__':
	train_all_approaches()