"""
Configuration settings for spine landmark detection and federated learning experiments.
"""
from datetime import datetime
from pathlib import Path


class Config:
	"""
	Configuration class containing all settings for training and evaluation.

	Settings are organized into categories:
	- Data: Paths, dimensions, and dataset-related configs
	- Model: Architecture and model parameters
	- Training: Learning rates, epochs, batch sizes
	- Augmentation: Data augmentation parameters
	- Federated Learning: Parameters specific to federated experiments
	- Output: Save paths and logging settings
	"""
	
	# --- DATA SETTINGS ---
	# Path to dataset (configurable for different environments)
	DATA_DIR = Path('/cluster/work/bmdslab/ancina/ESSG/lumbar_dataset/')
	# Input image dimensions
	IMAGE_SIZE = (768, 768)
	# Use reduced dataset for debugging
	DEBUG = False
	
	# --- MODEL SETTINGS ---
	# Backbone architecture ('hg1', 'hg2', 'hg8')
	ARCH = 'hg2'
	# Number of residual blocks in the hourglass module
	N_BLOCKS = 2
	# Gaussian sigma for coordinate regression
	SIGMA_T = 1.0
	
	# --- TRAINING SETTINGS ---
	# Mini-batch size
	BATCH_SIZE = 4
	# Number of data loading worker processes (0 for debugging)
	NUM_WORKERS = 0
	# Base learning rate
	LEARNING_RATE = 1e-4
	# Number of epochs for centralized/local training
	NUM_EPOCHS = 100
	# Frequency of saving model checkpoints
	MODEL_CHECKPOINT_INTERVAL = 10
	
	# --- FEDERATED LEARNING SETTINGS ---
	# Number of communication rounds for FedAvg
	NUM_ROUNDS_FL = 30
	# Number of rounds for FedOpt
	NUM_ROUNDS_FL_OPT = 100
	# Number of local epochs per federated round for FedAvg
	LOCAL_EPOCHS = 10
	# Number of local epochs per federated round for FedProx
	LOCAL_EPOCHS_PROX = 8
	# Number of local epochs per federated round for FedOpt
	LOCAL_EPOCHS_OPT = 3
	
	# --- AUGMENTATION SETTINGS ---
	# Probability of applying brightness/contrast adjustments
	AUG_BRIGHTNESS_CONTRAST_PROB = 0.8
	# Probability of adding noise
	AUG_NOISE_PROB = 0.5
	# Maximum shift as proportion of image size
	AUG_SHIFT_LIMIT = 0.2
	# Maximum scale change
	AUG_SCALE_LIMIT = 0.2
	# Maximum rotation angle in degrees
	AUG_ROTATE_LIMIT = 15
	
	# --- OUTPUT SETTINGS ---
	# Timestamp for unique run identification
	TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	# Directory for saving results
	SAVE_DIR = Path(f'./results_{ARCH}_{TIMESTAMP}')