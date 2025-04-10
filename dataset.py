"""
Dataset management for spine landmark detection project.

This module includes:
- Data loading and preprocessing for spine landmark detection
- Data augmentation configurations
- Data splitting utilities for experiments
"""
import os
import dsntnn
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config


def read_landmark_file(filename: str) -> Tuple[Dict[str, List[Tuple[float, float]]], np.ndarray]:
	"""
	Read a landmark annotation file and extract coordinates.

	Args:
		filename: Path to the landmark text file

	Returns:
		Tuple containing:
			- Dictionary mapping vertebrae labels to coordinate lists
			- Numpy array of all coordinates in sequential order
	"""
	coordinates = {}
	current_label = None
	all_coords = []
	
	with open(filename, 'r') as file:
		for line in file:
			line = line.strip()
			if not line:
				continue
			
			# Lines with commas are coordinate pairs
			if ',' in line:
				x, y = map(float, line.split(', '))
				all_coords.append([x, y])
				
				if current_label:
					if current_label not in coordinates:
						coordinates[current_label] = []
					coordinates[current_label].append((x, y))
			else:
				# Lines without commas are vertebrae labels
				current_label = line
	
	# Convert to numpy array
	coords_array = np.array(all_coords)
	
	return coordinates, coords_array


def create_train_val_test_split(
		root_dir: Union[str, Path],
		debug: bool = False,
		train_ratio: float = 0.8,
		val_ratio: float = 0.1,
		seed: int = 42
) -> Dict[str, Any]:
	"""
	Create train/val/test splits while maintaining hospital distribution.

	Args:
		root_dir: Path to the dataset root directory
		debug: If True, use a small subset of data for debugging
		train_ratio: Proportion of data for training
		val_ratio: Proportion of data for validation
		seed: Random seed for reproducibility

	Returns:
		Dictionary containing the splits by set and by hospital
	"""
	np.random.seed(seed)
	root_dir = Path(root_dir)
	
	# Get all lateral image files
	img_dir = root_dir / 'images'
	all_images = [os.path.splitext(f)[0] for f in os.listdir(img_dir)
	              if f.endswith('lat.jpg') or f.endswith('lat.png')]
	all_images = sorted(all_images)
	
	# Create DataFrame to help with organizing
	df = pd.DataFrame(index=all_images)
	
	# Filter to include only specific hospitals
	df = df[df.index.str.startswith(('MAD', 'BCN', 'BOR', 'IST', 'ZUR'))]
	
	if debug:
		df = df.sample(n=40, replace=False)
	
	# Extract patient and center identifiers
	df['patient_id'] = df.index.to_series().apply(lambda x: x.split('-')[0])
	df['center'] = df['patient_id'].str[:3]
	
	# Initialize splits dictionary
	splits = {
		'train': [], 'val': [], 'test': [],
		'by_hospital': {
			'MAD': {'train': [], 'val': [], 'test': []},
			'BCN': {'train': [], 'val': [], 'test': []},
			'BOR': {'train': [], 'val': [], 'test': []},
			'IST': {'train': [], 'val': [], 'test': []},
			'ZUR': {'train': [], 'val': [], 'test': []}
		}
	}
	
	# Split each hospital's data separately to maintain distribution
	for center in ['MAD', 'BCN', 'BOR', 'IST', 'ZUR']:
		center_df = df[df['center'] == center]
		center_patients = center_df['patient_id'].unique()
		
		# First split: train vs (val+test)
		train_patients, temp_patients = train_test_split(
			center_patients,
			train_size=train_ratio,
			random_state=seed,
			shuffle=True
		)
		
		# Second split: val vs test
		val_ratio_adjusted = val_ratio / (1 - train_ratio)
		val_patients, test_patients = train_test_split(
			temp_patients,
			train_size=val_ratio_adjusted,
			random_state=seed,
			shuffle=True
		)
		
		# Add to hospital-specific splits
		splits['by_hospital'][center]['train'].extend(
			center_df[center_df['patient_id'].isin(train_patients)].index.tolist()
		)
		splits['by_hospital'][center]['val'].extend(
			center_df[center_df['patient_id'].isin(val_patients)].index.tolist()
		)
		splits['by_hospital'][center]['test'].extend(
			center_df[center_df['patient_id'].isin(test_patients)].index.tolist()
		)
		
		# Add to combined splits
		splits['train'].extend(splits['by_hospital'][center]['train'])
		splits['val'].extend(splits['by_hospital'][center]['val'])
		splits['test'].extend(splits['by_hospital'][center]['test'])
	
	# Sort all lists for consistency
	for split in ['train', 'val', 'test']:
		splits[split].sort()
		for center in splits['by_hospital']:
			splits['by_hospital'][center][split].sort()
	
	# Print distribution statistics
	print("\nData distribution across splits:")
	for split in ['train', 'val', 'test']:
		print(f"\n{split.upper()} set:")
		total = len(splits[split])
		print(f"Total: {total}")
		for center in ['MAD', 'BCN', 'BOR', 'IST', 'ZUR']:
			count = len(splits['by_hospital'][center][split])
			print(f"{center}: {count} ({count / total * 100:.1f}%)")
	
	return splits


class SpineDataset(Dataset):
	"""
	Dataset for spine landmark detection.

	Loads X-ray images and landmark annotations, with optional augmentation
	for training mode.
	"""
	
	def __init__(self, root_dir: Union[str, Path], split_list: Optional[List[str]] = None, mode: str = 'train'):
		"""
		Initialize the spine dataset.

		Args:
			root_dir: Path to dataset root directory
			split_list: List of image IDs to include in this dataset split
			mode: 'train' for training mode with augmentation, or 'val'/'test' for evaluation
		"""
		self.root_dir = Path(root_dir)
		self.mode = mode
		
		# Setup paths
		self.img_dir = self.root_dir / 'images'
		self.ann_dir = self.root_dir / 'annotations'
		
		# Get image files
		if split_list is not None:
			self.img_files = [f for f in os.listdir(self.img_dir)
			                  if os.path.splitext(f)[0] in split_list and
			                  (f.endswith('.jpg') or f.endswith('.png'))]
		else:
			self.img_files = [f for f in os.listdir(self.img_dir)
			                  if f.endswith('.jpg') or f.endswith('.png')]
		
		self.img_files = sorted(self.img_files)
		
		# Define transformations based on mode
		self.transform = self._get_transforms(mode)
	
	def _get_transforms(self, mode: str) -> A.Compose:
		"""
		Get the appropriate image transformations based on mode.

		Args:
			mode: 'train' or 'val'/'test'

		Returns:
			Albumentations composition of transforms
		"""
		if mode == 'train':
			return A.Compose([
				A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
				
				# Geometric transformations
				A.SafeRotate((-10, 10), p=0.5),
				A.VerticalFlip(p=0.3),
				A.Affine(scale=(0.8, 1.2), p=0.3),
				A.Affine(translate_px=(-5, 5), p=0.3),
				
				# Intensity augmentations
				A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.3),
				A.InvertImg(p=0.3),
				
				# Dropout simulation for robustness
				A.Lambda(image=lambda img, **kwargs: A.CoarseDropout(
					num_holes_range=(10, 12),
					hole_height_range=(5, Config.IMAGE_SIZE[0] // 15),
					hole_width_range=(5, Config.IMAGE_SIZE[1] // 4),
					fill=255,
					p=0.5
				)(image=img)['image']),
				
				# Intensity/contrast variations
				A.OneOf([
					A.RandomBrightnessContrast(
						brightness_limit=(-0.7, 0.3),
						contrast_limit=(-0.3, 0.3),
						p=1.0
					),
					A.MultiplicativeNoise(
						multiplier=(0.4, 1.2),
						per_channel=False,
						elementwise=True,
						p=1.0
					),
					A.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), p=1.0)
				], p=0.4),
				
				# Simulate X-ray noise and blur
				A.OneOf([
					A.GaussianBlur(blur_limit=(3, 7), p=1.0),
					A.MotionBlur(blur_limit=(3, 7), p=1.0),
				], p=0.5),
				
				A.RandomGamma(gamma_limit=(70, 130), p=0.7),
				A.GaussNoise(std_range=(0.1, 0.2), p=0.3),
				
				ToTensorV2()
			], seed=42,
				keypoint_params=A.KeypointParams(
					format='xy',
					remove_invisible=False,
					check_each_transform=False
				))
		else:
			# Validation/test transforms - only resizing
			return A.Compose([
				A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
				ToTensorV2()
			],
				keypoint_params=A.KeypointParams(
					format='xy',
					remove_invisible=False,
					check_each_transform=False
				))
	
	def __len__(self) -> int:
		"""Return the number of samples in the dataset."""
		return len(self.img_files)
	
	def __getitem__(self, idx: int) -> Dict[str, Any]:
		"""
		Get a sample from the dataset.

		Args:
			idx: Index of the sample to fetch

		Returns:
			Dictionary containing the processed image, landmarks, and metadata
		"""
		# Load image
		img_name = self.img_files[idx]
		img_path = self.img_dir / img_name
		image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
		height, width = image.shape[:2]
		
		# Load landmarks
		ann_name = os.path.splitext(img_name)[0] + '.txt'
		ann_path = self.ann_dir / ann_name
		_, landmarks = read_landmark_file(str(ann_path))
		
		# Apply transformations
		keypoints = landmarks.tolist()
		transformed = self.transform(
			image=image,
			keypoints=keypoints
		)
		
		# Extract transformed data
		image = transformed['image'].float() / 255.0  # Normalize to [0, 1]
		keypoints = np.array(transformed['keypoints'])
		
		# Convert pixel coordinates to normalized coordinates [-1, 1]
		landmarks_norm = dsntnn.pixel_to_normalized_coordinates(
			keypoints, size=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])
		)
		
		return {
			'image': image,
			'landmarks': keypoints,
			'landmarks_norm': landmarks_norm,
			'image_path': str(img_path),
			'orig_size': (width, height)
		}


def create_data_loaders(
		dataset_dir: Union[str, Path],
		splits: Dict[str, List[str]],
		batch_size: int = 4,
		num_workers: int = 0,
		pin_memory: bool = True
) -> Dict[str, DataLoader]:
	"""
	Create PyTorch DataLoaders for train/val/test splits.

	Args:
		dataset_dir: Path to dataset root directory
		splits: Dictionary with train/val/test split image IDs
		batch_size: Batch size for DataLoader
		num_workers: Number of worker processes for data loading
		pin_memory: If True, pin memory for faster GPU transfer

	Returns:
		Dictionary of DataLoaders for each split
	"""
	# Create datasets
	train_dataset = SpineDataset(dataset_dir, splits['train'], mode='train')
	val_dataset = SpineDataset(dataset_dir, splits['val'], mode='val')
	test_dataset = SpineDataset(dataset_dir, splits['test'], mode='val')  # Use val mode for test (no augmentation)
	
	# Create dataloaders
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=pin_memory
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory
	)
	
	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory
	)
	
	return {
		'train': train_loader,
		'val': val_loader,
		'test': test_loader
	}


# For debugging/demonstration purposes
if __name__ == '__main__':
	# Create splits
	splits = create_train_val_test_split(Config.DATA_DIR, Config.DEBUG, train_ratio=0.8, val_ratio=0.1, seed=42)
	
	# Print hospital distribution
	for center in ('MAD', 'BCN', 'BOR', 'IST', 'ZUR'):
		count = sum(1 for ind in splits['train'] if ind.startswith(center))
		print(f'{center} has {count} samples in training set')
	
	# Create sample dataset and dataloader
	train_dataset = SpineDataset(Config.DATA_DIR, splits['train'], mode='train')
	train_loader = DataLoader(
		train_dataset,
		batch_size=Config.BATCH_SIZE,
		shuffle=True,
		num_workers=Config.NUM_WORKERS,
		pin_memory=True
	)
	
	# Visualize a few samples
	for batch in train_loader:
		imgs = batch['image']
		landmarks = batch['landmarks']
		print(f"Landmarks shape: {landmarks.shape}")
		
		# Show the first few images with landmarks
		for i in range(min(2, len(imgs))):
			plt.figure(figsize=(8, 8))
			plt.imshow(imgs[i].squeeze(), cmap='gray')
			plt.scatter(landmarks[i, :, 0], landmarks[i, :, 1], c='r', s=5)
			plt.title(f"Sample {i + 1}")
			plt.show()
			plt.close()
		
		break  # Just show one batch