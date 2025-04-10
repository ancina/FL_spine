import os
import dsntnn
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def create_train_val_test_split(root_dir, debug, train_ratio=0.8, val_ratio=0.1, seed=42):
	"""
	Create train/val/test splits while maintaining hospital distribution.
	"""
	np.random.seed(seed)
	
	img_dir = os.path.join(root_dir, 'images')
	all_images = [os.path.splitext(f)[0] for f in os.listdir(img_dir)
	              if f.endswith('lat.jpg') or f.endswith('lat.png')]
	all_images = sorted(all_images)
	
	df = pd.DataFrame(index=all_images)
	# filter patients of the centers
	df = df[df.index.str.startswith(('MAD', 'BCN', 'BOR', 'IST', 'ZUR'))]
	
	if debug:
		df = df.sample(n=100, replace=False)
	
	# Extract patient and center identifiers
	df['patient_id'] = df.index.to_series().apply(lambda x: x.split('-')[0])
	df['center'] = df['patient_id'].str[:3]
	
	n_patients = np.unique(df['patient_id'])
	
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
	
	# Split each hospital's data
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
		
		# Add to splits
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

def read_txt(filename):
	"""
	Reads a file containing vertebrae coordinates and returns both a dictionary
	of coordinates organized by vertebrae labels and a numpy array of all coordinates.

	Args:
		filename (str): Path to the input file

	Returns:
		tuple: (dict, numpy.ndarray) where:
			- dict: Dictionary with vertebrae labels as keys and lists of (x,y) coordinates as values
			- numpy.ndarray: 28x2 array containing all x,y coordinates in order
	"""
	coordinates = {}
	current_label = None
	all_coords = []  # List to store all coordinates in order
	
	with open(filename, 'r') as file:
		for line in file:
			# Strip whitespace and skip empty lines
			line = line.strip()
			if not line:
				continue
			
			# If line contains a comma, it's a coordinate pair
			if ',' in line:
				x, y = map(float, line.split(', '))
				all_coords.append([x, y])  # Add to the sequential list
				
				if current_label:
					if current_label not in coordinates:
						coordinates[current_label] = []
					coordinates[current_label].append((x, y))
			else:
				# Line without comma is a vertebrae label
				current_label = line
	
	# Convert the list of coordinates to a numpy array
	coords_array = np.array(all_coords)
	
	return coordinates, coords_array

class SpineDataset(Dataset):
	"""Dataset for spine landmark detection."""
	
	def __init__(self, root_dir, split_list=None, mode='train'):
		self.root_dir = root_dir
		self.mode = mode
		
		# Setup paths
		self.img_dir = os.path.join(root_dir, 'images')
		self.ann_dir = os.path.join(root_dir, 'annotations')
		
		# Get image files
		if split_list is not None:
			self.img_files = [f for f in os.listdir(self.img_dir)
			                  if os.path.splitext(f)[0] in split_list and (f.endswith('.jpg') or f.endswith('.png'))]
		else:
			self.img_files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')]
		
		self.img_files = sorted(self.img_files)
		
		# Define augmentations
		if mode == 'train':
			self.transform = A.Compose([
				A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
				
				A.SafeRotate((-10, 10), p=0.5),
				# A.HorizontalFlip(p = 0.3),
				A.VerticalFlip(p = 0.3),
				A.Affine(scale=(0.8, 1.2), p=0.3),
				A.Affine(translate_px=(-5, 5), p=0.3),
				
				# Increase intensity augmentations
				A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.3),
				
				# New inversion augmentation
				A.InvertImg(p=0.3),
				
				# More aggressive CoarseDropout
				A.Lambda(image=lambda img, **kwargs: A.CoarseDropout(
					num_holes_range = (10, 12),
					hole_height_range = (5, Config.IMAGE_SIZE[0] // 15),
					hole_width_range=(5, Config.IMAGE_SIZE[1] // 4),
					# max_holes = 12,
					# max_height=Config.IMAGE_SIZE[0] // 15,  # Slightly larger
					# max_width=Config.IMAGE_SIZE[1] // 4,
					# min_holes=5,
					# min_height=5,
					# min_width=20,
					fill=255,
					p=0.5
				)(image=img)['image']),
				
				# More diverse brightness/contrast changes
				A.OneOf([
					A.RandomBrightnessContrast(
						brightness_limit=(-0.7, 0.3),  # Allow some brightening too
						contrast_limit=(-0.3, 0.3),  # Add contrast variation
						p=1.0
					),
					A.MultiplicativeNoise(
						multiplier=(0.4, 1.2),  # Wider range
						per_channel=False,
						elementwise=True,
						p=1.0
					),
					A.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), p=1.0)
				], p=0.4),
				
				# Add more realistic X-ray specific augmentations
				A.OneOf([
					A.GaussianBlur(blur_limit=(3, 7), p=1.0),
					A.MotionBlur(blur_limit=(3, 7), p=1.0),
				], p=0.5),
				
				A.RandomGamma(gamma_limit=(70, 130), p=0.7),  # Wider gamma range
				A.GaussNoise(std_range=(0.1, 0.2), p=0.3),
				
				ToTensorV2()
			], seed=42,
				keypoint_params=A.KeypointParams(
					format='xy',
					remove_invisible=False,
					check_each_transform=False
				))
		
		else:
			self.transform = A.Compose([
				A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
				# Increase intensity augmentations
				# A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=1.0),  # Increased probability and range
				ToTensorV2()
			], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, check_each_transform=False))
	
	def __len__(self):
		return len(self.img_files)
	
	def __getitem__(self, idx):
		# Load image
		img_name = self.img_files[idx]
		img_path = os.path.join(self.img_dir, img_name)
		image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		height, width = image.shape[:2]
		
		# Load landmarks
		ann_name = os.path.splitext(img_name)[0] + '.txt'
		ann_path = os.path.join(self.ann_dir, ann_name)
		t, landmarks = read_txt(ann_path)  # landmarks shape: (num_points, 2)
		
		# Prepare for augmentation
		keypoints = landmarks.tolist()
		transformed = self.transform(
			image=image,
			keypoints=keypoints
		)
		
		image = transformed['image'].div(255.)
		keypoints = np.array(transformed['keypoints'])
		
		landmarks_norm = dsntnn.pixel_to_normalized_coordinates(keypoints, size = (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]))
		
		return {
			'image': image,
			'landmarks': keypoints,
			'landmarks_norm':landmarks_norm,
			'image_path': img_path,
			'orig_size': (width, height)
		}



if __name__ == '__main__':
	
	debug = False
	splits = create_train_val_test_split(Config.DATA_DIR, debug, train_ratio=0.8, val_ratio=0.1, seed=42)
	
	for center in ('MAD', 'BCN', 'BOR', 'IST', 'ZUR'):
		cont = 0
		for ind in splits['train']:
			if ind.startswith(center):
				cont +=1
		print(f'{center} has {cont} samples')
	
	# Create datasets and dataloaders
	train_dataset = SpineDataset(Config.DATA_DIR, splits['train'], mode='train')
	val_dataset = SpineDataset(Config.DATA_DIR, splits['val'], mode='val')


	train_loader = DataLoader(
		train_dataset,
		batch_size=Config.BATCH_SIZE,
		shuffle=True,
		num_workers=Config.NUM_WORKERS,
		pin_memory=True
	)

	for batch in train_loader:
		imgs = batch['image']
		lands = batch['landmarks']
		print(lands.shape)
		for img, land in zip(imgs, lands):
			plt.imshow(img.squeeze(), cmap = 'gray')
			plt.scatter(land[:, 0], land[:, 1])
			plt.show()