"""
Inference module for spine landmark detection.

This module provides utilities for loading trained models and running inference
on new spine X-ray images, with visualization of predicted landmarks.
"""
import os
import gc
import json
from pathlib import Path
from typing import List, Tuple, Union, Optional

import cv2
import numpy as np
import torch
import dsntnn
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import CoordRegressionNetwork
from config import Config

# Configure matplotlib for consistent output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def get_transforms(image_shape: Tuple[int, int]) -> A.Compose:
	"""
	Get image transformations for inference.

	Args:
		image_shape: Target shape (height, width) for the transformed image

	Returns:
		Albumentations composition of transforms
	"""
	transform = A.Compose([
		A.Resize(image_shape[0], image_shape[1], cv2.INTER_CUBIC),
		ToTensorV2()
	])
	return transform


@torch.no_grad()
def load_model(
		model_path: Union[str, Path],
		federated: bool = False,
		device: Optional[torch.device] = None
) -> torch.nn.Module:
	"""
	Load a trained spine landmark detection model.

	Args:
		model_path: Path to the saved model weights
		federated: Whether the model was trained with federated learning
		device: Device to load the model on (default: CPU)

	Returns:
		Loaded and evaluation-ready model
	"""
	# Set device
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Initialize model architecture
	model = CoordRegressionNetwork(
		arch=Config.ARCH,
		n_locations_global=28,
		n_ch=1,
		n_blocks=Config.N_BLOCKS
	)
	
	# Load weights based on training approach
	if federated:
		# Federated models are saved directly as state_dict
		model.load_state_dict(torch.load(model_path, map_location=device))
	else:
		# Non-federated models are saved with additional metadata
		checkpoint = torch.load(model_path, map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])
	
	# Set to evaluation mode and move to specified device
	model.eval()
	model.to(device)
	
	return model


def load_split_file(json_path: Union[str, Path]) -> List[str]:
	"""
	Load the test filenames from the JSON split file.

	Args:
		json_path: Path to the splits JSON file

	Returns:
		List of test filenames
	"""
	with open(json_path, 'r') as f:
		splits = json.load(f)
	return splits['test']


def get_test_image_paths(json_path: Union[str, Path], image_dir: Union[str, Path]) -> List[str]:
	"""
	Get paths to all test images based on the splits file.

	Args:
		json_path: Path to the splits JSON file
		image_dir: Directory containing the images

	Returns:
		List of paths to test images
	"""
	test_files = load_split_file(json_path)
	image_paths = []
	
	for filename in test_files:
		# Check for both JPG and PNG extensions
		jpg_path = os.path.join(image_dir, f"{filename}.jpg")
		png_path = os.path.join(image_dir, f"{filename}.png")
		
		if os.path.exists(jpg_path):
			image_paths.append(jpg_path)
		elif os.path.exists(png_path):
			image_paths.append(png_path)
	
	print(f"Found {len(image_paths)} test images out of {len(test_files)} test files")
	return image_paths


def predict_landmarks(
		model: torch.nn.Module,
		image: torch.Tensor,
		image_shape: Tuple[int, int],
		original_shape: Tuple[int, int]
) -> torch.Tensor:
	"""
	Predict landmarks on an image and convert to original image coordinates.

	Args:
		model: Trained landmark detection model
		image: Preprocessed image tensor [1, C, H, W]
		image_shape: Shape of the processed image
		original_shape: Shape of the original image

	Returns:
		Predicted landmark coordinates in original image space
	"""
	# Get model predictions
	with torch.no_grad():
		global_coords, _ = model(image.unsqueeze(dim=0))
	
	# Convert normalized coordinates to pixel coordinates
	coords_processed = dsntnn.normalized_to_pixel_coordinates(global_coords, size=image_shape)
	
	# Scale coordinates back to original image dimensions
	h_org, w_org = original_shape
	h, w = image_shape
	
	scale_factors = torch.tensor([w_org / w, h_org / h])
	coords_original = coords_processed.cpu() * scale_factors
	
	return coords_original.squeeze()


def visualize_and_save_predictions(
		original_image: np.ndarray,
		coords: torch.Tensor,
		img_path: str,
		output_dir: Union[str, Path],
		annotation_dir: Union[str, Path]
) -> None:
	"""
	Visualize predictions and save both the visualization and annotation file.

	Args:
		original_image: Original image array
		coords: Predicted landmark coordinates
		img_path: Path to the input image
		output_dir: Directory to save the visualization
		annotation_dir: Directory to save the annotation file
	"""
	# Vertebrae labels and colors for visualization
	vertebra_labels = ['FEM1', 'FEM2', 'SACRUM', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12']
	colors = ['b', 'b', 'g', 'r', 'c', 'm', 'r', 'orange', 'r']
	
	# Create figure
	plt.figure(figsize=(8, 8))
	plt.imshow(original_image, cmap='gray')
	
	# Get base filename without extension
	img_name = os.path.splitext(os.path.basename(img_path))[0]
	
	# Track position in coordinates array
	vertebra_offset = 0
	
	# Save predictions to annotation file
	with open(os.path.join(annotation_dir, f'{img_name}.txt'), 'w') as annotation_file:
		for i, vertebra in enumerate(vertebra_labels):
			# Determine number of points for this landmark
			if vertebra.startswith('FEM'):
				num_points = 1
			elif vertebra == 'SACRUM':
				num_points = 2
			else:
				num_points = 4
			
			# Get coordinates for this landmark
			landmark_coords = coords[vertebra_offset:vertebra_offset + num_points]
			
			# Plot coordinates
			plt.scatter(
				landmark_coords[:, 0],
				landmark_coords[:, 1],
				s=5, color=colors[i], marker='o'
			)
			
			# Write to annotation file
			annotation_file.write(f"{vertebra}\n")
			np.savetxt(
				annotation_file,
				landmark_coords.cpu().numpy(),
				fmt='%f, %f'
			)
			
			# Update offset for next landmark
			vertebra_offset += num_points
	
	# Add legend and save figure
	plt.scatter([], [], color='k', marker='+', label='Predictions')
	plt.legend(loc='upper left', bbox_to_anchor=(1., 1.), fontsize=10)
	plt.title(img_name)
	plt.tight_layout()
	
	# Save visualization
	plt.savefig(os.path.join(output_dir, f'{img_name}.jpg'))
	plt.close()


def run_inference_on_image(
		model: torch.nn.Module,
		img_path: str,
		output_dir: Union[str, Path],
		annotation_dir: Union[str, Path],
		image_shape: Tuple[int, int] = (768, 768)
) -> None:
	"""
	Run inference on a single image and save results.

	Args:
		model: Trained landmark detection model
		img_path: Path to the input image
		output_dir: Directory to save visualizations
		annotation_dir: Directory to save annotation files
		image_shape: Target shape for the model input
	"""
	# Load and preprocess image
	original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	original_shape = original_image.shape
	
	# Apply transformations
	transform = get_transforms(image_shape)
	transformed = transform(image=original_image)
	image_tensor = transformed['image'].div(255.)  # Normalize to [0, 1]
	
	# Predict landmarks
	coords = predict_landmarks(
		model=model,
		image=image_tensor,
		image_shape=image_shape,
		original_shape=original_shape
	)
	
	# Visualize and save results
	visualize_and_save_predictions(
		original_image=original_image,
		coords=coords,
		img_path=img_path,
		output_dir=output_dir,
		annotation_dir=annotation_dir
	)
	
	# Clean up to reduce memory usage
	del image_tensor, transformed
	gc.collect()
	torch.cuda.empty_cache()


def run_inference(
		results_dir: Union[str, Path],
		training_dir: Union[str, Path],
		data_dir: Union[str, Path],
		approaches: List[str] = None,
		centers: List[str] = None
) -> None:
	"""
	Run inference for all specified approaches and centers.

	Args:
		results_dir: Directory to save inference results
		training_dir: Directory containing trained models
		data_dir: Directory containing test images
		approaches: List of approaches to evaluate ('centralized_model', 'local_models', 'federated_*')
		centers: List of centers for local models ('BOR', 'BCN', 'IST', 'MAD', 'ZUR)
	"""
	# Default values
	if approaches is None:
		approaches = ['centralized_model', 'local_models', 'federated_FedProx']
	
	if centers is None:
		centers = ['BOR', 'BCN', 'IST', 'MAD', 'ZUR']
	
	# Ensure directories exist
	os.makedirs(results_dir, exist_ok=True)
	
	# Load test image paths
	json_path = os.path.join(training_dir, 'splits.json')
	test_image_paths = get_test_image_paths(json_path, data_dir)
	
	# Process each approach
	for approach in approaches:
		approach_dir = os.path.join(results_dir, approach)
		os.makedirs(approach_dir, exist_ok=True)
		
		print(f"\nRunning inference for approach: {approach}")
		
		# Handle local models differently (one model per center)
		if approach == 'local_models':
			for center in centers:
				print(f"Processing center: {center}")
				
				# Create output directories
				pred_imgs_dir = os.path.join(approach_dir, center, 'images')
				os.makedirs(pred_imgs_dir, exist_ok=True)
				
				pred_annot_dir = os.path.join(approach_dir, center, 'annotations')
				os.makedirs(pred_annot_dir, exist_ok=True)
				
				# Load model once per center
				model_path = os.path.join(training_dir, f'{approach}/{center}', 'best_model.pth')
				model = load_model(
					model_path=model_path,
					federated=approach.startswith('federated')
				)
				
				# Process all test images
				for img_path in tqdm(test_image_paths, desc=f"Inferring with {center} model"):
					run_inference_on_image(
						model=model,
						img_path=img_path,
						output_dir=pred_imgs_dir,
						annotation_dir=pred_annot_dir
					)
				
				# Clean up after center processing
				del model
				gc.collect()
				torch.cuda.empty_cache()
		
		else:
			# For centralized and federated approaches
			pred_imgs_dir = os.path.join(approach_dir, 'images')
			os.makedirs(pred_imgs_dir, exist_ok=True)
			
			pred_annot_dir = os.path.join(approach_dir, 'annotations')
			os.makedirs(pred_annot_dir, exist_ok=True)
			
			# Load model once
			model_path = os.path.join(training_dir, approach, 'best_model.pth')
			model = load_model(
				model_path=model_path,
				federated=approach.startswith('federated')
			)
			
			# Process all test images
			for img_path in tqdm(test_image_paths, desc=f"Inferring with {approach} model"):
				run_inference_on_image(
					model=model,
					img_path=img_path,
					output_dir=pred_imgs_dir,
					annotation_dir=pred_annot_dir
				)
			
			# Clean up
			del model
			gc.collect()
			torch.cuda.empty_cache()


def main():
	"""Main entry point for the inference script."""
	# Paths configuration
	base_dir = Path(__file__).parent.absolute()
	data_dir = base_dir / "data" / "images"
	
	# Results directory
	results_dir = base_dir / "results" / "inference"
	os.makedirs(results_dir, exist_ok=True)
	
	# Training directory with saved models. Put the name of the created folder starting with results_namemodel_....
	training_dir = base_dir / "results_hg2_latest"
	
	# Approaches to evaluate
	approaches = ['centralized_model',
	              'local_models',
	              'federated_FedAvg',
	              'federated_FedOpt',
	              'federated_FedProx']
	
	# Run inference
	run_inference(
		results_dir=results_dir,
		training_dir=training_dir,
		data_dir=data_dir,
		approaches=approaches
	)
	
	print("Inference completed successfully!")


if __name__ == '__main__':
	main()