import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import configparser
from pathlib import Path
import glob
from tqdm import tqdm
import time
import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import CoordRegressionNetwork
from config import Config
import torch
import dsntnn
import gc
from typing import List, Tuple, Union
import json
from PIL import Image

plt.rcParams['figure.dpi'] = 300  # Reduced DPI for lower memory usage
plt.rcParams['savefig.dpi'] = 300

main_path = Path(__file__).parent.absolute()


def get_transforms(image_shape):
	'''CHECK THAT CLAHE IS USED DURING TRAINING'''
	transform = A.Compose([
		A.Resize(image_shape[0], image_shape[1], cv.INTER_CUBIC),
		# Increase intensity augmentations
		# A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=1.0),  # Increased probability and range
		ToTensorV2()
	])
	return transform


@torch.no_grad()  # Decorator to disable gradient computation
def get_trained_model(path, federated):
	model = CoordRegressionNetwork(
			arch=Config.ARCH,
			n_locations_global=28,
			#kernel_size=Config.KERNEL_SIZE,
			n_ch=1,
			n_blocks=Config.N_BLOCKS
		)
	
	if federated:
		model.load_state_dict(torch.load(path, map_location='cpu'))
	else:
		model.load_state_dict(torch.load(path, map_location='cpu')['model_state_dict'])
	model.eval()  # Set to evaluation mode
	return model


def load_split_file(json_path: str) -> List[str]:
	"""
	Load the test filenames from the JSON split file.

	Args:
		json_path (str): Path to the splits JSON file

	Returns:
		List[str]: List of test filenames
	"""
	with open(json_path, 'r') as f:
		splits = json.load(f)
	return splits['test']


def process_test_images(json_path: str, image_dir: str) -> List[Tuple[str, np.ndarray]]:
	"""
	Load and process all test images.

	Args:
		json_path (str): Path to the splits JSON file
		image_dir (str): Directory containing the images

	Returns:
		List[Tuple[str, np.ndarray]]: List of tuples containing (filename, image)
	"""
	test_files = load_split_file(json_path)
	images_paths = []
	
	for filename in test_files:
		# Construct full image path
		image_path = os.path.join(image_dir, filename + '.jpg')
		
		images_paths.append(image_path)
		
	print(f"Successfully loaded {len(images_paths)} test images")
	return images_paths


def eval_img(img_path, pred_imgs_fold, pred_annot_fold, model):  # Pass model as parameter
	image_shape = (768, 768)
	transform = get_transforms(image_shape)
	
	im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	org_image = im.copy()
	org_shape = org_image.shape
	
	# Transform image
	transformation_dict = transform(image=org_image)
	image = transformation_dict['image'].div(255.)
	
	# Predict
	with torch.no_grad():
		outputs = model(image.unsqueeze(dim=0))
	global_coords, heatmaps = outputs
	# global_coords, heatmaps, local_coords = outputs
	
	# Process coordinates
	h_org, w_org = org_shape[:2]
	h, w = image_shape
	h_org, w_org = int(h_org), int(w_org)
	h, w = int(h), int(w)
	
	coords_retransformed_pred_global = dsntnn.normalized_to_pixel_coordinates(global_coords, size=image_shape)
	
	dims = torch.stack((torch.tensor(w_org / w), torch.tensor(h_org / h)))
	coords_original_pred = coords_retransformed_pred_global.cpu() * dims
	
	coords_original_pred = coords_original_pred.squeeze()
	
	# Plot results
	list_vertebrae = ['FEM1', 'FEM2', 'SACRUM', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12']
	colors = ['b', 'b', 'g', 'r', 'c', 'm', 'r', 'orange', 'r']
	
	plt.figure(figsize=(8, 8), dpi=300)
	plt.imshow(im, cmap='gray')
	
	vertebra_num = 0
	n_point = 0
	
	img_name = os.path.splitext(os.path.basename(img_path))[0]
	# Save predictions
	with open(os.path.join(pred_annot_fold, f'{img_name}.txt'), 'w') as glob_txt:
		
		for i, vertebra in enumerate(list_vertebrae):
			vertebra_num += n_point
			if vertebra.startswith('FEM'):
				n_point = 1
			elif vertebra == 'SACRUM':
				n_point = 2
			else:
				n_point = 4
			
			plt.scatter(coords_original_pred[vertebra_num:vertebra_num + n_point, 0].cpu(),
			            coords_original_pred[vertebra_num:vertebra_num + n_point, 1].cpu(),
			            s=5, color=colors[i], marker='o')
			
			# Save coordinates
			glob_txt.write(f"{vertebra}\n")
			coords = coords_original_pred[vertebra_num:vertebra_num + n_point, :].cpu().numpy()
			np.savetxt(glob_txt, coords, fmt='%f, %f')
			
	
	plt.scatter([], [], color='k', marker='+', label='Global')
	# plt.scatter([], [], color='k', marker='o', label='Local')
	plt.legend(loc='upper left', bbox_to_anchor=(1., 1.), fontsize=10)
	plt.title(os.path.basename(img_path)[:-8])
	plt.tight_layout()
	plt.savefig(os.path.join(pred_imgs_fold, os.path.splitext(os.path.basename(img_path))[0] + '.jpg'))
	plt.close()  # Explicitly close the figure
	
	# Clear some memory
	del image, transformation_dict, outputs, global_coords, heatmaps
	gc.collect()
	torch.cuda.empty_cache()  # Still useful even on CPU to clear memory


def process():
	path_imgs = r"/Users/ancina/Documents/ESSG project/lumbar_dataset/images"
	# path_annotations = r"/Users/ancina/Documents/ESSG project/lumbar_dataset/annotations"
	
	path_save_results = r'/Users/ancina/Documents/ESSG project/FederatedLearning/'
	fold_res_name = 'results_hg2_new_final_additional_approaches'
	predicted_imgs_folder = os.path.join(path_save_results, fold_res_name)
	os.makedirs(predicted_imgs_folder, exist_ok=True)
	
	# fold_training = 'results_hg2_2025-02-16_21-28-46'
	fold_training = 'results_hg2_2025-03-20_11-47-54'
	json_path = os.path.join(fold_training, 'splits.json')
	paths_to_imgs = process_test_images(json_path, image_dir=path_imgs)
	
	# approaches = ['centralized_model', 'local_models', 'federated']
	approaches = ['federated_FedProx']
	
	for approach in approaches:
		approach_folder = os.path.join(predicted_imgs_folder, approach)
		os.makedirs(approach_folder, exist_ok=True)
		
		if approach == 'local_models':
			for center in ['BOR', 'BCN', 'IST', 'MAD']:
				pred_imgs_folder = os.path.join(approach_folder, center, 'images')
				os.makedirs(pred_imgs_folder, exist_ok=True)
				pred_annot_folder = os.path.join(approach_folder, center, 'annotations')
				os.makedirs(pred_annot_folder, exist_ok=True)
				# Load model once
				model = get_trained_model(
					path=os.path.join(fold_training, f'{approach}/{center}', 'best_model.pth'), federated = True if approach.startswith('federated') else False)
				
				# Process files one by one using generators
				for name in tqdm(paths_to_imgs, desc=f'Computing landmarks for approach {approach}'):
					print(name)
					eval_img(img_path=name, pred_imgs_fold=pred_imgs_folder, pred_annot_fold = pred_annot_folder, model=model)
		else:
			pred_imgs_folder = os.path.join(approach_folder, 'images')
			os.makedirs(pred_imgs_folder, exist_ok=True)
			pred_annot_folder = os.path.join(approach_folder, 'annotations')
			os.makedirs(pred_annot_folder, exist_ok=True)
			# Load model once
			model = get_trained_model(
				path=os.path.join(fold_training, f'{approach}', 'best_model.pth'),
				federated=True if approach.startswith('federated') else False)
			
			# Process files one by one using generators
			for name in tqdm(paths_to_imgs, desc=f'Computing landmarks'):
				print(name)
				eval_img(img_path=name, pred_imgs_fold=pred_imgs_folder, pred_annot_fold = pred_annot_folder, model=model)
			
			
		# Force garbage collection after each image
		gc.collect()
		torch.cuda.empty_cache()


if __name__ == '__main__':
	SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
	process()