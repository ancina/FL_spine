"""
Utility to visualize spine landmarks on X-ray images.

This script loads a spine X-ray image and its corresponding annotation file,
then displays the image with landmarks overlaid.
"""
import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from pathlib import Path


def parse_annotation_file(annotation_path):
	"""
	Parse a spine landmark annotation file.

	Args:
		annotation_path: Path to the annotation file

	Returns:
		Dictionary mapping landmark labels to lists of coordinate points
	"""
	landmarks = {}
	current_label = None
	
	with open(annotation_path, 'r') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			
			# Lines with commas are coordinate pairs
			if ',' in line:
				x, y = map(float, line.split(','))
				
				if current_label:
					if current_label not in landmarks:
						landmarks[current_label] = []
					landmarks[current_label].append((x, y))
			else:
				# Lines without commas are landmark labels
				current_label = line
	
	return landmarks


def visualize_landmarks(image_path, annotation_path, output_path=None):
	"""
	Visualize spine landmarks on an X-ray image.

	Args:
		image_path: Path to the X-ray image
		annotation_path: Path to the annotation file with landmarks
		output_path: Optional path to save the visualization
	"""
	# Load the image
	try:
		image = Image.open(image_path)
		img_array = np.array(image)
	except Exception as e:
		print(f"Error loading image {image_path}: {e}")
		return
	
	# Parse the annotation file
	try:
		landmarks = parse_annotation_file(annotation_path)
	except Exception as e:
		print(f"Error parsing annotation file {annotation_path}: {e}")
		return
	
	# Setup the plot
	plt.figure(figsize=(10, 16))
	plt.imshow(img_array, cmap='gray')
	
	# Color mapping for different landmark types
	colors = {
		'FEM': 'red',
		'SACRUM': 'orange',
		'L': 'yellow',
		'T': 'green'
	}
	
	# Plot each landmark
	for label, points in landmarks.items():
		# Determine color based on label prefix
		color = 'white'  # Default color
		for prefix, c in colors.items():
			if label.startswith(prefix):
				color = c
				break
		
		# Plot points
		x_coords = [p[0] for p in points]
		y_coords = [p[1] for p in points]
		
		plt.scatter(x_coords, y_coords, color=color, s=20, label=label)
		
		# For vertebrae with 4 points, draw a polygon
		if len(points) == 4:
			poly = patches.Polygon(points, closed=True, fill=False,
			                       edgecolor=color, linewidth=1)
			plt.gca().add_patch(poly)
		
		# Add label near the centroid
		centroid_x = sum(x_coords) / len(x_coords)
		centroid_y = sum(y_coords) / len(y_coords)
		plt.text(centroid_x, centroid_y, label, color='white',
		         fontsize=8, ha='center', va='center',
		         bbox=dict(facecolor='black', alpha=0.5))
	
	# Set title and hide axes
	plt.title(f"Spine Landmarks: {os.path.basename(image_path)}")
	plt.axis('off')
	
	# Use a unique legend that doesn't repeat
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys(),
	           loc='upper right', fontsize='small')
	
	# Show or save
	if output_path:
		plt.savefig(output_path, bbox_inches='tight')
		print(f"Visualization saved to {output_path}")
	else:
		plt.tight_layout()
		plt.show()


def visualize_all_in_directory(data_dir, output_dir=None):
	"""
	Visualize all image-annotation pairs in a dataset directory.

	Args:
		data_dir: Base directory containing 'images' and 'annotations' folders
		output_dir: Optional directory to save visualizations
	"""
	images_dir = os.path.join(data_dir, 'images')
	annotations_dir = os.path.join(data_dir, 'annotations')
	
	if output_dir and not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	# Get all image files
	image_files = [f for f in os.listdir(images_dir)
	               if f.endswith(('.jpg', '.png'))]
	
	print(f"Found {len(image_files)} images to visualize")
	
	for img_file in image_files:
		base_name = os.path.splitext(img_file)[0]
		annotation_file = f"{base_name}.txt"
		
		img_path = os.path.join(images_dir, img_file)
		ann_path = os.path.join(annotations_dir, annotation_file)
		
		if not os.path.exists(ann_path):
			print(f"Warning: No annotation file found for {img_file}")
			continue
		
		if output_dir:
			out_path = os.path.join(output_dir, f"{base_name}_visualization.png")
		else:
			out_path = None
		
		print(f"Visualizing {base_name}...")
		visualize_landmarks(img_path, ann_path, out_path)
	
	print("Visualization complete!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Visualize spine landmarks on X-ray images")
	parser.add_argument("--data_dir", default="./data", help="Base directory for dataset")
	parser.add_argument("--output_dir", help="Directory to save visualizations (optional)")
	parser.add_argument("--image", help="Path to specific image file to visualize (optional)")
	parser.add_argument("--annotation", help="Path to specific annotation file to visualize (optional)")
	
	args = parser.parse_args()
	
	if args.image and args.annotation:
		# Visualize a specific image-annotation pair
		visualize_landmarks(args.image, args.annotation)
	else:
		# Visualize all pairs in the dataset
		visualize_all_in_directory(args.data_dir, args.output_dir)