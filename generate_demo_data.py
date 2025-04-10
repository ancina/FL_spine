"""
Script to generate synthetic spine X-ray data for testing.

This script creates:
- A data structure with images and annotations folders
- 20 synthetic X-ray images (5 for each center: MAD, BCN, BOR, IST)
- 20 matching annotation files with spine landmark coordinates
"""
import os
import random
import numpy as np
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path


def create_directory_structure(base_path):
	"""Create the necessary directory structure for the dataset."""
	# Create main directories
	images_dir = os.path.join(base_path, 'images')
	annotations_dir = os.path.join(base_path, 'annotations')
	
	os.makedirs(images_dir, exist_ok=True)
	os.makedirs(annotations_dir, exist_ok=True)
	
	print(f"Created directory structure at: {base_path}")
	return images_dir, annotations_dir


def generate_filename(center_code):
	"""Generate a random filename following the pattern in the dataset."""
	# Generate a random date between 2010 and 2023
	start_date = datetime(2010, 1, 1).toordinal()
	end_date = datetime(2023, 12, 31).toordinal()
	random_date = datetime.fromordinal(random.randint(start_date, end_date))
	date_str = random_date.strftime("%d.%m.%Y")
	
	# Generate a random patient ID (4 digits)
	patient_id = f"{random.randint(1, 9999):04d}"
	
	return f"{center_code}ISSN{patient_id}-SC_{date_str}_lat"


def generate_landmark_coordinates(width, height):
	"""Generate realistic spine landmark coordinates for annotation files."""
	# Calculate a slightly curved spine line
	center_x = width / 2
	spine_deviation = width * 0.05  # How much the spine can deviate from center
	
	# Initialize dictionary for landmarks
	landmarks = {}
	
	# FEM1 (bottom-most point)
	fem1_x = center_x + random.uniform(-spine_deviation, spine_deviation)
	fem1_y = height * 0.95  # Near bottom of image
	landmarks["FEM1"] = [(fem1_x, fem1_y)]
	
	# FEM2
	fem2_x = center_x + random.uniform(-spine_deviation, spine_deviation)
	fem2_y = height * 0.85
	landmarks["FEM2"] = [(fem2_x, fem2_y)]
	
	# SACRUM (2 points)
	sacrum_center_x = center_x + random.uniform(-spine_deviation, spine_deviation)
	sacrum_y = height * 0.75
	landmarks["SACRUM"] = [
		(sacrum_center_x - width * 0.05, sacrum_y),
		(sacrum_center_x + width * 0.05, sacrum_y - height * 0.05)
	]
	
	# Vertebrae (L5 to T12)
	vertebrae = ['L5', 'L4', 'L3', 'L2', 'L1', 'T12']
	current_y = height * 0.65  # Starting position for L5
	vertebra_height = height * 0.08  # Height of each vertebra
	vertebra_width = width * 0.1  # Width of each vertebra
	
	for name in vertebrae:
		vertebra_center_x = center_x + random.uniform(-spine_deviation, spine_deviation)
		
		landmarks[name] = [
			# Bottom left
			(vertebra_center_x - vertebra_width / 2, current_y),
			# Bottom right
			(vertebra_center_x + vertebra_width / 2, current_y),
			# Top right
			(vertebra_center_x + vertebra_width / 2, current_y - vertebra_height),
			# Top left
			(vertebra_center_x - vertebra_width / 2, current_y - vertebra_height)
		]
		
		# Move up for next vertebra
		current_y -= vertebra_height * 1.2
	
	return landmarks


def create_annotation_file(landmarks, annotation_path):
	"""Create an annotation file with the landmarks in the specified format."""
	with open(annotation_path, 'w') as f:
		for label, points in landmarks.items():
			f.write(f"{label}\n")
			for point in points:
				f.write(f"{point[0]}, {point[1]}\n")


def create_synthetic_xray(width, height, image_path, landmarks):
	"""Create a synthetic X-ray image with spine-like structures."""
	# Create a black canvas
	image = Image.new('L', (width, height), color=0)
	draw = ImageDraw.Draw(image)
	
	# Add some random noise for texture
	noise = np.random.normal(loc=30, scale=10, size=(height, width))
	noise = np.clip(noise, 0, 255).astype('uint8')
	noise_img = Image.fromarray(noise)
	image = Image.blend(image, noise_img, alpha=0.7)
	draw = ImageDraw.Draw(image)
	
	# Draw a pelvis-like shape at the bottom
	pelvis_center_x = width / 2
	pelvis_y = height * 0.9
	pelvis_width = width * 0.4
	pelvis_height = height * 0.15
	draw.ellipse(
		[(pelvis_center_x - pelvis_width / 2, pelvis_y - pelvis_height / 2),
		 (pelvis_center_x + pelvis_width / 2, pelvis_y + pelvis_height / 2)],
		fill=80
	)
	
	# Draw vertebrae
	for label in ['L5', 'L4', 'L3', 'L2', 'L1', 'T12']:
		if label in landmarks:
			points = landmarks[label]
			# Calculate the polygon of the vertebra
			draw.polygon(points, fill=120, outline=150)
	
	# Draw the spine line
	spine_points = []
	for label in ['SACRUM', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12']:
		if label in landmarks:
			# Use the middle points for the spine line
			points = landmarks[label]
			if len(points) >= 2:
				center_x = sum(p[0] for p in points) / len(points)
				center_y = sum(p[1] for p in points) / len(points)
				spine_points.append((center_x, center_y))
	
	if spine_points:
		# Draw a thicker line for the spine
		for i in range(len(spine_points) - 1):
			draw.line([spine_points[i], spine_points[i + 1]], fill=180, width=10)
	
	# Add some text to indicate this is a synthetic image
	try:
		# Try to add a small font for the watermark
		font = ImageFont.load_default()
		draw.text((width * 0.05, height * 0.02), "SYNTHETIC IMAGE - FOR TESTING ONLY",
		          fill=50, font=font)
	except Exception:
		# If font loading fails, continue without text
		pass
	
	# Save the image
	image.save(image_path)


def generate_dataset(base_path, num_samples_per_center=5):
	"""Generate the complete synthetic dataset."""
	# Create directory structure
	images_dir, annotations_dir = create_directory_structure(base_path)
	
	# Centers to generate data for
	centers = ['MAD', 'BCN', 'BOR', 'IST']
	
	# Keep track of filenames to avoid duplicates
	filenames = set()
	
	for center in centers:
		for _ in range(num_samples_per_center):
			# Generate dimensions within the specified range
			width = random.randint(1200, 2600)
			height = random.randint(2700, 3000)
			
			# Generate a unique filename
			while True:
				filename = generate_filename(center)
				if filename not in filenames:
					filenames.add(filename)
					break
			
			# Generate landmarks
			landmarks = generate_landmark_coordinates(width, height)
			
			# Create annotation file
			annotation_path = os.path.join(annotations_dir, f"{filename}.txt")
			create_annotation_file(landmarks, annotation_path)
			
			# Create X-ray image
			image_path = os.path.join(images_dir, f"{filename}.jpg")
			create_synthetic_xray(width, height, image_path, landmarks)
			
			print(f"Generated sample: {filename}")
	
	print(f"\nDataset generation complete. Created {len(filenames)} samples.")
	print(f"Images: {images_dir}")
	print(f"Annotations: {annotations_dir}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate synthetic spine X-ray dataset")
	parser.add_argument("--output", default="./data", help="Base directory for the generated dataset")
	parser.add_argument("--samples", type=int, default=20, help="Number of samples per center")
	
	args = parser.parse_args()
	
	generate_dataset(args.output, args.samples)