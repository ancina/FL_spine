from datetime import datetime

class Config:
	#DATA_DIR = '/cluster/work/bmdslab/ancina/ESSG/lumbar_dataset/'
	DATA_DIR = '../../lumbar_dataset/'
	BATCH_SIZE = 4
	NUM_WORKERS = 0 # put to zero if debugging on laptop
	LEARNING_RATE = 1e-4
	NUM_EPOCHS = 10   # num epochs for non federated
	NUM_ROUNDS_FL = 3
	NUM_ROUNDS_FL_OPT = 10
	LOCAL_EPOCHS = 2
	SIGMA_T = 1.0
	SMOOTHNESS_WEIGHT = 0.0
	MODEL_CHECKPOINT_INTERVAL = 10
	# Data
	IMAGE_SIZE = (768, 768)
	DEBUG = True
	
	# Augmentation
	AUG_BRIGHTNESS_CONTRAST_PROB = 0.8
	AUG_NOISE_PROB = 0.5
	AUG_SHIFT_LIMIT = 0.2
	AUG_SCALE_LIMIT = 0.2
	AUG_ROTATE_LIMIT = 15

	ARCH = 'hg2'
	N_BLOCKS = 2
	
	TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	SAVE_DIR = f'./results_{ARCH}_{TIMESTAMP}'