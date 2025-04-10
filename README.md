# Spine Landmark Detection with Federated Learning

This repository contains code for spine landmark detection in X-ray images using different learning approaches: centralized, local, and federated learning.

## Project Structure

```
federatedlearning/
├── backbones.py            # Hourglass network backbone implementations
├── client_app.py           # Federated learning client implementation
├── config.py               # Configuration settings
├── dataset.py              # Dataset loading and preprocessing
├── inference.py            # Inference module
├── fl_experiment.py        # Federated learning experiment runner
├── main_comparison.py      # Main script to compare training approaches
├── model.py                # Landmark detection model definitions
├── server_app.py           # Federated learning server implementation
├── strategy.py             # Custom federated learning strategies
├── training_utils.py       # Training and evaluation utilities
├── data/                   # Sample data directory
│     ├── images/           # Sample X-ray images
│     └── annotations/      # Landmark annotations
├── requirements.txt        # Project dependencies
├── LICENSE                 # License information
└── README.md               # Project documentation
```

## Data Structure
```
data/
├── images/                # Contains synthetic X-ray images
│   ├── MADISSNxxxx-SC_DD.MM.YYYY_lat.jpg
│   ├── BCNISSNxxxx-SC_DD.MM.YYYY_lat.jpg
│   ├── BORISSNxxxx-SC_DD.MM.YYYY_lat.jpg
│   └── ISTISSNxxxx-SC_DD.MM.YYYY_lat.jpg
│
├── annotations/           # Contains landmark annotation files
│   ├── MADISSNxxxx-SC_DD.MM.YYYY_lat.txt
│   ├── BCNISSNxxxx-SC_DD.MM.YYYY_lat.txt
│   ├── BORISSNxxxx-SC_DD.MM.YYYY_lat.txt
│   └── ISTISSNxxxx-SC_DD.MM.YYYY_lat.txt
│
└── README.md              # This file
```
## Annotation file structure

Each vertebra label is followed by the x,y coordinates of its landmarks.
```
FEM1
818.6675991632925, 2840.527337204665
FEM2
748.8119363003934, 2569.028446821698
SACRUM
913.4623205899692, 1882.708664751055
1052.2984045899689, 1707.923951751055
L5
760.4899095922626, 1580.6389793386952
965.770459457981, 1493.962652705206
...etc...
```

## Key Features

- Spine landmark detection using coordinate regression
- Multiple training approaches:
  - Centralized: Single model trained on all data
  - Local: Independent models for each hospital
  - Federated: Collaborative learning across hospitals
- Various federated learning strategies (FedAvg, FedProx, FedOpt)
- Data preprocessing and augmentation for X-ray images
- Training and evaluation utilities
- Experiment tracking and visualization

## Installation

1. Clone the repository:
```bash
git clone git@gitlab.ethz.ch:BMDSlab/publications/low-back/federatedlearning.git
cd federatedlearning
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Demo Data

The repository includes a small demo dataset in the `data/` directory to allow testing the code without requiring access to the full dataset. The demo data consists of:

- X-ray images (synthetic images for demonstration purposes)
- Annotation files with landmark coordinates

## Usage

### Running a Training Comparison

To compare different training approaches, run:

```bash
python main_comparison.py
```

Modify `config.py` to adjust experiment parameters.

### Customizing the Experiment

- Edit `config.py` to change model architecture, training parameters, etc.
- Uncomment specific training approaches in `main_comparison.py`
- Adjust federation strategies in the `strategies` dictionary

## Federated Learning Details

This implementation uses the Flower framework for federated learning, with custom strategies:

- **FedAvg**: Federated Averaging for basic parameter aggregation
- **FedProx**: Adds proximal term regularization to accommodate heterogeneous clients
- **FedOpt**: Applies server-side optimization to federated learning

## Contributing

## License

## Citation

