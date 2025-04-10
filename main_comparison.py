"""
Main script to compare different training approaches for spine landmark detection.

This script compares:
1. Centralized training (single model on all data)
2. Local training (independent models for each hospital)
3. Federated training (collaborative models across hospitals)

Results are saved for comparison and analysis.
"""
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader

from dataset import SpineDataset, create_train_val_test_split
from model import CoordRegressionNetwork, LandmarkLoss, init_weights
from training_utils import train_centralised_and_local
from config import Config
from fl_experiment import run_federated_experiment
from client_app import client_fn
from server_app import get_server_fn
from strategy import CustomFedAvg, CustomFedOpt, CustomFedProx


def setup_experiment_directories() -> Path:
    """
    Create directories for saving experiment results.
    
    Returns:
        Base directory for experiment results
    """
    base_save_dir = Path(Config.SAVE_DIR)
    base_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {base_save_dir}")
    return base_save_dir


def save_experiment_config(base_dir: Path, splits: Dict[str, Any]) -> None:
    """
    Save experiment configuration and data splits.
    
    Args:
        base_dir: Base directory for saving results
        splits: Dataset splits information
    """
    # Save splits for future reference
    with open(base_dir / 'splits.json', 'w') as f:
        json.dump(splits, f)
    
    # Save configuration
    config_dict = {k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(Config).items()
                  if not k.startswith('__')}
    
    with open(base_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)


def create_model() -> CoordRegressionNetwork:
    """
    Create and initialize the spine landmark detection model.
    
    Returns:
        Initialized CoordRegressionNetwork model
    """
    model = CoordRegressionNetwork(
        arch=Config.ARCH,
        n_locations_global=28,
        n_ch=1,
        n_blocks=Config.N_BLOCKS
    )
    init_weights(model)
    return model


def train_centralized_model(
    splits: Dict[str, Any],
    base_save_dir: Path,
    device: torch.device,
    logger: Optional[Any] = None,
    early_stopping_patience: int = 10
) -> None:
    """
    Train a centralized model on all hospital data with early stopping.
    
    Args:
        splits: Dataset splits information
        base_save_dir: Base directory for saving results
        device: Device to train on
        logger: Optional logger for tracking experiments
        early_stopping_patience: Number of epochs to wait before early stopping
    """
    print("\n" + "="*50)
    print("Training Centralized Model...")
    print("="*50)
    
    centralized_save_dir = base_save_dir / "centralized_model"
    centralized_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create pooled training dataset
    train_dataset = SpineDataset(Config.DATA_DIR, splits['train'], mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create pooled validation dataset
    val_dataset = SpineDataset(Config.DATA_DIR, splits['val'], mode='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Initialize model and optimizer
    model = create_model()
    criterion = LandmarkLoss(
        sigma_t=Config.SIGMA_T
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE
    )
    
    # Train the model with early stopping
    if logger:
        logger.log_training_start('centralized')
        
    train_centralised_and_local(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=Config.NUM_EPOCHS,
        device=device,
        save_dir=centralized_save_dir,
        checkpoint_interval=Config.MODEL_CHECKPOINT_INTERVAL,
        logger=logger,
        patience=early_stopping_patience
    )
    
    print(f"Centralized model saved to {centralized_save_dir}")


def train_local_models(
    splits: Dict[str, Any],
    base_save_dir: Path,
    device: torch.device,
    logger: Optional[Any] = None,
    early_stopping_patience: int = 10
) -> None:
    """
    Train independent local models for each hospital with early stopping.
    
    Args:
        splits: Dataset splits information
        base_save_dir: Base directory for saving results
        device: Device to train on
        logger: Optional logger for tracking experiments
        early_stopping_patience: Number of epochs to wait before early stopping
    """
    print("\n" + "="*50)
    print("Training Local Models...")
    print("="*50)
    
    local_save_dir = base_save_dir / "local_models"
    local_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train a separate model for each hospital
    for center in ['MAD', 'BCN', 'BOR', 'IST', 'ZUR']:
        print(f"\nTraining {center} Model...")
        center_save_dir = local_save_dir / center
        
        if logger:
            logger.log_training_start('local', hospital=center)
        
        # Create hospital-specific datasets
        center_train_dataset = SpineDataset(
            Config.DATA_DIR,
            splits['by_hospital'][center]['train'],
            mode='train'
        )
        center_val_dataset = SpineDataset(
            Config.DATA_DIR,
            splits['by_hospital'][center]['val'],
            mode='val'
        )
        
        # Create DataLoaders
        center_train_loader = DataLoader(
            center_train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
        center_val_loader = DataLoader(
            center_val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
        
        # Initialize model and optimizer
        model = create_model()
        criterion = LandmarkLoss(
            sigma_t=Config.SIGMA_T
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE
        )
        
        # Train hospital-specific model with early stopping
        train_centralised_and_local(
            model=model,
            train_loader=center_train_loader,
            val_loader=center_val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=Config.NUM_EPOCHS,
            device=device,
            save_dir=center_save_dir,
            checkpoint_interval=Config.MODEL_CHECKPOINT_INTERVAL,
            model_type='local',
            hospital=center,
            logger=logger,
            patience=early_stopping_patience
        )
        
        print(f"{center} model saved to {center_save_dir}")


def train_federated_models(
    device: torch.device,
    strategies: Dict[str, Any]
) -> None:
    """
    Train models using federated learning strategies.
    
    Args:
        device: Device to run training on
        strategies: Dictionary mapping strategy names to strategy classes
    """
    print("\n" + "="*50)
    print("Training Federated Models...")
    print("="*50)
    
    for name, strategy_class in strategies.items():
        print(f"\nRunning federated experiment with {name}...")
        
        # Create server function with specified strategy
        server_fn = get_server_fn(
            strategy_class=strategy_class,
            save_suffix=name,
            fl_approach=name
        )
        
        # Run federated learning experiment
        # Note: No early stopping for federated learning
        run_federated_experiment(
            client_fn=client_fn,
            server_fn=server_fn,
            device=device
        )


def train_all_approaches(early_stopping_patience: int = 10) -> None:
    """
    Run all training approaches (centralized, local, federated) and save results.
    
    Args:
        early_stopping_patience: Number of epochs to wait before early stopping
                                (for centralized and local training only)
    """
    # Set up device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset splits
    splits = create_train_val_test_split(
        Config.DATA_DIR,
        Config.DEBUG,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )
    
    # Setup directories and save configuration
    base_save_dir = setup_experiment_directories()
    save_experiment_config(base_save_dir, splits)
    
    # 1. Train Centralized Model with early stopping
    train_centralized_model(
        splits,
        base_save_dir,
        device,
        early_stopping_patience=early_stopping_patience
    )
    
    # 2. Train Local Models with early stopping
    train_local_models(
        splits,
        base_save_dir,
        device,
        early_stopping_patience=early_stopping_patience
    )
    
    # 3. Train Federated Models (no early stopping)
    strategies = {
        #"FedProx": CustomFedProx,
        "FedAvg": CustomFedAvg,
        #"FedOpt": CustomFedOpt,
    }
    
    train_federated_models(device, strategies)


if __name__ == '__main__':
    # Run with early stopping (patience=10) for centralized and local training
    train_all_approaches(early_stopping_patience=10)