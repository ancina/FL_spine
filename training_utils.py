"""
Training utilities for spine landmark detection models.

This module contains helper functions and classes for model training,
evaluation, and metric tracking, including early stopping functionality.
"""
import os
import json
import gc
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LandmarkLoss
from config import Config


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss doesn't improve for a specified patience.
    
    Saves the best model based on validation loss.
    """
    
    def __init__(self, patience: int = 10, delta: float = 0, save_path: Union[str, Path] = None):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            delta: Minimum change in the monitored quantity to qualify as an improvement
            save_path: Path to save the best model
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            val_loss: Validation loss
            model: Model to save if validation loss improves
            
        Returns:
            True if early stopping criteria are met, False otherwise
        """
        score = -val_loss  # Higher score is better (negative loss)
        
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # Loss did not improve enough
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Loss improved
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """
        Save model when validation loss improves.
        
        Args:
            val_loss: Validation loss
            model: Model to save
        """
        if self.save_path is None:
            return
        
        if val_loss < self.best_loss:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
            self.best_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, self.save_path / 'best_model.pth')


class TrainingMetrics:
    """
    Class to store and manage training metrics.
    
    Tracks losses during training and provides utilities for saving
    and visualizing metrics.
    """
    
    def __init__(self):
        """Initialize an empty metrics tracker."""
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        # Optional: Add more detailed metrics
        self.component_losses = {
            'global_loss': []
        }
    
    def update_train(self, total_loss: float, component_losses: Optional[Dict[str, float]] = None) -> None:
        """
        Update training metrics after an epoch.
        
        Args:
            total_loss: Overall training loss for the epoch
            component_losses: Optional dictionary of loss components
        """
        self.train_losses.append(total_loss)
        
        # Update component losses if provided
        if component_losses:
            for name, value in component_losses.items():
                if name in self.component_losses:
                    self.component_losses[name].append(value)
    
    def update_val(self, total_loss: float) -> bool:
        """
        Update validation metrics after an epoch.
        
        Args:
            total_loss: Validation loss for the epoch
            
        Returns:
            True if this is the best validation loss so far, False otherwise
        """
        self.val_losses.append(total_loss)
        if total_loss < self.best_val_loss:
            self.best_val_loss = total_loss
            return True
        return False
    
    def save_metrics(self, save_dir: Union[str, Path]) -> None:
        """
        Save training metrics to a JSON file.
        
        Args:
            save_dir: Directory to save metrics file
        """
        metrics_dict = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'component_losses': self.component_losses
        }
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_dict, f)
    
    def plot_losses(self, save_dir: Union[str, Path]) -> None:
        """
        Generate and save plots of training metrics.
        
        Args:
            save_dir: Directory to save the plots
        """
        # Plot overall losses
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Total Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot component losses if available
        if any(len(losses) > 0 for losses in self.component_losses.values()):
            plt.subplot(2, 1, 2)
            for name, values in self.component_losses.items():
                if values:
                    plt.plot(values, label=name)
            plt.title('Loss Components vs Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_plots.png'))
        plt.close()


def train(
    net: nn.Module,
    trainloader: DataLoader,
    lr: float,
    device: torch.device,
    epochs: int,
    verbose: bool = False,
    global_params: Optional[List[torch.Tensor]] = None,
    proximal_mu: float = 0.0
) -> float:
    """
    Train a network on the training set, optionally with FedProx regularization.
    
    Args:
        net: PyTorch model to train
        trainloader: DataLoader for training data
        lr: Learning rate
        device: Device to run training on
        epochs: Number of training epochs
        verbose: Whether to print progress information
        global_params: List of global parameters for FedProx
        proximal_mu: Coefficient for proximal term in FedProx
        
    Returns:
        Final training loss
    """
    criterion = LandmarkLoss(
        sigma_t=Config.SIGMA_T
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    
    final_loss = 0.0
    
    for epoch in tqdm(range(epochs), desc='Training local model...'):
        epoch_loss = 0.0
        samples_count = 0
        
        for batch in trainloader:
            images = batch["image"].to(device)
            landmarks = batch["landmarks_norm"].to(device)
            batch_size = images.shape[0]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(images)
            losses = criterion(outputs, landmarks)
            loss_value = losses['total_loss']
            
            # Add FedProx regularization if applicable
            if global_params is not None:
                proximal_term = 0.0
                for local_param, global_param in zip(net.parameters(), global_params):
                    proximal_term += torch.square((local_param - global_param).norm(2))
                loss_value = loss_value + (proximal_mu / 2) * proximal_term
            
            # Backward pass and optimization
            loss_value.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss_value.item() * batch_size
            samples_count += batch_size
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / samples_count
        final_loss = avg_epoch_loss
        
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}: train loss {avg_epoch_loss:.6f}")
        
        # Free up memory
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return final_loss


def test(net: nn.Module, testloader: DataLoader, device: torch.device) -> float:
    """
    Evaluate a network on the test/validation set.
    
    Args:
        net: PyTorch model to evaluate
        testloader: DataLoader for test/validation data
        device: Device to run evaluation on
        
    Returns:
        Average loss on the dataset
    """
    criterion = LandmarkLoss(
        sigma_t=Config.SIGMA_T
    )
    
    total_loss = 0.0
    samples_count = 0
    net.eval()
    
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            landmarks = batch["landmarks_norm"].to(device)
            batch_size = images.shape[0]
            
            # Forward pass
            outputs = net(images)
            losses = criterion(outputs, landmarks)
            
            # Track loss
            total_loss += losses['total_loss'].item() * batch_size
            samples_count += batch_size
    
    return total_loss / samples_count


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device
) -> Dict[str, float]:
    """
    Train a model for one epoch.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        epoch: Current epoch number
        device: Device to run training on
        
    Returns:
        Dictionary with average loss values
    """
    model.train()
    total_loss = 0.0
    component_losses = {
        'global_loss': 0.0
    }
    samples_count = 0
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}')
    
    for batch in progress_bar:
        imgs = batch['image'].to(device)
        landmarks = batch['landmarks_norm'].to(device)
        batch_size = imgs.shape[0]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(imgs)
        losses = criterion(outputs, landmarks)
        
        # Backward pass
        losses['total_loss'].backward()
        optimizer.step()
        
        # Track losses
        total_loss += losses['total_loss'].item() * batch_size
        for name in component_losses:
            if name in losses:
                component_losses[name] += losses[name].item() * batch_size
        
        samples_count += batch_size
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f"{losses['total_loss'].item():.4f}"})
    
    # Calculate average losses
    avg_loss = total_loss / samples_count
    avg_component_losses = {}
    avg_component_losses['total_loss'] = avg_loss
    
    return avg_component_losses


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    device: torch.device
) -> float:
    """
    Validate a model for one epoch.
    
    Args:
        model: PyTorch model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        epoch: Current epoch number
        device: Device to run validation on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    samples_count = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}')
        
        for batch in progress_bar:
            imgs = batch['image'].to(device)
            landmarks = batch['landmarks_norm'].to(device)
            batch_size = imgs.shape[0]
            
            # Forward pass
            outputs = model(imgs)
            losses = criterion(outputs, landmarks)
            
            # Track loss
            total_loss += losses['total_loss'].item() * batch_size
            samples_count += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({'val_loss': f"{losses['total_loss'].item():.4f}"})
    
    return total_loss / samples_count


def train_centralised_and_local(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: Union[str, Path],
    checkpoint_interval: int = 5,
    model_type: str = 'centralized',
    hospital: Optional[str] = None,
    logger: Optional[Any] = None,
    patience: int = 10
) -> Tuple[nn.Module, TrainingMetrics]:
    """
    Train a model with regular checkpointing, validation, and early stopping.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to run training on
        save_dir: Directory to save checkpoints and metrics
        checkpoint_interval: Interval for saving model checkpoints
        model_type: Type of model ('centralized' or 'local')
        hospital: Hospital code for local models
        logger: Optional logger for tracking experiments
        patience: Number of epochs to wait before early stopping
        
    Returns:
        Tuple of (trained model, training metrics)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = TrainingMetrics()
    model = model.to(device)
    
    # Store initial learning rate for potential LR scheduling
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, save_path=save_dir)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Learning rate adjustment (simple step decay)
        if epoch == 30:  # Could be parameterized
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr / 10
            print(f"Reducing learning rate to {initial_lr / 10}")
        
        # Training phase
        train_losses = train_epoch(model, train_loader, criterion, optimizer, epoch, device)
        metrics.update_train(train_losses['total_loss'], train_losses)
        
        # Validation phase
        val_loss = validate_epoch(model, val_loader, criterion, epoch, device)
        is_best = metrics.update_val(val_loss)
        
        # Save metrics and plots
        metrics.save_metrics(save_dir)
        metrics.plot_losses(save_dir)
        
        # Save model checkpoint periodically
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, save_dir, name=f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save best model
        if is_best:
            if logger is not None:
                logger.log_best_loss(model_type=model_type, loss=val_loss, epoch=epoch + 1, hospital=hospital)
            print(f"New best model with validation loss: {val_loss:.4f}")
            save_checkpoint(model, save_dir, name='best_model.pth')
        
        # Check early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load the best model before returning
    best_model_path = save_dir / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model with validation loss: {early_stopping.best_loss:.4f}")
    
    return model, metrics


def save_checkpoint(model: nn.Module, save_dir: Union[str, Path], name: str) -> None:
    """
    Save a model checkpoint.
    
    Args:
        model: PyTorch model to save
        save_dir: Directory to save the checkpoint
        name: Filename for the checkpoint
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        # Optionally add more information:
        # 'epoch': epoch,
        # 'optimizer_state_dict': optimizer.state_dict(),
    }
    
    torch.save(checkpoint, save_dir / name)
    print(f"Checkpoint saved to {save_dir / name}")


def load_checkpoint(model: nn.Module, checkpoint_path: Union[str, Path], device: torch.device) -> nn.Module:
    """
    Load a model from a checkpoint.
    
    Args:
        model: PyTorch model architecture to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model onto
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model