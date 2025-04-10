# logger_utils.py
import logging
from pathlib import Path
from datetime import datetime
import sys

class TrainingLogger:
    """A custom logger for tracking model training progress.
    
    This logger creates formatted log messages for training events across
    centralized, local, and federated learning approaches. It saves logs
    to both a file and prints to console.
    
    Attributes:
        logger: The Python logger instance
        log_file: Path to the log file
    """
    
    def __init__(self, base_save_dir: Path, filename: str = "training.log"):
        """Initialize the training logger.
        
        Args:
            base_save_dir: Base directory where logs will be saved
            filename: Name of the log file (default: "training.log")
        """
        self.log_file = base_save_dir / filename
        
        # Create logger
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.INFO)
        
        # Create formatters and handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_training_start(self, model_type: str, hospital: str = None):
        """Log the start of model training.
        
        Args:
            model_type: Type of model ('centralized', 'local', or 'federated')
            hospital: Hospital name for local models
        """
        if hospital:
            self.logger.info(f"Starting {model_type} model training for {hospital}")
        else:
            self.logger.info(f"Starting {model_type} model training")
    
    def log_best_loss(self, model_type: str, loss: float, epoch: int = None, 
                      round: int = None, hospital: str = None):
        """Log when a new best loss is achieved.
        
        Args:
            model_type: Type of model ('centralized', 'local', or 'federated')
            loss: The loss value achieved
            epoch: Training epoch (for centralized and local models)
            round: FL round (for federated model)
            hospital: Hospital name for local models
        """
        if round is not None:
            self.logger.info(f"New best loss for {model_type} model: {loss:.4f} at round {round}")
        else:
            if hospital:
                self.logger.info(f"New best loss for {model_type} model ({hospital}): {loss:.4f} at epoch {epoch}")
            else:
                self.logger.info(f"New best loss for {model_type} model: {loss:.4f} at epoch {epoch}")

# Create global logger instance
training_logger = None

def initialize_logger(base_save_dir: Path):
    """Initialize the global logger instance.
    
    Args:
        base_save_dir: Base directory where logs will be saved
    """
    global training_logger
    training_logger = TrainingLogger(base_save_dir)
    return training_logger
