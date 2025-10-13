"""
Trainer para mini-modelos de LatticeWeaver.

Este módulo implementa el sistema de entrenamiento completo para la suite
de 120 mini-modelos, con soporte para:
- Entrenamiento estándar
- Early stopping
- Learning rate scheduling
- Checkpointing
- Logging de métricas
- Validación
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import time


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    
    # Hiperparámetros básicos
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Learning rate scheduling
    lr_scheduler: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "step"
    lr_factor: float = 0.5
    lr_patience: int = 5
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    save_frequency: int = 10  # epochs
    
    # Logging
    log_frequency: int = 10  # batches
    verbose: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Validation
    validation_split: float = 0.2
    
    # Gradient clipping
    max_grad_norm: Optional[float] = 1.0


class MiniModelDataset(Dataset):
    """Dataset para mini-modelos."""
    
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            features: Tensor de features [N, feature_dim]
            targets: Tensor de targets [N, target_dim] o [N]
        """
        assert len(features) == len(targets), "Features and targets must have same length"
        self.features = features
        self.targets = targets
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]


class MiniModelTrainer:
    """Trainer para mini-modelos."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        loss_fn: Optional[Callable] = None,
        metrics: Optional[Dict[str, Callable]] = None
    ):
        """
        Args:
            model: Modelo a entrenar
            config: Configuración de entrenamiento
            loss_fn: Función de pérdida (default: MSE para regresión)
            metrics: Diccionario de métricas adicionales
        """
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        # Loss function
        self.loss_fn = loss_fn or nn.MSELoss()
        
        # Metrics
        self.metrics = metrics or {}
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Add metric history
        for metric_name in self.metrics:
            self.history[f'train_{metric_name}'] = []
            self.history[f'val_{metric_name}'] = []
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_scheduler(self):
        """Crear learning rate scheduler."""
        if self.config.lr_scheduler == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_factor,
                patience=self.config.lr_patience
            )
        elif self.config.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=self.config.lr_factor
            )
        else:
            return None
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ) -> Dict[str, List[float]]:
        """
        Entrenar modelo.
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación (opcional)
        
        Returns:
            Historial de entrenamiento
        """
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self._train_epoch(train_loader)
            
            # Validate
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)
            else:
                val_metrics = {'loss': 0.0}
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            for metric_name in self.metrics:
                if metric_name in train_metrics:
                    self.history[f'train_{metric_name}'].append(train_metrics[metric_name])
                if metric_name in val_metrics:
                    self.history[f'val_{metric_name}'].append(val_metrics[metric_name])
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Logging
            if self.config.verbose:
                self._log_epoch(epoch, train_metrics, val_metrics)
            
            # Checkpointing
            if (epoch + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(epoch, val_metrics['loss'])
            
            # Early stopping
            if val_loader is not None:
                if self._check_early_stopping(val_metrics['loss']):
                    if self.config.verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        # Save final checkpoint
        self._save_checkpoint(self.current_epoch, val_metrics['loss'], final=True)
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Entrenar una época."""
        self.model.train()
        
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metrics}
        num_batches = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            # Move to device
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            
            # Compute loss
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            
            for metric_name, metric_fn in self.metrics.items():
                metric_value = metric_fn(outputs, targets)
                total_metrics[metric_name] += metric_value.item() if torch.is_tensor(metric_value) else metric_value
            
            num_batches += 1
            
            # Batch logging
            if self.config.verbose and (batch_idx + 1) % self.config.log_frequency == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}: loss={avg_loss:.6f}", end='\r')
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {name: value / num_batches for name, value in total_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validar una época."""
        self.model.eval()
        
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metrics}
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                # Move to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Compute loss
                loss = self.loss_fn(outputs, targets)
                
                # Accumulate metrics
                total_loss += loss.item()
                
                for metric_name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(outputs, targets)
                    total_metrics[metric_name] += metric_value.item() if torch.is_tensor(metric_value) else metric_value
                
                num_batches += 1
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {name: value / num_batches for name, value in total_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Verificar early stopping."""
        if val_loss < self.best_val_loss - self.config.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.patience:
                return True
            return False
    
    def _log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log de época."""
        log_str = f"\nEpoch {epoch + 1}/{self.config.num_epochs}"
        log_str += f" - train_loss: {train_metrics['loss']:.6f}"
        log_str += f" - val_loss: {val_metrics['loss']:.6f}"
        
        for metric_name in self.metrics:
            if metric_name in train_metrics:
                log_str += f" - train_{metric_name}: {train_metrics[metric_name]:.4f}"
            if metric_name in val_metrics:
                log_str += f" - val_{metric_name}: {val_metrics[metric_name]:.4f}"
        
        log_str += f" - lr: {self.optimizer.param_groups[0]['lr']:.6f}"
        
        print(log_str)
    
    def _save_checkpoint(self, epoch: int, val_loss: float, final: bool = False):
        """Guardar checkpoint."""
        if self.config.save_best_only and val_loss >= self.best_val_loss and not final:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        if final:
            checkpoint_path = self.checkpoint_dir / "final_model.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        
        if self.config.verbose and not final:
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Cargar checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.config.verbose:
            print(f"Loaded checkpoint from epoch {self.current_epoch + 1}")
    
    def save_history(self, path: str):
        """Guardar historial de entrenamiento."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


# Métricas comunes
def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Accuracy para clasificación."""
    predictions = torch.argmax(outputs, dim=-1)
    correct = (predictions == targets).sum().item()
    return correct / len(targets)


def mae(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(outputs - targets)).item()


def mape(outputs: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    return torch.mean(torch.abs((targets - outputs) / (targets + epsilon))).item() * 100


def r2_score(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """R² score."""
    ss_res = torch.sum((targets - outputs) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return (1 - ss_res / ss_tot).item()


if __name__ == "__main__":
    # Demo: entrenar modelo simple
    print("=== Demo: MiniModelTrainer ===\n")
    
    # Crear modelo simple
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    
    # Generar datos sintéticos
    torch.manual_seed(42)
    X_train = torch.randn(1000, 10)
    y_train = torch.sum(X_train ** 2, dim=1, keepdim=True) + torch.randn(1000, 1) * 0.1
    
    X_val = torch.randn(200, 10)
    y_val = torch.sum(X_val ** 2, dim=1, keepdim=True) + torch.randn(200, 1) * 0.1
    
    # Datasets
    train_dataset = MiniModelDataset(X_train, y_train)
    val_dataset = MiniModelDataset(X_val, y_val)
    
    # Configuración
    config = TrainingConfig(
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-3,
        patience=10,
        verbose=True,
        checkpoint_dir="/tmp/checkpoints"
    )
    
    # Trainer
    trainer = MiniModelTrainer(
        model=model,
        config=config,
        loss_fn=nn.MSELoss(),
        metrics={'mae': mae, 'r2': r2_score}
    )
    
    # Entrenar
    print("Training...")
    history = trainer.train(train_dataset, val_dataset)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")

