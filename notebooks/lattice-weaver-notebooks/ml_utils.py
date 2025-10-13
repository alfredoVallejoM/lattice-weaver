"""
ML Utilities for LatticeWeaver

Consolidated utilities for dataset generation, training, validation, and optimization.

Author: LatticeWeaver ML Team
Version: 1.0
Date: 2025-10-13
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    model_name: str = "CostPredictor"
    suite_name: str = "costs_memoization"
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    num_epochs: int = 100
    early_stopping_patience: int = 10
    lr_scheduler: str = "reduce_on_plateau"
    dropout: float = 0.1
    label_smoothing: float = 0.0
    save_every_n_epochs: int = 5
    keep_best_n: int = 3
    log_every_n_steps: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    seed: int = 42

# ============================================================================
# MODEL REGISTRY
# ============================================================================

class ModelRegistry:
    """Central registry for all models."""
    
    def __init__(self, base_path="/content/lattice-weaver"):
        self.base_path = Path(base_path)
        self.models_dir = self.base_path / "models"
        self.logs_dir = self.base_path / "logs"
        self.reports_dir = self.base_path / "reports"
        self.datasets_dir = self.base_path / "datasets"
        
        # Create directories
        for dir_path in [self.models_dir, self.logs_dir, self.reports_dir, self.datasets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Model definitions
        self.model_definitions = {
            "costs_memoization": [
                {"name": "CostPredictor", "params": 3395, "input_dim": 18, "output_dim": 3},
                {"name": "MemoizationGuide", "params": 1345, "input_dim": 18, "output_dim": 1},
                {"name": "CacheValueEstimator", "params": 1153, "input_dim": 18, "output_dim": 1},
                {"name": "ComputationReusabilityScorer", "params": 705, "input_dim": 18, "output_dim": 1},
                {"name": "DynamicCacheManager", "params": 60547, "input_dim": 18, "output_dim": 3},
                {"name": "WorkloadPredictor", "params": 56400, "input_dim": 18, "output_dim": 18},
            ]
        }
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a model."""
        for suite_name, models in self.model_definitions.items():
            for model in models:
                if model["name"] == model_name:
                    checkpoint_path = self.models_dir / suite_name / f"{model_name}.pt"
                    onnx_path = self.models_dir / suite_name / f"{model_name}.onnx"
                    
                    return {
                        "suite": suite_name,
                        "name": model_name,
                        "params": model["params"],
                        "input_dim": model["input_dim"],
                        "output_dim": model["output_dim"],
                        "checkpoint_exists": checkpoint_path.exists(),
                        "onnx_exists": onnx_path.exists(),
                        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
                        "onnx_path": str(onnx_path) if onnx_path.exists() else None,
                    }
        return None
    
    def get_status_summary(self) -> Dict:
        """Get summary of all models."""
        total_models = sum(len(models) for models in self.model_definitions.values())
        trained = self._count_trained_models()
        optimized = self._count_optimized_models()
        
        return {
            "total": total_models,
            "implemented": 6,  # Suite 1
            "trained": trained,
            "optimized": optimized,
            "progress_pct": (6 / total_models) * 100
        }
    
    def _count_trained_models(self) -> int:
        """Count models with checkpoints."""
        if not self.models_dir.exists():
            return 0
        return len(list(self.models_dir.glob("**/*.pt")))
    
    def _count_optimized_models(self) -> int:
        """Count optimized models (ONNX)."""
        if not self.models_dir.exists():
            return 0
        return len(list(self.models_dir.glob("**/*.onnx")))

# ============================================================================
# DATASET GENERATION
# ============================================================================

class SyntheticDatasetGenerator:
    """Generate synthetic datasets for training."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_csp_dataset(self, num_samples: int = 1000, 
                            size: str = 'medium') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic CSP dataset.
        
        Args:
            num_samples: Number of samples to generate
            size: 'small', 'medium', or 'large'
        
        Returns:
            (features, labels) tensors
        """
        features = []
        labels = []
        
        for _ in range(num_samples):
            # Generate problem parameters
            if size == 'small':
                n_vars = np.random.randint(5, 20)
                n_constraints = np.random.randint(5, 30)
            elif size == 'medium':
                n_vars = np.random.randint(20, 60)
                n_constraints = np.random.randint(30, 100)
            else:  # large
                n_vars = np.random.randint(60, 120)
                n_constraints = np.random.randint(100, 250)
            
            avg_domain_size = np.random.uniform(2, 10)
            constraint_density = n_constraints / (n_vars * (n_vars - 1) / 2)
            avg_degree = np.random.uniform(2, min(10, n_vars - 1))
            
            # Generate 18 features (matching CSPFeatureExtractor)
            feature_vector = [
                n_vars,
                n_constraints,
                avg_domain_size,
                constraint_density,
                avg_degree,
                np.random.randint(0, 10),  # depth
                np.random.randint(0, 100),  # backtracks
                np.random.randint(0, 500),  # propagations
                np.random.randint(0, 1000),  # constraint_checks
                np.random.uniform(0, 1000),  # time_elapsed_ms
                # Graph features
                np.random.uniform(0, 1),  # clustering_coefficient
                np.random.randint(1, n_vars),  # diameter
                np.random.uniform(0, 1),  # density
                # Domain features
                np.random.uniform(2, 10),  # min_domain
                np.random.uniform(2, 10),  # max_domain
                np.random.uniform(0, 1),  # domain_std
                # Constraint features
                np.random.uniform(0, 1),  # constraint_tightness
                np.random.uniform(0, 1),  # constraint_variance
            ]
            
            # Generate labels (log-scaled for numerical stability)
            time_ms = n_vars * n_constraints * 0.1 * np.random.lognormal(0, 0.5)
            memory_mb = n_vars * avg_domain_size * 0.01 * np.random.lognormal(0, 0.3)
            nodes = n_vars ** 2 * np.random.lognormal(0, 0.7)
            
            label_vector = [
                np.log(time_ms + 1),
                np.log(memory_mb + 1),
                np.log(nodes + 1)
            ]
            
            features.append(feature_vector)
            labels.append(label_vector)
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
    
    def save_dataset(self, features: torch.Tensor, labels: torch.Tensor, 
                    path: Path, split: str = 'train'):
        """Save dataset to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = {
            'features': features,
            'labels': labels,
            'metadata': {
                'num_samples': len(features),
                'feature_dim': features.shape[1],
                'label_dim': labels.shape[1],
                'split': split,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        torch.save(dataset, path)
        print(f"✅ Dataset saved to {path}")

# ============================================================================
# DATASET CLASS
# ============================================================================

class MLDataset(Dataset):
    """PyTorch Dataset for mini-models."""
    
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ============================================================================
# TRAINER
# ============================================================================

class AutomatedTrainer:
    """Automated training with monitoring and checkpointing."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.metrics_history = []
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
             val_loader: DataLoader, config: TrainingConfig) -> Dict:
        """
        Train model with given configuration.
        
        Returns:
            Dict with training results
        """
        # Setup
        device = torch.device(config.device)
        model = model.to(device)
        
        # Optimizer
        if config.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                                  weight_decay=config.weight_decay)
        elif config.optimizer == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                                   weight_decay=config.weight_decay)
        else:  # sgd
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay, momentum=0.9)
        
        # Scheduler
        if config.lr_scheduler == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                             factor=0.5, patience=5)
        elif config.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        else:  # step
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # Loss
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        self.metrics_history = []
        
        print(f"🚀 Starting training: {config.model_name}")
        print(f"   Device: {device}")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Learning rate: {config.learning_rate}")
        print()
        
        for epoch in range(config.num_epochs):
            # Train
            train_metrics = self._train_epoch(model, train_loader, optimizer, 
                                             criterion, device, config)
            
            # Validate
            val_metrics = self._validate_epoch(model, val_loader, criterion, device)
            
            # Scheduler step
            if config.lr_scheduler == "reduce_on_plateau":
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
            
            # Metrics
            current_lr = optimizer.param_groups[0]['lr']
            metrics = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'learning_rate': current_lr,
                'time': train_metrics['time']
            }
            self.metrics_history.append(metrics)
            
            # Logging
            if epoch % config.log_every_n_steps == 0:
                print(f"Epoch {epoch:3d}/{config.num_epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"LR: {current_lr:.6f}")
            
            # Checkpointing
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self._save_checkpoint(model, optimizer, epoch, metrics, config, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % config.save_every_n_epochs == 0:
                self._save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False)
            
            # Early stopping
            if patience_counter >= config.early_stopping_patience:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break
        
        print(f"\n✅ Training completed!")
        print(f"   Best val loss: {best_val_loss:.4f}")
        print(f"   Total epochs: {epoch + 1}")
        
        return {
            'best_val_loss': best_val_loss,
            'final_epoch': epoch,
            'metrics_history': self.metrics_history
        }
    
    def _train_epoch(self, model, train_loader, optimizer, criterion, device, config):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        return {'loss': avg_loss, 'time': epoch_time}
    
    def _validate_epoch(self, model, val_loader, criterion, device):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return {'loss': avg_loss}
    
    def _save_checkpoint(self, model, optimizer, epoch, metrics, config, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': asdict(config),
            'timestamp': datetime.now().isoformat()
        }
        
        # Create directory
        save_dir = self.registry.models_dir / config.suite_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save
        if is_best:
            path = save_dir / f"{config.model_name}_best.pt"
            torch.save(checkpoint, path)
        else:
            path = save_dir / f"{config.model_name}_epoch{epoch:03d}.pt"
            torch.save(checkpoint, path)

# ============================================================================
# VALIDATOR
# ============================================================================

class ModelValidator:
    """Validate trained models."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def validate_precision(self, model: nn.Module, test_loader: DataLoader, 
                          device: str = 'cpu') -> Dict:
        """
        Validate model precision.
        
        Returns:
            Dict with precision metrics
        """
        model.eval()
        device = torch.device(device)
        model = model.to(device)
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                predictions.append(outputs.cpu())
                targets.append(labels.cpu())
        
        predictions = torch.cat(predictions).numpy()
        targets = torch.cat(targets).numpy()
        
        # Metrics
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # R² per output dimension
        r2_scores = []
        for i in range(predictions.shape[1]):
            ss_res = np.sum((targets[:, i] - predictions[:, i]) ** 2)
            ss_tot = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            r2_scores.append(r2)
        
        r2 = np.mean(r2_scores)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'num_samples': len(predictions)
        }
    
    def benchmark_inference_speed(self, model: nn.Module, input_dim: int, 
                                  num_samples: int = 10000, device: str = 'cpu') -> Dict:
        """
        Benchmark inference speed.
        
        Returns:
            Dict with speed metrics
        """
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        # Generate dummy inputs
        inputs = torch.randn(num_samples, input_dim).to(device)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(100):
                _ = model(inputs[:32])
        
        # Benchmark
        times = []
        with torch.no_grad():
            for i in range(0, num_samples, 32):
                batch = inputs[i:i+32]
                
                start = time.perf_counter()
                _ = model(batch)
                end = time.perf_counter()
                
                times.append(end - start)
        
        avg_batch_time = np.mean(times)
        avg_per_sample = avg_batch_time / 32
        throughput = 1 / avg_per_sample
        
        return {
            'inference_time_ms': avg_per_sample * 1000,
            'throughput_per_sec': int(throughput),
            'batch_time_ms': avg_batch_time * 1000
        }

# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generate reports in various formats."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def generate_training_report(self, config: TrainingConfig, 
                                 results: Dict, metrics_history: List[Dict]) -> str:
        """Generate training report in Markdown."""
        report = f"""# Training Report - {config.model_name}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | {config.model_name} |
| Suite | {config.suite_name} |
| Batch size | {config.batch_size} |
| Learning rate | {config.learning_rate} |
| Optimizer | {config.optimizer} |
| Epochs | {config.num_epochs} |
| Device | {config.device} |

## Results

| Metric | Value |
|--------|-------|
| Best val loss | {results['best_val_loss']:.4f} |
| Final epoch | {results['final_epoch']} |
| Early stopped | {'Yes' if results['final_epoch'] < config.num_epochs - 1 else 'No'} |

## Training Metrics

Final metrics:
- Train loss: {metrics_history[-1]['train_loss']:.4f}
- Val loss: {metrics_history[-1]['val_loss']:.4f}
- Learning rate: {metrics_history[-1]['learning_rate']:.6f}

## Conclusion

✅ Training completed successfully
"""
        return report
    
    def save_report(self, report: str, filename: str):
        """Save report to file."""
        path = self.registry.reports_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved to {path}")

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(metrics_history: List[Dict], save_path: Optional[Path] = None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = [m['epoch'] for m in metrics_history]
    train_loss = [m['train_loss'] for m in metrics_history]
    val_loss = [m['val_loss'] for m in metrics_history]
    lr = [m['learning_rate'] for m in metrics_history]
    
    # Loss curves
    axes[0].plot(epochs, train_loss, label='Train', linewidth=2)
    axes[0].plot(epochs, val_loss, label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1].plot(epochs, lr, linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🎮 GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    return device

def print_model_summary(model: nn.Module):
    """Print model summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Summary:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024:.2f} KB")
    print()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("✅ ML Utilities loaded successfully")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")

