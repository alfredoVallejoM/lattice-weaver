"""
Notebook 03: Training

Automated training of mini-models with monitoring, checkpointing, and early stopping.

Usage:
    python 03_Training.py
"""

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("NOTEBOOK 03: TRAINING")
print("=" * 80)
print()

# Imports
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, '/content/lattice-weaver')
sys.path.insert(0, str(Path.cwd()))

# Import utilities
try:
    from ml_utils import (
        ModelRegistry,
        TrainingConfig,
        MLDataset,
        AutomatedTrainer,
        ReportGenerator,
        set_seed,
        get_device,
        print_model_summary,
        plot_training_curves
    )
    print("✅ Utilities imported successfully")
except ImportError as e:
    print(f"⚠️  Failed to import utilities: {e}")
    sys.exit(1)

# Import mini-models
try:
    sys.path.insert(0, '/content/lattice-weaver')
    from lattice_weaver.ml.mini_nets.costs_memoization import (
        CostPredictor,
        MemoizationGuide,
        CacheValueEstimator
    )
    MODELS_AVAILABLE = True
    print("✅ Mini-models imported")
except ImportError:
    MODELS_AVAILABLE = False
    print("⚠️  Mini-models not available, using dummy model")

print()

# ============================================================================
# CONFIGURATION
# ============================================================================

print("📋 Configuration")
print("-" * 80)

# Training configuration
config = TrainingConfig(
    model_name="CostPredictor",
    suite_name="costs_memoization",
    batch_size=32,
    num_workers=2,
    learning_rate=1e-3,
    weight_decay=1e-5,
    optimizer="adam",
    num_epochs=100,
    early_stopping_patience=10,
    lr_scheduler="reduce_on_plateau",
    dropout=0.1,
    save_every_n_epochs=10,
    keep_best_n=3,
    log_every_n_steps=5,
    seed=42
)

print(f"  Model: {config.model_name}")
print(f"  Suite: {config.suite_name}")
print(f"  Batch size: {config.batch_size}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Optimizer: {config.optimizer}")
print(f"  Max epochs: {config.num_epochs}")
print(f"  Early stopping patience: {config.early_stopping_patience}")
print(f"  Device: {config.device}")
print()

# ============================================================================
# INITIALIZATION
# ============================================================================

print("🔧 Initialization")
print("-" * 80)

# Set seed
set_seed(config.seed)
print(f"✅ Random seed set to {config.seed}")

# Get device
device = get_device()
config.device = str(device)

# Initialize registry
registry = ModelRegistry()
print(f"✅ Model registry initialized")

# Initialize trainer
trainer = AutomatedTrainer(registry)
print(f"✅ Trainer initialized")

# Initialize report generator
report_gen = ReportGenerator(registry)
print(f"✅ Report generator initialized")

print()

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================

print("📂 STEP 1: Load Dataset")
print("-" * 80)

dataset_dir = Path('/content/lattice-weaver/datasets/csp')

# Check if dataset exists
if not dataset_dir.exists():
    print(f"⚠️  Dataset not found at {dataset_dir}")
    print("   Please run Notebook 02 (Dataset Generation) first")
    sys.exit(1)

# Load datasets
print(f"  Loading from {dataset_dir}...")

train_data = torch.load(dataset_dir / 'train.pt')
val_data = torch.load(dataset_dir / 'val.pt')
test_data = torch.load(dataset_dir / 'test.pt')

print(f"✅ Datasets loaded")
print(f"   Train: {len(train_data['features'])} samples")
print(f"   Val: {len(val_data['features'])} samples")
print(f"   Test: {len(test_data['features'])} samples")

# Create PyTorch datasets
train_dataset = MLDataset(train_data['features'], train_data['labels'])
val_dataset = MLDataset(val_data['features'], val_data['labels'])
test_dataset = MLDataset(test_data['features'], test_data['labels'])

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True if device.type == 'cuda' else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=True if device.type == 'cuda' else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)

print(f"✅ DataLoaders created")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")

print()

# ============================================================================
# STEP 2: CREATE MODEL
# ============================================================================

print("🏗️  STEP 2: Create Model")
print("-" * 80)

if MODELS_AVAILABLE:
    if config.model_name == "CostPredictor":
        model = CostPredictor(input_dim=18, hidden_dim=32, dropout=config.dropout)
    elif config.model_name == "MemoizationGuide":
        model = MemoizationGuide(input_dim=18, hidden_dim=24, dropout=config.dropout)
    elif config.model_name == "CacheValueEstimator":
        model = CacheValueEstimator(input_dim=18, hidden_dim=20, dropout=config.dropout)
    else:
        print(f"⚠️  Unknown model: {config.model_name}")
        sys.exit(1)
else:
    # Dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self, input_dim=18, output_dim=3):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )
        
        def forward(self, x):
            return self.fc(x)
    
    model = DummyModel(input_dim=18, output_dim=3)

print(f"✅ Model created: {config.model_name}")
print_model_summary(model)

print()

# ============================================================================
# STEP 3: TRAIN MODEL
# ============================================================================

print("🚀 STEP 3: Train Model")
print("-" * 80)
print()

start_time = time.time()

results = trainer.train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

training_time = time.time() - start_time

print()
print(f"✅ Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")
print(f"   Best val loss: {results['best_val_loss']:.4f}")
print(f"   Final epoch: {results['final_epoch']}")

print()

# ============================================================================
# STEP 4: PLOT TRAINING CURVES
# ============================================================================

print("📊 STEP 4: Plot Training Curves")
print("-" * 80)

plot_path = registry.reports_dir / f"training_{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_curves.png"
plot_training_curves(trainer.metrics_history, save_path=plot_path)

print()

# ============================================================================
# STEP 5: EVALUATE ON TEST SET
# ============================================================================

print("✅ STEP 5: Evaluate on Test Set")
print("-" * 80)

# Load best model
best_checkpoint_path = registry.models_dir / config.suite_name / f"{config.model_name}_best.pt"

if best_checkpoint_path.exists():
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded best model from epoch {checkpoint['epoch']}")
else:
    print(f"⚠️  Best checkpoint not found, using current model")

# Evaluate
model.eval()
model = model.to(device)

test_loss = 0
criterion = nn.MSELoss()

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

test_loss /= len(test_loader)

print(f"✅ Test evaluation completed")
print(f"   Test loss: {test_loss:.4f}")

print()

# ============================================================================
# STEP 6: GENERATE REPORT
# ============================================================================

print("📄 STEP 6: Generate Report")
print("-" * 80)

# Add test loss to results
results['test_loss'] = test_loss
results['training_time'] = training_time

report = report_gen.generate_training_report(
    config=config,
    results=results,
    metrics_history=trainer.metrics_history
)

# Add test results
report += f"""
## Test Results

| Metric | Value |
|--------|-------|
| Test loss | {test_loss:.4f} |

## Training Time

| Metric | Value |
|--------|-------|
| Total time | {training_time:.2f}s ({training_time/60:.2f} min) |
| Avg time per epoch | {training_time / (results['final_epoch'] + 1):.2f}s |

## Files Generated

- Best model: `{best_checkpoint_path}`
- Training curves: `{plot_path}`
- This report: See below

---

**Training completed successfully! ✅**
"""

report_filename = f"training_{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
report_gen.save_report(report, report_filename)

print()
print(report)

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"✅ Training completed successfully!")
print(f"   Model: {config.model_name}")
print(f"   Best val loss: {results['best_val_loss']:.4f}")
print(f"   Test loss: {test_loss:.4f}")
print(f"   Training time: {training_time/60:.2f} min")
print(f"   Model saved to: {best_checkpoint_path}")
print()
print("Next step: Run Notebook 04 (Validation & Benchmarks) to validate the model")
print("=" * 80)

