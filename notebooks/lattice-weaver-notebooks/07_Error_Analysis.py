"""
Notebook 07: Error Analysis

Comprehensive error analysis with pattern detection and improvement suggestions.

Usage:
    python 07_Error_Analysis.py
"""

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("NOTEBOOK 07: ERROR ANALYSIS")
print("=" * 80)
print()

# Imports
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, '/content/lattice-weaver')
sys.path.insert(0, str(Path.cwd()))

# Import utilities
try:
    from ml_utils import (
        ModelRegistry,
        MLDataset,
        ReportGenerator,
        set_seed,
        get_device
    )
    print("✅ Utilities imported successfully")
except ImportError as e:
    print(f"⚠️  Failed to import utilities: {e}")
    sys.exit(1)

# Import mini-models
try:
    sys.path.insert(0, '/content/lattice-weaver')
    from lattice_weaver.ml.mini_nets.costs_memoization import CostPredictor
    MODELS_AVAILABLE = True
    print("✅ Mini-models imported")
except ImportError:
    MODELS_AVAILABLE = False
    print("⚠️  Mini-models not available")

print()

# ============================================================================
# CONFIGURATION
# ============================================================================

print("📋 Configuration")
print("-" * 80)

CONFIG = {
    'model_name': 'CostPredictor',
    'suite_name': 'costs_memoization',
    'batch_size': 32,
    'error_threshold': 0.5,  # MAE threshold for "large errors"
    'seed': 42
}

for key, value in CONFIG.items():
    print(f"  {key}: {value}")

print()

# ============================================================================
# INITIALIZATION
# ============================================================================

print("🔧 Initialization")
print("-" * 80)

# Set seed
set_seed(CONFIG['seed'])
print(f"✅ Random seed set to {CONFIG['seed']}")

# Get device
device = get_device()

# Initialize registry
registry = ModelRegistry()
print(f"✅ Model registry initialized")

# Initialize report generator
report_gen = ReportGenerator(registry)
print(f"✅ Report generator initialized")

print()

# ============================================================================
# STEP 1: LOAD MODEL AND DATA
# ============================================================================

print("📂 STEP 1: Load Model and Data")
print("-" * 80)

# Load model
model_info = registry.get_model_info(CONFIG['model_name'])

if model_info is None or not model_info['checkpoint_exists']:
    print(f"⚠️  Model checkpoint not found")
    sys.exit(1)

checkpoint_path = Path(model_info['checkpoint_path'])
checkpoint = torch.load(checkpoint_path, map_location=device)

if MODELS_AVAILABLE:
    model = CostPredictor(input_dim=18, hidden_dim=32)
else:
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
    model = DummyModel()

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to(device)

print(f"✅ Model loaded")

# Load test data
dataset_dir = Path('/content/lattice-weaver/datasets/csp')
test_data = torch.load(dataset_dir / 'test.pt')

test_dataset = MLDataset(test_data['features'], test_data['labels'])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

print(f"✅ Test data loaded ({len(test_dataset)} samples)")

print()

# ============================================================================
# STEP 2: COLLECT PREDICTION ERRORS
# ============================================================================

print("🔍 STEP 2: Collect Prediction Errors")
print("-" * 80)

all_predictions = []
all_targets = []
all_features = []
all_errors = []

model.eval()
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        
        # Calculate errors
        errors = torch.abs(outputs - labels)
        
        all_predictions.append(outputs.cpu())
        all_targets.append(labels.cpu())
        all_features.append(features.cpu())
        all_errors.append(errors.cpu())

all_predictions = torch.cat(all_predictions).numpy()
all_targets = torch.cat(all_targets).numpy()
all_features = torch.cat(all_features).numpy()
all_errors = torch.cat(all_errors).numpy()

# Calculate per-sample error (mean across outputs)
per_sample_error = all_errors.mean(axis=1)

print(f"✅ Predictions collected")
print(f"   Total samples: {len(all_predictions)}")
print(f"   Mean error: {per_sample_error.mean():.4f}")
print(f"   Median error: {np.median(per_sample_error):.4f}")
print(f"   95th percentile: {np.percentile(per_sample_error, 95):.4f}")

# Identify large errors
large_error_mask = per_sample_error > CONFIG['error_threshold']
num_large_errors = large_error_mask.sum()

print(f"\n  Large errors (> {CONFIG['error_threshold']}):")
print(f"   Count: {num_large_errors} ({num_large_errors/len(all_predictions)*100:.1f}%)")

print()

# ============================================================================
# STEP 3: ERROR DISTRIBUTION ANALYSIS
# ============================================================================

print("📊 STEP 3: Error Distribution Analysis")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Error histogram
axes[0, 0].hist(per_sample_error, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(CONFIG['error_threshold'], color='r', linestyle='--', 
                   label=f'Threshold ({CONFIG["error_threshold"]})')
axes[0, 0].set_xlabel('Error (MAE)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Error Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Error by problem size (num_variables)
num_vars = all_features[:, 0]  # First feature is num_variables
axes[0, 1].scatter(num_vars, per_sample_error, alpha=0.5, s=10)
axes[0, 1].set_xlabel('Number of Variables')
axes[0, 1].set_ylabel('Error (MAE)')
axes[0, 1].set_title('Error vs Problem Size')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Prediction vs Target (for first output dimension)
axes[1, 0].scatter(all_targets[:, 0], all_predictions[:, 0], alpha=0.3, s=10)
axes[1, 0].plot([all_targets[:, 0].min(), all_targets[:, 0].max()],
                [all_targets[:, 0].min(), all_targets[:, 0].max()],
                'r--', label='Perfect prediction')
axes[1, 0].set_xlabel('Target (log time)')
axes[1, 0].set_ylabel('Prediction (log time)')
axes[1, 0].set_title('Prediction vs Target (Time)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Error by output dimension
error_by_dim = all_errors.mean(axis=0)
output_names = ['log(time)', 'log(memory)', 'log(nodes)']
axes[1, 1].bar(output_names, error_by_dim, color=['#3498db', '#e74c3c', '#2ecc71'])
axes[1, 1].set_ylabel('Mean Absolute Error')
axes[1, 1].set_title('Error by Output Dimension')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

plot_path = registry.reports_dir / f"error_analysis_{CONFIG['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✅ Plots saved to {plot_path}")

plt.show()

print()

# ============================================================================
# STEP 4: ERROR PATTERN ANALYSIS
# ============================================================================

print("🔬 STEP 4: Error Pattern Analysis")
print("-" * 80)

# Analyze errors by problem size
small_mask = all_features[:, 0] < 20
medium_mask = (all_features[:, 0] >= 20) & (all_features[:, 0] < 60)
large_mask = all_features[:, 0] >= 60

print(f"\n  Error by problem size:")
print(f"  {'Size':<10} {'Count':<8} {'Mean Error':<12} {'Error Rate':<12}")
print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*12}")

for mask, name in [(small_mask, 'Small'), (medium_mask, 'Medium'), (large_mask, 'Large')]:
    if mask.sum() > 0:
        mean_error = per_sample_error[mask].mean()
        error_rate = (per_sample_error[mask] > CONFIG['error_threshold']).mean() * 100
        print(f"  {name:<10} {mask.sum():<8} {mean_error:<12.4f} {error_rate:<12.1f}%")

print()

# ============================================================================
# STEP 5: TOP WORST CASES
# ============================================================================

print("❌ STEP 5: Top 10 Worst Cases")
print("-" * 80)

# Find worst cases
worst_indices = np.argsort(per_sample_error)[::-1][:10]

print(f"\n  {'Rank':<6} {'Error':<10} {'Vars':<8} {'Constraints':<12} {'Predicted Time':<15} {'Actual Time':<15}")
print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*12} {'-'*15} {'-'*15}")

for rank, idx in enumerate(worst_indices, 1):
    error = per_sample_error[idx]
    n_vars = int(all_features[idx, 0])
    n_constraints = int(all_features[idx, 1])
    pred_time = np.exp(all_predictions[idx, 0])
    actual_time = np.exp(all_targets[idx, 0])
    
    print(f"  {rank:<6} {error:<10.4f} {n_vars:<8} {n_constraints:<12} {pred_time:<15.2f} {actual_time:<15.2f}")

print()

# ============================================================================
# STEP 6: IMPROVEMENT SUGGESTIONS
# ============================================================================

print("💡 STEP 6: Improvement Suggestions")
print("-" * 80)

suggestions = []

# Suggestion 1: Data coverage
large_problem_errors = per_sample_error[large_mask]
if len(large_problem_errors) > 0 and large_problem_errors.mean() > per_sample_error.mean() * 1.5:
    suggestions.append({
        'priority': 'HIGH',
        'type': 'data_collection',
        'message': 'Collect more training data for large problems',
        'details': f'Large problems have {large_problem_errors.mean() / per_sample_error.mean():.2f}x higher error'
    })

# Suggestion 2: Model capacity
if per_sample_error.mean() > 0.15:
    suggestions.append({
        'priority': 'MEDIUM',
        'type': 'model_capacity',
        'message': 'Consider increasing model capacity',
        'details': f'Mean error ({per_sample_error.mean():.4f}) is relatively high'
    })

# Suggestion 3: Feature engineering
if all_errors[:, 0].mean() > all_errors[:, 1:].mean() * 1.5:
    suggestions.append({
        'priority': 'MEDIUM',
        'type': 'feature_engineering',
        'message': 'Add features related to time prediction',
        'details': 'Time prediction has higher error than other outputs'
    })

# Suggestion 4: Hyperparameter tuning
if num_large_errors > len(all_predictions) * 0.1:
    suggestions.append({
        'priority': 'LOW',
        'type': 'hyperparameters',
        'message': 'Try different hyperparameters',
        'details': f'{num_large_errors/len(all_predictions)*100:.1f}% of samples have large errors'
    })

print(f"\n  Found {len(suggestions)} suggestions:\n")

for i, sug in enumerate(suggestions, 1):
    print(f"  {i}. [{sug['priority']}] {sug['message']}")
    print(f"     {sug['details']}")
    print()

# ============================================================================
# STEP 7: GENERATE REPORT
# ============================================================================

print("📄 STEP 7: Generate Report")
print("-" * 80)

report = f"""# Error Analysis Report - {CONFIG['model_name']}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total samples | {len(all_predictions)} |
| Mean error | {per_sample_error.mean():.4f} |
| Median error | {np.median(per_sample_error):.4f} |
| 95th percentile | {np.percentile(per_sample_error, 95):.4f} |
| Large errors (> {CONFIG['error_threshold']}) | {num_large_errors} ({num_large_errors/len(all_predictions)*100:.1f}%) |

## Error by Problem Size

| Size | Count | Mean Error | Error Rate |
|------|-------|------------|------------|
"""

for mask, name in [(small_mask, 'Small'), (medium_mask, 'Medium'), (large_mask, 'Large')]:
    if mask.sum() > 0:
        mean_error = per_sample_error[mask].mean()
        error_rate = (per_sample_error[mask] > CONFIG['error_threshold']).mean() * 100
        report += f"| {name} | {mask.sum()} | {mean_error:.4f} | {error_rate:.1f}% |\n"

report += f"""
## Error by Output Dimension

| Output | Mean Error |
|--------|------------|
| log(time) | {all_errors[:, 0].mean():.4f} |
| log(memory) | {all_errors[:, 1].mean():.4f} |
| log(nodes) | {all_errors[:, 2].mean():.4f} |

## Top 10 Worst Cases

| Rank | Error | Variables | Constraints | Predicted Time | Actual Time |
|------|-------|-----------|-------------|----------------|-------------|
"""

for rank, idx in enumerate(worst_indices, 1):
    error = per_sample_error[idx]
    n_vars = int(all_features[idx, 0])
    n_constraints = int(all_features[idx, 1])
    pred_time = np.exp(all_predictions[idx, 0])
    actual_time = np.exp(all_targets[idx, 0])
    report += f"| {rank} | {error:.4f} | {n_vars} | {n_constraints} | {pred_time:.2f} ms | {actual_time:.2f} ms |\n"

report += """
## Improvement Suggestions

"""

for i, sug in enumerate(suggestions, 1):
    report += f"### {i}. [{sug['priority']}] {sug['message']}\n\n"
    report += f"{sug['details']}\n\n"

report += f"""
## Files Generated

- Error analysis plots: `{plot_path}`
- This report: See below

---

**Error analysis completed successfully! ✅**
"""

report_filename = f"error_analysis_{CONFIG['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
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
print(f"✅ Error analysis completed!")
print(f"   Model: {CONFIG['model_name']}")
print(f"   Mean error: {per_sample_error.mean():.4f}")
print(f"   Large errors: {num_large_errors} ({num_large_errors/len(all_predictions)*100:.1f}%)")
print(f"   Suggestions: {len(suggestions)}")
print()
print("All notebooks completed! 🎉")
print("=" * 80)

