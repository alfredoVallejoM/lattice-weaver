"""
Notebook 06: Model Explorer

Interactive exploration of trained models with inference testing and analysis.

Usage:
    python 06_Model_Explorer.py
"""

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("NOTEBOOK 06: MODEL EXPLORER")
print("=" * 80)
print()

# Imports
import sys
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, '/content/lattice-weaver')
sys.path.insert(0, str(Path.cwd()))

# Import utilities
try:
    from ml_utils import (
        ModelRegistry,
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

print()

# ============================================================================
# STEP 1: LOAD MODEL
# ============================================================================

print("📂 STEP 1: Load Model")
print("-" * 80)

model_info = registry.get_model_info(CONFIG['model_name'])

if model_info is None or not model_info['checkpoint_exists']:
    print(f"⚠️  Model checkpoint not found")
    print("   Please run Notebook 03 (Training) first")
    sys.exit(1)

# Load checkpoint
checkpoint_path = Path(model_info['checkpoint_path'])
checkpoint = torch.load(checkpoint_path, map_location=device)

print(f"✅ Checkpoint loaded from {checkpoint_path}")
print(f"   Epoch: {checkpoint['epoch']}")
print(f"   Val loss: {checkpoint['metrics']['val_loss']:.4f}")

# Create model
if MODELS_AVAILABLE:
    model = CostPredictor(input_dim=18, hidden_dim=32)
else:
    # Dummy model
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

print(f"✅ Model loaded and ready")

# Load normalization stats
stats_path = Path('/content/lattice-weaver/datasets/csp/normalization_stats.json')
if stats_path.exists():
    with open(stats_path, 'r') as f:
        norm_stats = json.load(f)
    print(f"✅ Normalization stats loaded")
else:
    print(f"⚠️  Normalization stats not found, using defaults")
    norm_stats = None

print()

# ============================================================================
# STEP 2: INTERACTIVE INFERENCE
# ============================================================================

print("🎮 STEP 2: Interactive Inference")
print("-" * 80)
print()

# Define test cases
test_cases = [
    {
        'name': 'Small CSP Problem',
        'features': [10, 15, 5, 0.3, 3, 0, 0, 0, 0, 0, 0.2, 3, 0.15, 2, 8, 1.5, 0.4, 0.2]
    },
    {
        'name': 'Medium CSP Problem',
        'features': [50, 80, 6, 0.5, 5, 5, 10, 50, 200, 100, 0.35, 5, 0.3, 3, 9, 2.0, 0.6, 0.3]
    },
    {
        'name': 'Large CSP Problem',
        'features': [100, 200, 8, 0.7, 8, 15, 50, 300, 1000, 500, 0.5, 8, 0.5, 4, 10, 2.5, 0.8, 0.4]
    }
]

feature_names = [
    'num_variables', 'num_constraints', 'avg_domain_size', 'constraint_density',
    'avg_degree', 'depth', 'backtracks', 'propagations', 'constraint_checks',
    'time_elapsed_ms', 'clustering_coefficient', 'diameter', 'density',
    'min_domain', 'max_domain', 'domain_std', 'constraint_tightness',
    'constraint_variance'
]

output_names = ['log(time_ms)', 'log(memory_mb)', 'log(nodes)']

for test_case in test_cases:
    print(f"🔍 Test Case: {test_case['name']}")
    print("-" * 40)
    
    # Prepare input
    features = torch.tensor(test_case['features'], dtype=torch.float32).unsqueeze(0)
    
    # Normalize if stats available
    if norm_stats:
        feature_mean = torch.tensor(norm_stats['feature_mean'])
        feature_std = torch.tensor(norm_stats['feature_std'])
        features_normalized = (features - feature_mean) / feature_std
    else:
        features_normalized = features
    
    # Inference
    with torch.no_grad():
        features_normalized = features_normalized.to(device)
        output = model(features_normalized)
        output = output.cpu()
    
    # Denormalize output if stats available
    if norm_stats:
        label_mean = torch.tensor(norm_stats['label_mean'])
        label_std = torch.tensor(norm_stats['label_std'])
        output_denormalized = output * label_std + label_mean
    else:
        output_denormalized = output
    
    # Convert from log scale
    predictions = torch.exp(output_denormalized).squeeze().numpy()
    
    # Display results
    print(f"\n  Input Features (selected):")
    print(f"    Variables: {test_case['features'][0]}")
    print(f"    Constraints: {test_case['features'][1]}")
    print(f"    Avg domain size: {test_case['features'][2]}")
    print(f"    Constraint density: {test_case['features'][3]:.2f}")
    
    print(f"\n  Predictions:")
    print(f"    Time: {predictions[0]:.2f} ms")
    print(f"    Memory: {predictions[1]:.2f} MB")
    print(f"    Nodes: {int(predictions[2]):,}")
    
    print()

print()

# ============================================================================
# STEP 3: SENSITIVITY ANALYSIS
# ============================================================================

print("📊 STEP 3: Sensitivity Analysis")
print("-" * 80)
print()

# Analyze sensitivity to number of variables
print("  Analyzing sensitivity to number of variables...")

base_features = [50, 80, 6, 0.5, 5, 5, 10, 50, 200, 100, 0.35, 5, 0.3, 3, 9, 2.0, 0.6, 0.3]
var_range = np.arange(10, 110, 10)
predictions_by_vars = []

for n_vars in var_range:
    features = base_features.copy()
    features[0] = n_vars  # num_variables
    features[1] = int(n_vars * 1.6)  # num_constraints (proportional)
    
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    if norm_stats:
        feature_mean = torch.tensor(norm_stats['feature_mean'])
        feature_std = torch.tensor(norm_stats['feature_std'])
        features_normalized = (features_tensor - feature_mean) / feature_std
    else:
        features_normalized = features_tensor
    
    with torch.no_grad():
        features_normalized = features_normalized.to(device)
        output = model(features_normalized)
        output = output.cpu()
    
    if norm_stats:
        label_mean = torch.tensor(norm_stats['label_mean'])
        label_std = torch.tensor(norm_stats['label_std'])
        output_denormalized = output * label_std + label_mean
    else:
        output_denormalized = output
    
    predictions = torch.exp(output_denormalized).squeeze().numpy()
    predictions_by_vars.append(predictions)

predictions_by_vars = np.array(predictions_by_vars)

print(f"✅ Sensitivity analysis completed")
print(f"\n  Results:")
print(f"  {'Vars':<8} {'Time (ms)':<12} {'Memory (MB)':<14} {'Nodes':<10}")
print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*10}")
for i, n_vars in enumerate(var_range):
    print(f"  {int(n_vars):<8} {predictions_by_vars[i, 0]:<12.2f} {predictions_by_vars[i, 1]:<14.2f} {int(predictions_by_vars[i, 2]):<10,}")

print()

# ============================================================================
# STEP 4: MODEL INSPECTION
# ============================================================================

print("🔬 STEP 4: Model Inspection")
print("-" * 80)

# Inspect model architecture
print("\n  Model Architecture:")
print(f"  {model}")

# Count parameters by layer
print("\n  Parameters by layer:")
total_params = 0
for name, param in model.named_parameters():
    num_params = param.numel()
    total_params += num_params
    print(f"    {name:<30} {num_params:>10,} params")

print(f"\n  Total parameters: {total_params:,}")

# Inspect weights
print("\n  Weight Statistics:")
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"    {name}:")
        print(f"      Mean: {param.data.mean().item():.6f}")
        print(f"      Std: {param.data.std().item():.6f}")
        print(f"      Min: {param.data.min().item():.6f}")
        print(f"      Max: {param.data.max().item():.6f}")

print()

# ============================================================================
# STEP 5: FEATURE IMPORTANCE (Approximation)
# ============================================================================

print("📈 STEP 5: Feature Importance (Approximation)")
print("-" * 80)
print()

print("  Computing feature importance via perturbation...")

# Use medium problem as baseline
baseline_features = [50, 80, 6, 0.5, 5, 5, 10, 50, 200, 100, 0.35, 5, 0.3, 3, 9, 2.0, 0.6, 0.3]
baseline_tensor = torch.tensor(baseline_features, dtype=torch.float32).unsqueeze(0)

if norm_stats:
    feature_mean = torch.tensor(norm_stats['feature_mean'])
    feature_std = torch.tensor(norm_stats['feature_std'])
    baseline_normalized = (baseline_tensor - feature_mean) / feature_std
else:
    baseline_normalized = baseline_tensor

with torch.no_grad():
    baseline_normalized = baseline_normalized.to(device)
    baseline_output = model(baseline_normalized)
    baseline_output = baseline_output.cpu()

# Perturb each feature and measure impact
importances = []

for i in range(len(baseline_features)):
    perturbed_features = baseline_features.copy()
    perturbed_features[i] *= 1.1  # 10% increase
    
    perturbed_tensor = torch.tensor(perturbed_features, dtype=torch.float32).unsqueeze(0)
    
    if norm_stats:
        perturbed_normalized = (perturbed_tensor - feature_mean) / feature_std
    else:
        perturbed_normalized = perturbed_tensor
    
    with torch.no_grad():
        perturbed_normalized = perturbed_normalized.to(device)
        perturbed_output = model(perturbed_normalized)
        perturbed_output = perturbed_output.cpu()
    
    # Measure change
    change = torch.abs(perturbed_output - baseline_output).mean().item()
    importances.append(change)

# Normalize importances
importances = np.array(importances)
importances = importances / importances.sum()

# Sort by importance
sorted_indices = np.argsort(importances)[::-1]

print(f"✅ Feature importance computed")
print(f"\n  Top 10 Most Important Features:")
print(f"  {'Rank':<6} {'Feature':<30} {'Importance':<12}")
print(f"  {'-'*6} {'-'*30} {'-'*12}")
for rank, idx in enumerate(sorted_indices[:10], 1):
    print(f"  {rank:<6} {feature_names[idx]:<30} {importances[idx]:.4f}")

print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"✅ Model exploration completed!")
print(f"   Model: {CONFIG['model_name']}")
print(f"   Total parameters: {total_params:,}")
print(f"   Test cases: {len(test_cases)} executed")
print(f"   Sensitivity analysis: ✓")
print(f"   Feature importance: ✓")
print()
print("Next step: Run Notebook 07 (Error Analysis) to analyze prediction errors")
print("=" * 80)

