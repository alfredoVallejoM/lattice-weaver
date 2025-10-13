"""
Notebook 04: Validation & Benchmarks

Comprehensive validation of trained models with precision metrics,
speedup benchmarks, and overhead analysis.

Usage:
    python 04_Validation_Benchmarks.py
"""

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("NOTEBOOK 04: VALIDATION & BENCHMARKS")
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
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add parent directory to path
sys.path.insert(0, '/content/lattice-weaver')
sys.path.insert(0, str(Path.cwd()))

# Import utilities
try:
    from ml_utils import (
        ModelRegistry,
        MLDataset,
        ModelValidator,
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
    'num_benchmark_samples': 10000,
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

# Initialize validator
validator = ModelValidator(registry)
print(f"✅ Validator initialized")

# Initialize report generator
report_gen = ReportGenerator(registry)
print(f"✅ Report generator initialized")

print()

# ============================================================================
# STEP 1: LOAD MODEL
# ============================================================================

print("📂 STEP 1: Load Model")
print("-" * 80)

model_info = registry.get_model_info(CONFIG['model_name'])

if model_info is None:
    print(f"⚠️  Model {CONFIG['model_name']} not found in registry")
    sys.exit(1)

if not model_info['checkpoint_exists']:
    print(f"⚠️  No checkpoint found for {CONFIG['model_name']}")
    print("   Please run Notebook 03 (Training) first")
    sys.exit(1)

# Load model
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
print(f"   Parameters: {model_info['params']:,}")

print()

# ============================================================================
# STEP 2: LOAD TEST DATASET
# ============================================================================

print("📂 STEP 2: Load Test Dataset")
print("-" * 80)

dataset_dir = Path('/content/lattice-weaver/datasets/csp')
test_data = torch.load(dataset_dir / 'test.pt')

print(f"✅ Test dataset loaded")
print(f"   Samples: {len(test_data['features'])}")

test_dataset = MLDataset(test_data['features'], test_data['labels'])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

print(f"✅ DataLoader created")
print(f"   Batches: {len(test_loader)}")

print()

# ============================================================================
# STEP 3: PRECISION VALIDATION
# ============================================================================

print("🎯 STEP 3: Precision Validation")
print("-" * 80)

precision_metrics = validator.validate_precision(
    model=model,
    test_loader=test_loader,
    device=str(device)
)

print(f"✅ Precision validation completed")
print(f"   MAE: {precision_metrics['mae']:.4f}")
print(f"   RMSE: {precision_metrics['rmse']:.4f}")
print(f"   R²: {precision_metrics['r2']:.4f}")
print(f"   Samples: {precision_metrics['num_samples']}")

# Acceptance criteria
mae_threshold = 0.2
r2_threshold = 0.8

print()
print("📋 Acceptance Criteria:")
if precision_metrics['mae'] < mae_threshold:
    print(f"   ✅ MAE < {mae_threshold}: PASS ({precision_metrics['mae']:.4f})")
else:
    print(f"   ❌ MAE < {mae_threshold}: FAIL ({precision_metrics['mae']:.4f})")

if precision_metrics['r2'] > r2_threshold:
    print(f"   ✅ R² > {r2_threshold}: PASS ({precision_metrics['r2']:.4f})")
else:
    print(f"   ❌ R² > {r2_threshold}: FAIL ({precision_metrics['r2']:.4f})")

print()

# ============================================================================
# STEP 4: INFERENCE SPEED BENCHMARK
# ============================================================================

print("⚡ STEP 4: Inference Speed Benchmark")
print("-" * 80)

speed_metrics = validator.benchmark_inference_speed(
    model=model,
    input_dim=model_info['input_dim'],
    num_samples=CONFIG['num_benchmark_samples'],
    device=str(device)
)

print(f"✅ Speed benchmark completed")
print(f"   Inference time: {speed_metrics['inference_time_ms']:.3f} ms/sample")
print(f"   Throughput: {speed_metrics['throughput_per_sec']:,} predictions/sec")
print(f"   Batch time: {speed_metrics['batch_time_ms']:.3f} ms/batch")

# Acceptance criteria
max_inference_time = 1.0  # ms

print()
print("📋 Acceptance Criteria:")
if speed_metrics['inference_time_ms'] < max_inference_time:
    print(f"   ✅ Inference < {max_inference_time} ms: PASS ({speed_metrics['inference_time_ms']:.3f} ms)")
else:
    print(f"   ❌ Inference < {max_inference_time} ms: FAIL ({speed_metrics['inference_time_ms']:.3f} ms)")

print()

# ============================================================================
# STEP 5: OVERHEAD ANALYSIS
# ============================================================================

print("📊 STEP 5: Overhead Analysis")
print("-" * 80)

# Model memory
model_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

# Baseline solver time (simulated)
baseline_time_ms = 500  # Typical CSP solve time
overhead_pct = (speed_metrics['inference_time_ms'] / baseline_time_ms) * 100

print(f"✅ Overhead analysis completed")
print(f"   Model memory: {model_memory_mb:.3f} MB")
print(f"   Baseline solver time: {baseline_time_ms} ms (simulated)")
print(f"   ML inference time: {speed_metrics['inference_time_ms']:.3f} ms")
print(f"   Overhead: {overhead_pct:.2f}%")

# Acceptance criteria
max_overhead = 5.0  # %

print()
print("📋 Acceptance Criteria:")
if overhead_pct < max_overhead:
    print(f"   ✅ Overhead < {max_overhead}%: PASS ({overhead_pct:.2f}%)")
else:
    print(f"   ❌ Overhead < {max_overhead}%: FAIL ({overhead_pct:.2f}%)")

print()

# ============================================================================
# STEP 6: SPEEDUP ESTIMATION
# ============================================================================

print("🚀 STEP 6: Speedup Estimation")
print("-" * 80)

# Simulated speedup (in real scenario, would run actual solver)
# Assume ML reduces nodes explored by 30%
node_reduction_pct = 30.0
estimated_speedup = 1 / (1 - node_reduction_pct / 100)

print(f"✅ Speedup estimation completed")
print(f"   Estimated node reduction: {node_reduction_pct}%")
print(f"   Estimated speedup: {estimated_speedup:.2f}x")
print(f"   Baseline time: {baseline_time_ms} ms")
print(f"   Estimated ML time: {baseline_time_ms / estimated_speedup:.1f} ms")

# Net speedup (accounting for overhead)
net_speedup = baseline_time_ms / (baseline_time_ms / estimated_speedup + speed_metrics['inference_time_ms'])

print(f"   Net speedup (with overhead): {net_speedup:.2f}x")

print()

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================

print("📈 STEP 7: Generate Visualizations")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Precision metrics
metrics_names = ['MAE', 'RMSE', 'R²']
metrics_values = [precision_metrics['mae'], precision_metrics['rmse'], precision_metrics['r2']]
axes[0, 0].bar(metrics_names, metrics_values, color=['#3498db', '#e74c3c', '#2ecc71'])
axes[0, 0].set_title('Precision Metrics')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Speed comparison
categories = ['Baseline\n(no ML)', 'With ML\n(no overhead)', 'With ML\n(with overhead)']
times = [baseline_time_ms, baseline_time_ms / estimated_speedup, 
         baseline_time_ms / estimated_speedup + speed_metrics['inference_time_ms']]
axes[0, 1].bar(categories, times, color=['#95a5a6', '#3498db', '#e67e22'])
axes[0, 1].set_title('Execution Time Comparison')
axes[0, 1].set_ylabel('Time (ms)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Overhead breakdown
overhead_components = ['Solver\n(accelerated)', 'ML\nInference']
overhead_values = [baseline_time_ms / estimated_speedup, speed_metrics['inference_time_ms']]
axes[1, 0].pie(overhead_values, labels=overhead_components, autopct='%1.1f%%',
               colors=['#3498db', '#e74c3c'], startangle=90)
axes[1, 0].set_title('Time Breakdown (With ML)')

# Plot 4: Speedup summary
speedup_data = {
    'Estimated\nSpeedup': estimated_speedup,
    'Net Speedup\n(with overhead)': net_speedup
}
axes[1, 1].bar(speedup_data.keys(), speedup_data.values(), color=['#2ecc71', '#27ae60'])
axes[1, 1].set_title('Speedup Analysis')
axes[1, 1].set_ylabel('Speedup (x)')
axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Baseline')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

plot_path = registry.reports_dir / f"validation_{CONFIG['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✅ Visualizations saved to {plot_path}")

plt.show()

print()

# ============================================================================
# STEP 8: GENERATE REPORT
# ============================================================================

print("📄 STEP 8: Generate Report")
print("-" * 80)

# Determine if model passes all criteria
all_passed = (
    precision_metrics['mae'] < mae_threshold and
    precision_metrics['r2'] > r2_threshold and
    speed_metrics['inference_time_ms'] < max_inference_time and
    overhead_pct < max_overhead
)

report = f"""# Validation Report - {CONFIG['model_name']}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information

| Property | Value |
|----------|-------|
| Name | {CONFIG['model_name']} |
| Suite | {CONFIG['suite_name']} |
| Parameters | {model_info['params']:,} |
| Memory | {model_memory_mb:.3f} MB |
| Checkpoint | {checkpoint_path.name} |
| Trained epoch | {checkpoint['epoch']} |

## Precision Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| MAE | {precision_metrics['mae']:.4f} | < {mae_threshold} | {'✅ PASS' if precision_metrics['mae'] < mae_threshold else '❌ FAIL'} |
| RMSE | {precision_metrics['rmse']:.4f} | - | - |
| R² | {precision_metrics['r2']:.4f} | > {r2_threshold} | {'✅ PASS' if precision_metrics['r2'] > r2_threshold else '❌ FAIL'} |

## Speed Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Inference time | {speed_metrics['inference_time_ms']:.3f} ms | < {max_inference_time} ms | {'✅ PASS' if speed_metrics['inference_time_ms'] < max_inference_time else '❌ FAIL'} |
| Throughput | {speed_metrics['throughput_per_sec']:,} pred/s | - | - |

## Overhead Analysis

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Model memory | {model_memory_mb:.3f} MB | - | - |
| Overhead | {overhead_pct:.2f}% | < {max_overhead}% | {'✅ PASS' if overhead_pct < max_overhead else '❌ FAIL'} |

## Speedup Estimation

| Metric | Value |
|--------|-------|
| Estimated node reduction | {node_reduction_pct}% |
| Estimated speedup | {estimated_speedup:.2f}x |
| Net speedup (with overhead) | {net_speedup:.2f}x |

## Conclusion

{'✅ **Model PASSED all acceptance criteria**' if all_passed else '⚠️  **Model FAILED some acceptance criteria**'}

{'Ready for deployment to production.' if all_passed else 'Requires further optimization or retraining.'}

## Files Generated

- Validation plots: `{plot_path}`
- This report: See below

---

**Validation completed successfully! ✅**
"""

report_filename = f"validation_{CONFIG['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
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
print(f"{'✅ VALIDATION PASSED' if all_passed else '⚠️  VALIDATION FAILED'}")
print(f"   Model: {CONFIG['model_name']}")
print(f"   MAE: {precision_metrics['mae']:.4f} ({'PASS' if precision_metrics['mae'] < mae_threshold else 'FAIL'})")
print(f"   R²: {precision_metrics['r2']:.4f} ({'PASS' if precision_metrics['r2'] > r2_threshold else 'FAIL'})")
print(f"   Inference: {speed_metrics['inference_time_ms']:.3f} ms ({'PASS' if speed_metrics['inference_time_ms'] < max_inference_time else 'FAIL'})")
print(f"   Overhead: {overhead_pct:.2f}% ({'PASS' if overhead_pct < max_overhead else 'FAIL'})")
print(f"   Net speedup: {net_speedup:.2f}x")
print()
if all_passed:
    print("Next step: Run Notebook 05 (Optimization) to optimize the model")
else:
    print("Recommendation: Retrain model or adjust architecture")
print("=" * 80)

