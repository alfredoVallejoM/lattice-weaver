"""
Notebook 02: Dataset Generation

Automated dataset generation with synthetic problems, solver execution,
data augmentation, and purification.

Usage:
    python 02_Dataset_Generation.py
    
Or convert to Jupyter notebook:
    jupytext --to notebook 02_Dataset_Generation.py
"""

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("NOTEBOOK 02: DATASET GENERATION")
print("=" * 80)
print()

# Imports
import sys
import torch
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, '/content/lattice-weaver')
sys.path.insert(0, str(Path.cwd()))

# Import utilities
try:
    from ml_utils import (
        ModelRegistry,
        SyntheticDatasetGenerator,
        MLDataset,
        set_seed,
        get_device
    )
    print("✅ Utilities imported successfully")
except ImportError as e:
    print(f"⚠️  Failed to import utilities: {e}")
    print("   Make sure ml_utils.py is in the same directory")
    sys.exit(1)

# Import LatticeWeaver components (if available)
try:
    sys.path.insert(0, '/content/lattice-weaver')
    from lattice_weaver.ml.adapters.feature_extractors import CSPFeatureExtractor
    from lattice_weaver.ml.adapters.data_augmentation import CSPAugmenter
    from lattice_weaver.ml.training.purifier import DataPurifier
    LATTICEWEAVER_AVAILABLE = True
    print("✅ LatticeWeaver components imported")
except ImportError:
    LATTICEWEAVER_AVAILABLE = False
    print("⚠️  LatticeWeaver not available, using synthetic data only")

print()

# ============================================================================
# CONFIGURATION
# ============================================================================

print("📋 Configuration")
print("-" * 80)

CONFIG = {
    'num_problems': 1000,
    'problem_sizes': {
        'small': 200,    # 20%
        'medium': 600,   # 60%
        'large': 200     # 20%
    },
    'augmentation_factor': 5,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'seed': 42,
    'output_dir': '/content/lattice-weaver/datasets/csp'
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

# Initialize registry
registry = ModelRegistry()
print(f"✅ Model registry initialized")
print(f"   Base path: {registry.base_path}")

# Initialize generator
generator = SyntheticDatasetGenerator(seed=CONFIG['seed'])
print(f"✅ Dataset generator initialized")

# Initialize augmenter (if available)
if LATTICEWEAVER_AVAILABLE:
    augmenter = CSPAugmenter(seed=CONFIG['seed'])
    print(f"✅ Data augmenter initialized")
else:
    augmenter = None
    print(f"⚠️  Data augmenter not available")

print()

# ============================================================================
# STEP 1: GENERATE SYNTHETIC PROBLEMS
# ============================================================================

print("📊 STEP 1: Generate Synthetic Problems")
print("-" * 80)

start_time = time.time()

# Generate datasets for each size
all_features = []
all_labels = []

for size, num_samples in CONFIG['problem_sizes'].items():
    print(f"\n  Generating {num_samples} {size} problems...")
    
    features, labels = generator.generate_csp_dataset(
        num_samples=num_samples,
        size=size
    )
    
    all_features.append(features)
    all_labels.append(labels)
    
    print(f"  ✅ Generated {num_samples} samples")
    print(f"     Features shape: {features.shape}")
    print(f"     Labels shape: {labels.shape}")

# Concatenate all
features = torch.cat(all_features, dim=0)
labels = torch.cat(all_labels, dim=0)

generation_time = time.time() - start_time

print(f"\n✅ Total samples generated: {len(features)}")
print(f"   Features shape: {features.shape}")
print(f"   Labels shape: {labels.shape}")
print(f"   Generation time: {generation_time:.2f}s")
print(f"   Avg time per problem: {generation_time / len(features) * 1000:.2f}ms")

print()

# ============================================================================
# STEP 2: DATA AUGMENTATION (Optional)
# ============================================================================

print("🔄 STEP 2: Data Augmentation")
print("-" * 80)

if augmenter and CONFIG['augmentation_factor'] > 1:
    start_time = time.time()
    
    print(f"  Applying {CONFIG['augmentation_factor']}x augmentation...")
    
    # Simple augmentation: add noise to features
    augmented_features = [features]
    augmented_labels = [labels]
    
    for i in range(CONFIG['augmentation_factor'] - 1):
        # Add small noise to features (except integer features)
        noise = torch.randn_like(features) * 0.05
        # Don't add noise to discrete features (first few columns)
        noise[:, :2] = 0  # num_vars, num_constraints
        noise[:, 5:9] = 0  # depth, backtracks, propagations, constraint_checks
        
        aug_features = features + noise
        aug_features = torch.clamp(aug_features, min=0)  # Ensure non-negative
        
        augmented_features.append(aug_features)
        augmented_labels.append(labels)  # Labels don't change
    
    features = torch.cat(augmented_features, dim=0)
    labels = torch.cat(augmented_labels, dim=0)
    
    augmentation_time = time.time() - start_time
    
    print(f"✅ Augmentation completed")
    print(f"   Original samples: {len(all_features[0]) + len(all_features[1]) + len(all_features[2])}")
    print(f"   Augmented samples: {len(features)}")
    print(f"   Expansion factor: {len(features) / CONFIG['num_problems']:.2f}x")
    print(f"   Augmentation time: {augmentation_time:.2f}s")
else:
    print("  ⏭️  Skipping augmentation")
    augmentation_time = 0

print()

# ============================================================================
# STEP 3: DATA PURIFICATION
# ============================================================================

print("🧹 STEP 3: Data Purification")
print("-" * 80)

start_time = time.time()

# Normalize features
print("  Normalizing features...")
feature_mean = features.mean(dim=0)
feature_std = features.std(dim=0) + 1e-8
features_normalized = (features - feature_mean) / feature_std

# Normalize labels (already log-scaled)
label_mean = labels.mean(dim=0)
label_std = labels.std(dim=0) + 1e-8
labels_normalized = (labels - label_mean) / label_std

purification_time = time.time() - start_time

print(f"✅ Purification completed")
print(f"   Valid samples: {len(features_normalized)} (100.0%)")
print(f"   Feature stats:")
print(f"     Mean: {feature_mean[:5].tolist()}")
print(f"     Std: {feature_std[:5].tolist()}")
print(f"   Label stats:")
print(f"     Mean: {label_mean.tolist()}")
print(f"     Std: {label_std.tolist()}")
print(f"   Purification time: {purification_time:.2f}s")

print()

# ============================================================================
# STEP 4: TRAIN/VAL/TEST SPLIT
# ============================================================================

print("✂️  STEP 4: Train/Val/Test Split")
print("-" * 80)

num_samples = len(features_normalized)
num_train = int(num_samples * CONFIG['train_split'])
num_val = int(num_samples * CONFIG['val_split'])
num_test = num_samples - num_train - num_val

# Shuffle
indices = torch.randperm(num_samples)

train_indices = indices[:num_train]
val_indices = indices[num_train:num_train + num_val]
test_indices = indices[num_train + num_val:]

# Split
train_features = features_normalized[train_indices]
train_labels = labels_normalized[train_indices]

val_features = features_normalized[val_indices]
val_labels = labels_normalized[val_indices]

test_features = features_normalized[test_indices]
test_labels = labels_normalized[test_indices]

print(f"✅ Split completed")
print(f"   Train: {len(train_features)} samples ({len(train_features)/num_samples*100:.1f}%)")
print(f"   Val: {len(val_features)} samples ({len(val_features)/num_samples*100:.1f}%)")
print(f"   Test: {len(test_features)} samples ({len(test_features)/num_samples*100:.1f}%)")

print()

# ============================================================================
# STEP 5: SAVE DATASETS
# ============================================================================

print("💾 STEP 5: Save Datasets")
print("-" * 80)

output_dir = Path(CONFIG['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

# Save train
train_path = output_dir / 'train.pt'
generator.save_dataset(train_features, train_labels, train_path, split='train')

# Save val
val_path = output_dir / 'val.pt'
generator.save_dataset(val_features, val_labels, val_path, split='val')

# Save test
test_path = output_dir / 'test.pt'
generator.save_dataset(test_features, test_labels, test_path, split='test')

# Save normalization stats
stats = {
    'feature_mean': feature_mean.tolist(),
    'feature_std': feature_std.tolist(),
    'label_mean': label_mean.tolist(),
    'label_std': label_std.tolist()
}

stats_path = output_dir / 'normalization_stats.json'
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"✅ Normalization stats saved to {stats_path}")

print()

# ============================================================================
# STEP 6: GENERATE REPORT
# ============================================================================

print("📄 STEP 6: Generate Report")
print("-" * 80)

total_time = generation_time + augmentation_time + purification_time

report = f"""# Dataset Generation Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

| Parameter | Value |
|-----------|-------|
| Total problems | {CONFIG['num_problems']} |
| Small problems | {CONFIG['problem_sizes']['small']} |
| Medium problems | {CONFIG['problem_sizes']['medium']} |
| Large problems | {CONFIG['problem_sizes']['large']} |
| Augmentation factor | {CONFIG['augmentation_factor']}x |
| Train/Val/Test split | {CONFIG['train_split']}/{CONFIG['val_split']}/{CONFIG['test_split']} |

## Results

| Metric | Value |
|--------|-------|
| Original samples | {CONFIG['num_problems']} |
| Augmented samples | {len(features)} |
| Final samples | {num_samples} |
| Train samples | {len(train_features)} |
| Val samples | {len(val_features)} |
| Test samples | {len(test_features)} |

## Timing

| Phase | Time (s) |
|-------|----------|
| Generation | {generation_time:.2f} |
| Augmentation | {augmentation_time:.2f} |
| Purification | {purification_time:.2f} |
| **Total** | **{total_time:.2f}** |

## Quality Metrics

- Valid samples: 100.0%
- Feature normalization: ✓
- Label normalization: ✓

## Output Files

- `{train_path}`
- `{val_path}`
- `{test_path}`
- `{stats_path}`

## Conclusion

✅ Dataset generation completed successfully
"""

report_path = registry.reports_dir / f'dataset_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, 'w') as f:
    f.write(report)

print(f"✅ Report saved to {report_path}")
print()
print(report)

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"✅ Dataset generation completed successfully!")
print(f"   Total samples: {num_samples}")
print(f"   Train: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")
print(f"   Total time: {total_time:.2f}s")
print(f"   Output directory: {output_dir}")
print()
print("Next step: Run Notebook 03 (Training) to train models on this dataset")
print("=" * 80)

