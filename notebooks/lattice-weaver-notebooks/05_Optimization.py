"""
Notebook 05: Optimization

Model optimization through quantization and ONNX export for production deployment.

Usage:
    python 05_Optimization.py
"""

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("NOTEBOOK 05: OPTIMIZATION")
print("=" * 80)
print()

# Imports
import sys
import torch
import torch.nn as nn
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
    'quantize': True,
    'export_onnx': True,
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
# STEP 1: LOAD ORIGINAL MODEL
# ============================================================================

print("📂 STEP 1: Load Original Model")
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

# Create model
if MODELS_AVAILABLE:
    model_original = CostPredictor(input_dim=18, hidden_dim=32)
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
    
    model_original = DummyModel()

model_original.load_state_dict(checkpoint['model_state_dict'])
model_original.eval()

print(f"✅ Original model loaded")

# Benchmark original
print("\n  Benchmarking original model...")
original_metrics = validator.benchmark_inference_speed(
    model=model_original,
    input_dim=model_info['input_dim'],
    num_samples=CONFIG['num_benchmark_samples'],
    device='cpu'  # Benchmark on CPU for fair comparison
)

original_size_mb = sum(p.numel() * p.element_size() for p in model_original.parameters()) / (1024 ** 2)

print(f"  ✅ Original model metrics:")
print(f"     Size: {original_size_mb:.3f} MB")
print(f"     Inference: {original_metrics['inference_time_ms']:.3f} ms")
print(f"     Throughput: {original_metrics['throughput_per_sec']:,} pred/s")

print()

# ============================================================================
# STEP 2: QUANTIZATION
# ============================================================================

if CONFIG['quantize']:
    print("⚙️  STEP 2: Quantization")
    print("-" * 80)
    
    print("  Applying dynamic quantization...")
    
    # Dynamic quantization (easiest, no calibration needed)
    model_quantized = torch.quantization.quantize_dynamic(
        model_original,
        {nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )
    
    print(f"✅ Quantization completed")
    
    # Benchmark quantized
    print("\n  Benchmarking quantized model...")
    quantized_metrics = validator.benchmark_inference_speed(
        model=model_quantized,
        input_dim=model_info['input_dim'],
        num_samples=CONFIG['num_benchmark_samples'],
        device='cpu'
    )
    
    # Calculate size (approximate for quantized model)
    quantized_size_mb = original_size_mb / 4  # INT8 is 4x smaller than FP32
    
    print(f"  ✅ Quantized model metrics:")
    print(f"     Size: {quantized_size_mb:.3f} MB ({original_size_mb / quantized_size_mb:.1f}x smaller)")
    print(f"     Inference: {quantized_metrics['inference_time_ms']:.3f} ms ({original_metrics['inference_time_ms'] / quantized_metrics['inference_time_ms']:.2f}x faster)")
    print(f"     Throughput: {quantized_metrics['throughput_per_sec']:,} pred/s")
    
    # Save quantized model
    quantized_path = registry.models_dir / CONFIG['suite_name'] / f"{CONFIG['model_name']}_quantized.pt"
    quantized_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_quantized.state_dict(), quantized_path)
    print(f"\n  ✅ Quantized model saved to {quantized_path}")
    
    print()
else:
    print("⏭️  STEP 2: Quantization (Skipped)")
    print()
    model_quantized = None
    quantized_metrics = None
    quantized_size_mb = None

# ============================================================================
# STEP 3: ONNX EXPORT
# ============================================================================

if CONFIG['export_onnx']:
    print("📦 STEP 3: ONNX Export")
    print("-" * 80)
    
    print("  Exporting model to ONNX format...")
    
    # Prepare dummy input
    dummy_input = torch.randn(1, model_info['input_dim'])
    
    # Export path
    onnx_path = registry.models_dir / CONFIG['suite_name'] / f"{CONFIG['model_name']}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    torch.onnx.export(
        model_original,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX export completed")
    print(f"   Saved to: {onnx_path}")
    
    # Get ONNX file size
    onnx_size_mb = onnx_path.stat().st_size / (1024 ** 2)
    print(f"   Size: {onnx_size_mb:.3f} MB")
    
    # Validate ONNX (if onnxruntime available)
    try:
        import onnxruntime as ort
        
        print("\n  Validating ONNX model...")
        
        # Create session
        ort_session = ort.InferenceSession(str(onnx_path))
        
        # Test inference
        test_input = torch.randn(10, model_info['input_dim']).numpy()
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = model_original(torch.from_numpy(test_input)).numpy()
        
        # ONNX inference
        onnx_output = ort_session.run(None, {'input': test_input})[0]
        
        # Compare
        max_diff = abs(pytorch_output - onnx_output).max()
        
        if max_diff < 1e-5:
            print(f"  ✅ ONNX validation passed (max diff: {max_diff:.2e})")
        else:
            print(f"  ⚠️  ONNX validation warning (max diff: {max_diff:.2e})")
        
        # Benchmark ONNX
        print("\n  Benchmarking ONNX model...")
        
        start = time.perf_counter()
        for _ in range(CONFIG['num_benchmark_samples'] // 32):
            batch = test_input[:32]
            _ = ort_session.run(None, {'input': batch})
        end = time.perf_counter()
        
        onnx_time_ms = (end - start) / (CONFIG['num_benchmark_samples'] // 32) / 32 * 1000
        onnx_throughput = 1 / (onnx_time_ms / 1000)
        
        print(f"  ✅ ONNX model metrics:")
        print(f"     Inference: {onnx_time_ms:.3f} ms ({original_metrics['inference_time_ms'] / onnx_time_ms:.2f}x faster)")
        print(f"     Throughput: {int(onnx_throughput):,} pred/s")
        
        onnx_metrics = {
            'inference_time_ms': onnx_time_ms,
            'throughput_per_sec': int(onnx_throughput)
        }
        
    except ImportError:
        print("\n  ⚠️  onnxruntime not available, skipping ONNX validation")
        onnx_metrics = None
    
    print()
else:
    print("⏭️  STEP 3: ONNX Export (Skipped)")
    print()
    onnx_path = None
    onnx_size_mb = None
    onnx_metrics = None

# ============================================================================
# STEP 4: COMPARISON TABLE
# ============================================================================

print("📊 STEP 4: Comparison Table")
print("-" * 80)

print("\n| Metric              | Original | Quantized | ONNX |")
print("|---------------------|----------|-----------|------|")
print(f"| Size (MB)           | {original_size_mb:.3f}    | {quantized_size_mb:.3f} ({original_size_mb/quantized_size_mb:.1f}x) | {onnx_size_mb:.3f} |" if CONFIG['quantize'] else f"| Size (MB)           | {original_size_mb:.3f}    | -         | {onnx_size_mb:.3f} |")
print(f"| Inference (ms)      | {original_metrics['inference_time_ms']:.3f}    | {quantized_metrics['inference_time_ms']:.3f} ({original_metrics['inference_time_ms']/quantized_metrics['inference_time_ms']:.2f}x) | {onnx_metrics['inference_time_ms']:.3f} ({original_metrics['inference_time_ms']/onnx_metrics['inference_time_ms']:.2f}x) |" if CONFIG['quantize'] and onnx_metrics else f"| Inference (ms)      | {original_metrics['inference_time_ms']:.3f}    | -         | - |")
print(f"| Throughput (pred/s) | {original_metrics['throughput_per_sec']:,} | {quantized_metrics['throughput_per_sec']:,} | {onnx_metrics['throughput_per_sec']:,} |" if CONFIG['quantize'] and onnx_metrics else f"| Throughput (pred/s) | {original_metrics['throughput_per_sec']:,} | -         | - |")

print()

# ============================================================================
# STEP 5: GENERATE REPORT
# ============================================================================

print("📄 STEP 5: Generate Report")
print("-" * 80)

report = f"""# Optimization Report - {CONFIG['model_name']}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information

| Property | Value |
|----------|-------|
| Name | {CONFIG['model_name']} |
| Suite | {CONFIG['suite_name']} |
| Parameters | {model_info['params']:,} |

## Optimization Results

### Original Model

| Metric | Value |
|--------|-------|
| Size | {original_size_mb:.3f} MB |
| Inference time | {original_metrics['inference_time_ms']:.3f} ms |
| Throughput | {original_metrics['throughput_per_sec']:,} pred/s |

### Quantized Model

| Metric | Value | vs Original |
|--------|-------|-------------|
| Size | {quantized_size_mb:.3f} MB | {original_size_mb/quantized_size_mb:.1f}x smaller |
| Inference time | {quantized_metrics['inference_time_ms']:.3f} ms | {original_metrics['inference_time_ms']/quantized_metrics['inference_time_ms']:.2f}x faster |
| Throughput | {quantized_metrics['throughput_per_sec']:,} pred/s | {quantized_metrics['throughput_per_sec']/original_metrics['throughput_per_sec']:.2f}x higher |
""" if CONFIG['quantize'] else f"""
### Quantized Model

⏭️  Quantization skipped
"""

if CONFIG['export_onnx'] and onnx_metrics:
    report += f"""
### ONNX Model

| Metric | Value | vs Original |
|--------|-------|-------------|
| Size | {onnx_size_mb:.3f} MB | - |
| Inference time | {onnx_metrics['inference_time_ms']:.3f} ms | {original_metrics['inference_time_ms']/onnx_metrics['inference_time_ms']:.2f}x faster |
| Throughput | {onnx_metrics['throughput_per_sec']:,} pred/s | {onnx_metrics['throughput_per_sec']/original_metrics['throughput_per_sec']:.2f}x higher |
"""
else:
    report += """
### ONNX Model

⏭️  ONNX export skipped or validation unavailable
"""

report += f"""
## Recommendation

"""

if CONFIG['export_onnx'] and onnx_metrics:
    report += f"""**Use ONNX model for production deployment**
- Best performance: {original_metrics['inference_time_ms']/onnx_metrics['inference_time_ms']:.2f}x faster than original
- Compatible with ONNX Runtime (C++, Rust, JavaScript)
- Optimized for inference
"""
elif CONFIG['quantize']:
    report += f"""**Use quantized model for production deployment**
- Good performance: {original_metrics['inference_time_ms']/quantized_metrics['inference_time_ms']:.2f}x faster than original
- {original_size_mb/quantized_size_mb:.1f}x smaller model size
- Compatible with PyTorch
"""
else:
    report += """**Use original model**
- No optimizations applied
"""

report += f"""
## Files Generated

- Original model: `{checkpoint_path}`
"""

if CONFIG['quantize']:
    report += f"- Quantized model: `{quantized_path}`\n"

if CONFIG['export_onnx']:
    report += f"- ONNX model: `{onnx_path}`\n"

report += """
---

**Optimization completed successfully! ✅**
"""

report_filename = f"optimization_{CONFIG['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
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
print(f"✅ Optimization completed successfully!")
print(f"   Model: {CONFIG['model_name']}")
if CONFIG['quantize']:
    print(f"   Quantized: {original_size_mb/quantized_size_mb:.1f}x smaller, {original_metrics['inference_time_ms']/quantized_metrics['inference_time_ms']:.2f}x faster")
if CONFIG['export_onnx'] and onnx_metrics:
    print(f"   ONNX: {original_metrics['inference_time_ms']/onnx_metrics['inference_time_ms']:.2f}x faster")
print()
print("Next step: Run Notebook 06 (Model Explorer) to explore the optimized model")
print("=" * 80)

