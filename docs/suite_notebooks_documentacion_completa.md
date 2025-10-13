# Suite de Notebooks de Google Colab - LatticeWeaver ML

**Versión:** 1.0  
**Fecha:** 13 de Octubre, 2025  
**Propósito:** Documentación completa de la suite de notebooks automatizados para entrenamiento, validación y optimización de mini-IAs

---

## Índice

1. [Visión General](#vision-general)
2. [Arquitectura de la Suite](#arquitectura)
3. [Notebook 01: Control Center](#notebook-01)
4. [Notebook 02: Dataset Generation](#notebook-02)
5. [Notebook 03: Training](#notebook-03)
6. [Notebook 04: Validation & Benchmarks](#notebook-04)
7. [Notebook 05: Optimization](#notebook-05)
8. [Notebook 06: Model Explorer](#notebook-06)
9. [Notebook 07: Error Analysis](#notebook-07)
10. [Flujo de Trabajo Completo](#flujo-trabajo)
11. [Captura de Métricas](#metricas)
12. [Manejo de Errores](#errores)
13. [Reportes Automáticos](#reportes)

---

## 1. Visión General {#vision-general}

### Objetivos de la Suite

La suite de notebooks proporciona una **interfaz completa y automatizada** para:

1. **Generar datasets** de entrenamiento desde trazas del solver
2. **Entrenar mini-IAs** con monitoreo en tiempo real
3. **Validar modelos** con benchmarks exhaustivos
4. **Optimizar modelos** (cuantización, ONNX, pruning)
5. **Explorar modelos** preentrenados interactivamente
6. **Analizar errores** y generar reportes detallados

### Características Clave

✅ **Automatización completa** - Un click para ejecutar pipelines completos  
✅ **Documentación exhaustiva** - Cada celda explicada en profundidad  
✅ **Captura de métricas** - Tiempo, memoria, precisión, speedup, overhead  
✅ **Manejo robusto de errores** - Try/catch con reportes detallados y sugerencias  
✅ **Interfaz interactiva** - Widgets, menús, dashboards en tiempo real  
✅ **Transparencia total** - Logs detallados, gráficas, reportes exportables  
✅ **Reproducibilidad** - Seeds fijas, versionado de modelos, tracking completo  

### Filosofía de Diseño

1. **Zero-Configuration:** Funciona out-of-the-box en Google Colab
2. **Progressive Disclosure:** Información básica visible, detalles expandibles
3. **Fail-Safe:** Checkpointing automático, recuperación de errores
4. **Observable:** Todas las métricas visibles en tiempo real
5. **Exportable:** Todos los resultados exportables (CSV, JSON, PNG, PDF)

---

## 2. Arquitectura de la Suite {#arquitectura}

### Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────┐
│                   01. Control Center                        │
│         (Dashboard, Menú, Navegación, Reportes)             │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│02. Dataset   │  │03. Training  │  │04. Validation│
│  Generation  │──│              │──│ & Benchmarks │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │                │
        │                ▼                │
        │         ┌──────────────┐        │
        │         │05.Optimization│       │
        │         └──────────────┘        │
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│06. Model     │  │07. Error     │  │  Reports &   │
│   Explorer   │  │   Analysis   │  │  Metrics     │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Componentes Compartidos

Todos los notebooks comparten:

1. **ModelRegistry** - Registro centralizado de modelos
2. **MetricsCollector** - Captura de métricas unificada
3. **ErrorHandler** - Manejo de errores con reportes
4. **ReportGenerator** - Generación de reportes automáticos
5. **VisualizationUtils** - Utilidades de visualización
6. **ConfigManager** - Gestión de configuración

---

## 3. Notebook 01: Control Center {#notebook-01}

### Propósito

Centro de control para gestión completa de la suite de mini-IAs.

### Funcionalidades

#### 3.1. Dashboard Interactivo

**Componente:** `create_dashboard()`

**Métricas mostradas:**
- Total de mini-IAs (120)
- Implementadas (6/120 = 5%)
- Entrenadas (checkpoints .pt)
- Optimizadas (modelos .onnx)
- Barra de progreso visual

**Actualización:** Tiempo real (cada 5 segundos)

**Código clave:**
```python
class ModelRegistry:
    def get_status_summary(self):
        return {
            "total": 120,
            "implemented": self._count_implemented(),
            "trained": self._count_trained_models(),
            "optimized": self._count_optimized_models(),
            "progress_pct": (implemented / 120) * 100
        }
```

#### 3.2. Menú Principal

**Componente:** `create_main_menu()`

**Botones:**
1. 📊 Generar Datasets → Notebook 02
2. 🚀 Entrenar Modelos → Notebook 03
3. ✅ Validar y Benchmark → Notebook 04
4. ⚡ Optimizar Modelos → Notebook 05
5. 🔬 Explorar Modelos → Notebook 06
6. 🔍 Análisis de Errores → Notebook 07

**Interactividad:** Widgets de ipywidgets con callbacks

#### 3.3. Generador de Reportes

**Componente:** `generate_progress_report()`

**Formato:** Markdown exportable

**Contenido:**
- Resumen general (tabla de métricas)
- Estado por suite (implementación, parámetros, speedup)
- Próximos pasos
- Timestamp de generación

**Exportación:** Guardado en `/content/lattice-weaver/reports/progress_report_YYYYMMDD_HHMMSS.md`

---

## 4. Notebook 02: Dataset Generation {#notebook-02}

### Propósito

Generación automatizada de datasets de entrenamiento desde trazas del solver con data augmentation.

### Funcionalidades

#### 4.1. Generación de Problemas Sintéticos

**Componente:** `SyntheticProblemGenerator`

**Tipos de problemas:**
1. **CSP:** N-Queens, Graph Coloring, Sudoku, Random CSP
2. **TDA:** Point clouds (círculos, esferas, toros, manifolds)
3. **Cubical:** Proof contexts sintéticos
4. **FCA:** Formal contexts aleatorios
5. **Homotopy:** CW complexes sintéticos

**Parámetros configurables:**
- Tamaño (small, medium, large, xlarge)
- Dificultad (easy, medium, hard)
- Número de instancias
- Seed para reproducibilidad

**Código clave:**
```python
class SyntheticProblemGenerator:
    def generate_csp_problems(self, num_problems=1000, size='medium', difficulty='medium'):
        """
        Genera problemas CSP sintéticos.
        
        Args:
            num_problems: Número de problemas a generar
            size: 'small' (10 vars), 'medium' (50 vars), 'large' (100 vars)
            difficulty: 'easy' (sparse), 'medium', 'hard' (dense)
        
        Returns:
            List[CSPProblem]
        """
        problems = []
        for i in range(num_problems):
            if size == 'small':
                n_vars = np.random.randint(5, 15)
            elif size == 'medium':
                n_vars = np.random.randint(20, 60)
            else:  # large
                n_vars = np.random.randint(70, 120)
            
            # Generar problema
            problem = self._generate_random_csp(n_vars, difficulty)
            problems.append(problem)
        
        return problems
```

#### 4.2. Logging de Trazas

**Componente:** `SolverLogger` (integrado en LatticeWeaver)

**Datos capturados por paso:**
```python
{
    "step_id": int,
    "timestamp": float,
    "state": {
        "num_variables": int,
        "num_constraints": int,
        "domains": Dict[int, Set[int]],
        "constraint_graph": nx.Graph,
        "depth": int,
        "num_backtracks": int,
        "num_propagations": int,
        "constraint_checks": int,
        "time_elapsed_ms": float
    },
    "decision": {
        "variable": int,
        "value": int,
        "heuristic_used": str,
        "alternatives_considered": List[Tuple[int, float]]
    },
    "result": {
        "success": bool,
        "propagations_triggered": int,
        "domain_reductions": int,
        "time_ms": float,
        "backtracked": bool
    }
}
```

**Almacenamiento:** JSON Lines (.jsonl) para streaming eficiente

#### 4.3. Data Augmentation

**Componente:** `DataAugmenter` (ya implementado)

**Augmenters disponibles:**
1. **CSPAugmenter:** Permutación de variables/valores, ruido en métricas
2. **TDAAugmenter:** Rotaciones, traslaciones, escalado, ruido
3. **CubicalAugmenter:** Reordenamiento de hipótesis
4. **FCAAugmenter:** Permutación de objetos/atributos
5. **HomotopyAugmenter:** Subdivisión, ruido

**Factor de expansión:** 4-10x (configurable)

**Código clave:**
```python
# Aplicar augmentation
augmenter = CSPAugmenter(seed=42)
augmented_data = []

for original_trace in traces:
    # Original + 5 augmentaciones
    augmented = augmenter.augment(original_trace, num_augmentations=5)
    augmented_data.extend(augmented)

print(f"Dataset expandido: {len(traces)} → {len(augmented_data)} ({len(augmented_data)/len(traces):.1f}x)")
```

#### 4.4. Purificación de Datos

**Componente:** `DataPurifier` (ya implementado)

**Pipeline de purificación:**
1. **Filtrar fallos:** Eliminar trazas de ejecuciones fallidas
2. **Etiquetar decisiones:** Marcar decisiones óptimas vs subóptimas
3. **Balancear clases:** Asegurar balance 50/50 buenas/malas decisiones
4. **Extraer features:** Convertir a tensores (18 dims para CSP)
5. **Normalizar:** Z-score normalization

**Métricas de calidad:**
- Porcentaje de trazas válidas
- Balance de clases
- Distribución de features

**Código clave:**
```python
purifier = DataPurifier()

# Purificar dataset
purified_data = purifier.purify(
    raw_traces=traces,
    min_quality_score=0.7,
    balance_classes=True,
    normalize=True
)

# Métricas de calidad
print(f"Trazas válidas: {purified_data['valid_pct']:.1f}%")
print(f"Balance de clases: {purified_data['class_balance']}")
```

#### 4.5. Exportación de Datasets

**Formatos soportados:**
1. **PyTorch (.pt):** Tensores listos para entrenamiento
2. **NumPy (.npz):** Arrays comprimidos
3. **JSON (.json):** Formato legible
4. **HDF5 (.h5):** Para datasets muy grandes

**Estructura de directorios:**
```
/content/lattice-weaver/datasets/
├── csp/
│   ├── train.pt
│   ├── val.pt
│   ├── test.pt
│   └── metadata.json
├── tda/
│   ├── train.pt
│   ├── val.pt
│   └── test.pt
└── ...
```

#### 4.6. Métricas Capturadas

**Durante generación:**
- Tiempo de generación por problema
- Tamaño de cada problema (variables, restricciones)
- Tiempo de resolución
- Número de pasos de búsqueda

**Durante augmentation:**
- Factor de expansión real
- Tiempo de augmentation
- Validación de invarianzas

**Durante purificación:**
- Porcentaje de datos descartados
- Balance de clases
- Estadísticas de features (media, std, min, max)

**Reporte generado:**
```markdown
# Dataset Generation Report

## Summary
- Problems generated: 1,000
- Augmentation factor: 5.2x
- Final dataset size: 5,200
- Train/val/test split: 70/15/15%

## Quality Metrics
- Valid traces: 94.3%
- Class balance: 51.2% / 48.8%
- Feature normalization: ✓

## Generation Time
- Problem generation: 2.3 min
- Solver execution: 15.7 min
- Augmentation: 1.2 min
- Purification: 0.8 min
- Total: 20.0 min
```

---

## 5. Notebook 03: Training {#notebook-03}

### Propósito

Entrenamiento automatizado de mini-IAs con monitoreo en tiempo real y checkpointing.

### Funcionalidades

#### 5.1. Configuración de Entrenamiento

**Componente:** `TrainingConfig`

**Parámetros configurables:**
```python
@dataclass
class TrainingConfig:
    # Modelo
    model_name: str = "CostPredictor"
    suite_name: str = "costs_memoization"
    
    # Datos
    batch_size: int = 32
    num_workers: int = 4
    
    # Optimización
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adam"  # adam, adamw, sgd
    
    # Entrenamiento
    num_epochs: int = 100
    early_stopping_patience: int = 10
    lr_scheduler: str = "reduce_on_plateau"  # reduce_on_plateau, cosine, step
    
    # Regularización
    dropout: float = 0.1
    label_smoothing: float = 0.0
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_best_n: int = 3
    
    # Logging
    log_every_n_steps: int = 10
    wandb_enabled: bool = False
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Automatic Mixed Precision (AMP)
```

**Interfaz interactiva:** Widgets para configurar todos los parámetros

#### 5.2. Entrenamiento Automatizado

**Componente:** `AutomatedTrainer`

**Pipeline de entrenamiento:**
1. Cargar dataset
2. Crear dataloaders
3. Inicializar modelo
4. Configurar optimizer y scheduler
5. Loop de entrenamiento con:
   - Forward pass
   - Loss computation
   - Backward pass
   - Gradient clipping
   - Optimizer step
   - Metrics logging
6. Validación cada epoch
7. Checkpointing
8. Early stopping

**Código clave:**
```python
class AutomatedTrainer:
    def train(self, config: TrainingConfig):
        """
        Entrena modelo con configuración dada.
        
        Returns:
            Dict con métricas finales y path al mejor modelo
        """
        # Setup
        model = self._create_model(config.model_name)
        train_loader, val_loader = self._create_dataloaders(config)
        optimizer = self._create_optimizer(model, config)
        scheduler = self._create_scheduler(optimizer, config)
        criterion = self._create_criterion(config)
        
        # Tracking
        best_val_loss = float('inf')
        patience_counter = 0
        metrics_history = []
        
        # Training loop
        for epoch in range(config.num_epochs):
            # Train
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, criterion, config
            )
            
            # Validate
            val_metrics = self._validate_epoch(
                model, val_loader, criterion, config
            )
            
            # Scheduler step
            if config.lr_scheduler == "reduce_on_plateau":
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
            
            # Logging
            metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            metrics_history.append(metrics)
            self._log_metrics(metrics, config)
            
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
                print(f"Early stopping at epoch {epoch}")
                break
        
        return {
            'best_val_loss': best_val_loss,
            'final_epoch': epoch,
            'metrics_history': metrics_history,
            'best_model_path': self._get_best_checkpoint_path(config)
        }
```

#### 5.3. Monitoreo en Tiempo Real

**Componente:** `RealTimeMonitor`

**Visualizaciones:**
1. **Loss curves:** Train vs val loss en tiempo real
2. **Learning rate:** Evolución del LR
3. **Gradient norms:** Detección de exploding/vanishing gradients
4. **Metrics:** Precisión, MAE, R² según el modelo
5. **Resource usage:** GPU memory, utilization

**Actualización:** Cada 10 steps (configurable)

**Código clave:**
```python
class RealTimeMonitor:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.metrics_buffer = []
    
    def update(self, metrics):
        """Actualiza visualizaciones con nuevas métricas."""
        self.metrics_buffer.append(metrics)
        
        # Plot loss
        self.axes[0, 0].clear()
        epochs = [m['epoch'] for m in self.metrics_buffer]
        train_loss = [m['train_loss'] for m in self.metrics_buffer]
        val_loss = [m['val_loss'] for m in self.metrics_buffer]
        self.axes[0, 0].plot(epochs, train_loss, label='Train')
        self.axes[0, 0].plot(epochs, val_loss, label='Val')
        self.axes[0, 0].set_title('Loss')
        self.axes[0, 0].legend()
        
        # Plot learning rate
        self.axes[0, 1].clear()
        lr = [m['learning_rate'] for m in self.metrics_buffer]
        self.axes[0, 1].plot(epochs, lr)
        self.axes[0, 1].set_title('Learning Rate')
        
        # Plot gradient norms
        self.axes[0, 2].clear()
        grad_norms = [m.get('grad_norm', 0) for m in self.metrics_buffer]
        self.axes[0, 2].plot(epochs, grad_norms)
        self.axes[0, 2].set_title('Gradient Norm')
        
        # Plot metrics (precision, MAE, etc.)
        # ...
        
        # Refresh display
        clear_output(wait=True)
        display(self.fig)
```

#### 5.4. Checkpointing Automático

**Estrategia:**
1. **Best model:** Guardado cuando val_loss mejora
2. **Periodic:** Cada N epochs (configurable)
3. **Last:** Último epoch siempre guardado
4. **Top-K:** Mantener solo los K mejores modelos

**Formato de checkpoint:**
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'scheduler_state_dict': OrderedDict,
    'metrics': Dict,
    'config': TrainingConfig,
    'timestamp': str,
    'git_commit': str,  # Para reproducibilidad
}
```

**Naming convention:**
```
models/costs_memoization/CostPredictor_epoch050_valloss0.123.pt
models/costs_memoization/CostPredictor_best.pt
models/costs_memoization/CostPredictor_last.pt
```

#### 5.5. Manejo de Errores

**Errores capturados:**
1. **OOM (Out of Memory):** Reducir batch size automáticamente
2. **NaN loss:** Reducir learning rate, reiniciar desde último checkpoint
3. **Exploding gradients:** Gradient clipping más agresivo
4. **Dataset corrupto:** Validación y limpieza automática

**Código clave:**
```python
try:
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
except RuntimeError as e:
    if "out of memory" in str(e):
        # OOM handling
        print(f"⚠️  OOM detected, reducing batch size from {config.batch_size} to {config.batch_size // 2}")
        config.batch_size = config.batch_size // 2
        torch.cuda.empty_cache()
        # Reload dataloaders
        train_loader, val_loader = self._create_dataloaders(config)
        continue
    else:
        raise e

# Check for NaN
if torch.isnan(loss):
    print("⚠️  NaN loss detected, loading last checkpoint and reducing LR")
    model, optimizer = self._load_last_checkpoint(config)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    continue
```

#### 5.6. Métricas Capturadas

**Por epoch:**
- Train loss, val loss
- Train accuracy/MAE/R² (según modelo)
- Val accuracy/MAE/R²
- Learning rate
- Gradient norm
- Epoch time
- GPU memory usage

**Por step:**
- Batch loss
- Batch time
- GPU utilization

**Agregadas:**
- Best val loss
- Best epoch
- Total training time
- Average epoch time
- Convergence speed (epochs to 90% best)

**Exportación:** CSV, JSON, TensorBoard logs

---

## 6. Notebook 04: Validation & Benchmarks {#notebook-04}

### Propósito

Validación exhaustiva de modelos entrenados con benchmarks de speedup, overhead y precisión.

### Funcionalidades

#### 6.1. Tests de Correctitud

**Componente:** `CorrectnessValidator`

**Tests implementados:**
1. **Dimensiones correctas:** Output shape matches expected
2. **Rango de valores:** Outputs en rango válido (e.g., probabilidades 0-1)
3. **Invarianzas:** Modelo respeta invarianzas conocidas (e.g., permutación)
4. **Casos extremos:** Comportamiento en inputs edge-case
5. **Determinismo:** Mismo input → mismo output (con seed fija)

**Código clave:**
```python
class CorrectnessValidator:
    def validate_model(self, model, test_cases):
        """
        Valida correctitud del modelo.
        
        Returns:
            Dict con resultados de cada test
        """
        results = {}
        
        # Test 1: Dimensiones
        for input_tensor, expected_shape in test_cases['dimensions']:
            output = model(input_tensor)
            results['dimensions'] = (output.shape == expected_shape)
        
        # Test 2: Rango de valores
        for input_tensor, (min_val, max_val) in test_cases['value_range']:
            output = model(input_tensor)
            in_range = (output >= min_val).all() and (output <= max_val).all()
            results['value_range'] = in_range
        
        # Test 3: Invarianzas
        for original, transformed in test_cases['invariances']:
            out_original = model(original)
            out_transformed = model(transformed)
            invariant = torch.allclose(out_original, out_transformed, atol=1e-5)
            results['invariances'] = invariant
        
        # Test 4: Determinismo
        torch.manual_seed(42)
        out1 = model(test_cases['determinism_input'])
        torch.manual_seed(42)
        out2 = model(test_cases['determinism_input'])
        results['determinism'] = torch.allclose(out1, out2)
        
        return results
```

#### 6.2. Benchmarks de Precisión

**Componente:** `PrecisionBenchmark`

**Métricas por tipo de modelo:**

**Regresión (CostPredictor, etc.):**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

**Clasificación (MemoizationGuide, etc.):**
- Accuracy
- Precision, Recall, F1
- ROC-AUC
- Confusion matrix

**Ranking (VariableSelector, etc.):**
- Top-1 accuracy
- Top-3 accuracy
- Top-5 accuracy
- MRR (Mean Reciprocal Rank)

**Código clave:**
```python
class PrecisionBenchmark:
    def benchmark_regression(self, model, test_loader):
        """Benchmark para modelos de regresión."""
        predictions = []
        targets = []
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                outputs = model(inputs)
                predictions.append(outputs.cpu())
                targets.append(labels.cpu())
        
        predictions = torch.cat(predictions).numpy()
        targets = torch.cat(targets).numpy()
        
        # Métricas
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        r2 = 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
```

#### 6.3. Benchmarks de Speedup

**Componente:** `SpeedupBenchmark`

**Metodología:**
1. Ejecutar solver **sin ML** (baseline)
2. Ejecutar solver **con ML**
3. Comparar:
   - Tiempo total
   - Nodos explorados
   - Backtracks
   - Propagaciones

**Problemas de benchmark:**
- Small (10 vars): 100 instancias
- Medium (50 vars): 50 instancias
- Large (100 vars): 20 instancias

**Código clave:**
```python
class SpeedupBenchmark:
    def benchmark_speedup(self, model, problems):
        """
        Compara solver con y sin ML.
        
        Returns:
            Dict con speedup metrics
        """
        baseline_results = []
        ml_results = []
        
        for problem in problems:
            # Baseline (sin ML)
            start = time.time()
            solution_baseline, stats_baseline = self.solver.solve(problem, use_ml=False)
            time_baseline = time.time() - start
            
            # Con ML
            start = time.time()
            solution_ml, stats_ml = self.solver.solve(problem, use_ml=True, ml_model=model)
            time_ml = time.time() - start
            
            baseline_results.append({
                'time': time_baseline,
                'nodes': stats_baseline['nodes_explored'],
                'backtracks': stats_baseline['backtracks']
            })
            
            ml_results.append({
                'time': time_ml,
                'nodes': stats_ml['nodes_explored'],
                'backtracks': stats_ml['backtracks']
            })
        
        # Calcular speedup
        avg_time_baseline = np.mean([r['time'] for r in baseline_results])
        avg_time_ml = np.mean([r['time'] for r in ml_results])
        speedup = avg_time_baseline / avg_time_ml
        
        avg_nodes_baseline = np.mean([r['nodes'] for r in baseline_results])
        avg_nodes_ml = np.mean([r['nodes'] for r in ml_results])
        node_reduction = (avg_nodes_baseline - avg_nodes_ml) / avg_nodes_baseline * 100
        
        return {
            'speedup': speedup,
            'time_baseline_ms': avg_time_baseline * 1000,
            'time_ml_ms': avg_time_ml * 1000,
            'node_reduction_pct': node_reduction,
            'nodes_baseline': avg_nodes_baseline,
            'nodes_ml': avg_nodes_ml
        }
```

#### 6.4. Benchmarks de Overhead

**Componente:** `OverheadBenchmark`

**Métricas:**
1. **Tiempo de inferencia:** Tiempo puro de forward pass
2. **Overhead de integración:** Tiempo de feature extraction + decoding
3. **Overhead de memoria:** Memoria adicional del modelo
4. **Throughput:** Predicciones por segundo

**Código clave:**
```python
class OverheadBenchmark:
    def benchmark_overhead(self, model, num_samples=10000):
        """
        Mide overhead del modelo.
        
        Returns:
            Dict con overhead metrics
        """
        # Generar inputs sintéticos
        inputs = torch.randn(num_samples, model.input_dim)
        
        # Warm-up
        for _ in range(100):
            _ = model(inputs[:32])
        
        # Benchmark de inferencia
        times = []
        for i in range(0, num_samples, 32):
            batch = inputs[i:i+32]
            
            start = time.perf_counter()
            _ = model(batch)
            end = time.perf_counter()
            
            times.append(end - start)
        
        # Métricas
        avg_batch_time = np.mean(times)
        avg_per_sample = avg_batch_time / 32
        throughput = 1 / avg_per_sample
        
        # Memoria
        model_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        
        return {
            'inference_time_ms': avg_per_sample * 1000,
            'throughput_per_sec': throughput,
            'model_memory_mb': model_memory / (1024 ** 2),
            'overhead_pct': self._calculate_overhead_pct(avg_per_sample)
        }
```

#### 6.5. Visualizaciones

**Gráficas generadas:**
1. **Speedup por tamaño de problema:** Bar chart
2. **Precisión vs baseline:** Scatter plot
3. **Overhead breakdown:** Pie chart
4. **Distribución de errores:** Histogram
5. **Confusion matrix:** Heatmap (para clasificación)
6. **ROC curve:** Curve (para clasificación binaria)

#### 6.6. Reportes de Validación

**Formato:** Markdown + PDF

**Contenido:**
```markdown
# Validation Report - CostPredictor

## Model Info
- Name: CostPredictor
- Parameters: 3,395
- Memory: 13.26 KB
- Trained epochs: 87
- Best val loss: 0.0234

## Correctness Tests
- ✅ Dimensions: PASS
- ✅ Value range: PASS
- ✅ Invariances: PASS
- ✅ Determinism: PASS

## Precision Metrics
- MAE: 0.123 (target: < 0.2)
- RMSE: 0.187
- R²: 0.912
- MAPE: 8.7%

## Speedup Benchmarks
- Average speedup: 1.67x
- Time reduction: 40.1%
- Node reduction: 32.5%

## Overhead
- Inference time: 0.02 ms
- Throughput: 50,000 pred/sec
- Memory overhead: 13.26 KB (0.03%)
- Total overhead: 3.2%

## Conclusion
✅ Model meets all acceptance criteria
✅ Ready for deployment
```

---

## 7. Notebook 05: Optimization {#notebook-05}

### Propósito

Optimización de modelos entrenados mediante cuantización, ONNX export y pruning.

### Funcionalidades

#### 7.1. Cuantización

**Componente:** `ModelQuantizer`

**Tipos de cuantización:**
1. **Dynamic quantization:** Pesos INT8, activaciones FP32
2. **Static quantization:** Pesos y activaciones INT8 (requiere calibración)
3. **Quantization-aware training:** Entrenamiento con cuantización simulada

**Código clave:**
```python
class ModelQuantizer:
    def quantize_dynamic(self, model):
        """
        Cuantización dinámica (más simple, menos speedup).
        
        Returns:
            Modelo cuantizado
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Capas a cuantizar
            dtype=torch.qint8
        )
        return quantized_model
    
    def quantize_static(self, model, calibration_loader):
        """
        Cuantización estática (más compleja, mayor speedup).
        
        Args:
            calibration_loader: DataLoader para calibración
        
        Returns:
            Modelo cuantizado
        """
        # Preparar modelo
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model)
        
        # Calibración
        model_prepared.eval()
        with torch.no_grad():
            for batch in calibration_loader:
                model_prepared(batch[0])
        
        # Convertir
        quantized_model = torch.quantization.convert(model_prepared)
        return quantized_model
```

**Validación post-cuantización:**
- Comparar precisión (debe perder < 1%)
- Medir speedup (esperado: 2-4x en CPU)
- Medir reducción de memoria (esperado: 4x)

#### 7.2. ONNX Export

**Componente:** `ONNXExporter`

**Ventajas de ONNX:**
- Interoperabilidad (C++, Rust, JavaScript)
- Optimizaciones adicionales (graph optimization)
- Deployment en producción (ONNX Runtime 6x más rápido)

**Código clave:**
```python
class ONNXExporter:
    def export_to_onnx(self, model, input_shape, output_path):
        """
        Exporta modelo a ONNX.
        
        Args:
            model: Modelo PyTorch
            input_shape: Tuple con shape de input
            output_path: Path para guardar .onnx
        """
        # Dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Export
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
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
        
        print(f"✅ Modelo exportado a {output_path}")
        
        # Validar
        self._validate_onnx(output_path, model, dummy_input)
    
    def _validate_onnx(self, onnx_path, original_model, test_input):
        """Valida que ONNX produce mismos resultados."""
        import onnxruntime as ort
        
        # Inferencia PyTorch
        original_model.eval()
        with torch.no_grad():
            pytorch_output = original_model(test_input).numpy()
        
        # Inferencia ONNX
        ort_session = ort.InferenceSession(onnx_path)
        onnx_output = ort_session.run(None, {'input': test_input.numpy()})[0]
        
        # Comparar
        assert np.allclose(pytorch_output, onnx_output, atol=1e-5), "ONNX output differs!"
        print("✅ ONNX validation passed")
```

#### 7.3. Pruning

**Componente:** `ModelPruner`

**Tipos de pruning:**
1. **Magnitude pruning:** Eliminar pesos con magnitud < threshold
2. **Structured pruning:** Eliminar neuronas/canales completos
3. **Iterative pruning:** Pruning gradual con fine-tuning

**Código clave:**
```python
class ModelPruner:
    def prune_magnitude(self, model, amount=0.3):
        """
        Pruning por magnitud.
        
        Args:
            amount: Fracción de pesos a eliminar (0.3 = 30%)
        
        Returns:
            Modelo pruned
        """
        import torch.nn.utils.prune as prune
        
        # Aplicar pruning a todas las capas lineales
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
        
        # Hacer permanente
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')
        
        return model
```

**Validación post-pruning:**
- Comparar precisión (debe perder < 2%)
- Medir reducción de parámetros
- Medir speedup (puede ser mínimo sin hardware especializado)

#### 7.4. Comparación Pre/Post Optimización

**Métricas comparadas:**
1. **Tamaño del modelo:** MB
2. **Tiempo de inferencia:** ms
3. **Throughput:** predicciones/segundo
4. **Precisión:** MAE, accuracy, etc.
5. **Memoria en runtime:** MB

**Tabla de comparación:**
```
| Métrica              | Original | Cuantizado | ONNX | Pruned |
|----------------------|----------|------------|------|--------|
| Tamaño (KB)          | 13.26    | 3.31       | 3.50 | 9.28   |
| Inferencia (ms)      | 0.020    | 0.015      | 0.003| 0.018  |
| Throughput (pred/s)  | 50,000   | 66,667     |333,333|55,556 |
| MAE                  | 0.123    | 0.125      | 0.123| 0.128  |
| Speedup vs original  | 1.0x     | 1.33x      | 6.67x| 1.11x  |
```

#### 7.5. Exportación de Modelos Optimizados

**Formatos:**
- `.pt` (PyTorch cuantizado)
- `.onnx` (ONNX Runtime)
- `.pt` (PyTorch pruned)

**Estructura de directorios:**
```
models/costs_memoization/
├── CostPredictor_original.pt
├── CostPredictor_quantized.pt
├── CostPredictor.onnx
├── CostPredictor_pruned.pt
└── optimization_report.md
```

---

## 8. Notebook 06: Model Explorer {#notebook-06}

### Propósito

Exploración interactiva de modelos preentrenados con inferencia en vivo y visualización de embeddings.

### Funcionalidades

#### 8.1. Carga de Modelos

**Componente:** `ModelLoader`

**Interfaz:**
- Dropdown para seleccionar modelo
- Botón "Load Model"
- Visualización de metadata (parámetros, precisión, etc.)

**Código clave:**
```python
class ModelLoader:
    def load_model(self, model_name):
        """
        Carga modelo y muestra metadata.
        
        Returns:
            Modelo cargado
        """
        checkpoint_path = self.registry.get_model_info(model_name)['checkpoint_path']
        
        # Cargar checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Crear modelo
        model = self._create_model_from_checkpoint(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Mostrar metadata
        self._display_metadata(checkpoint)
        
        return model
```

#### 8.2. Inferencia Interactiva

**Componente:** `InteractiveInference`

**Interfaz:**
- Sliders/inputs para features
- Botón "Predict"
- Visualización de predicción
- Comparación con ground truth (si disponible)

**Ejemplo para CostPredictor:**
```python
# Widgets
num_variables = widgets.IntSlider(min=5, max=100, value=20, description='Variables')
num_constraints = widgets.IntSlider(min=5, max=200, value=40, description='Constraints')
avg_domain_size = widgets.FloatSlider(min=2, max=10, value=5, description='Avg Domain')
# ... (18 features en total)

def on_predict_click(b):
    # Extraer features de widgets
    features = torch.tensor([
        num_variables.value,
        num_constraints.value,
        avg_domain_size.value,
        # ...
    ]).float().unsqueeze(0)
    
    # Predicción
    with torch.no_grad():
        prediction = model(features)
    
    # Mostrar
    print(f"Predicted time: {np.exp(prediction[0, 0].item()):.2f} ms")
    print(f"Predicted memory: {np.exp(prediction[0, 1].item()):.2f} MB")
    print(f"Predicted nodes: {int(np.exp(prediction[0, 2].item()))}")

predict_button.on_click(on_predict_click)
```

#### 8.3. Visualización de Embeddings

**Componente:** `EmbeddingVisualizer`

**Técnicas:**
1. **t-SNE:** Proyección 2D de embeddings
2. **UMAP:** Alternativa más rápida a t-SNE
3. **PCA:** Componentes principales

**Visualización:**
- Scatter plot interactivo (plotly)
- Coloreado por clase/label
- Hover para ver detalles

**Código clave:**
```python
class EmbeddingVisualizer:
    def visualize_embeddings(self, model, data_loader, method='tsne'):
        """
        Visualiza embeddings del modelo.
        
        Args:
            method: 'tsne', 'umap', o 'pca'
        """
        # Extraer embeddings
        embeddings = []
        labels = []
        
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                # Obtener embeddings (capa intermedia)
                emb = model.get_embeddings(inputs)
                embeddings.append(emb.cpu())
                labels.append(targets.cpu())
        
        embeddings = torch.cat(embeddings).numpy()
        labels = torch.cat(labels).numpy()
        
        # Reducción de dimensionalidad
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'umap':
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:  # pca
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot interactivo
        import plotly.express as px
        fig = px.scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            color=labels,
            title=f'Embeddings Visualization ({method.upper()})'
        )
        fig.show()
```

#### 8.4. Análisis de Atención

**Para modelos con atención (Transformers):**
- Visualización de attention weights
- Heatmap de qué features son más importantes

#### 8.5. Análisis de Sensibilidad

**Componente:** `SensitivityAnalyzer`

**Pregunta:** ¿Cómo cambia la predicción al variar cada feature?

**Visualización:**
- Gráfica de sensibilidad por feature
- Identificación de features más importantes

---

## 9. Notebook 07: Error Analysis {#notebook-07}

### Propósito

Análisis detallado de errores del modelo con sugerencias de corrección.

### Funcionalidades

#### 9.1. Captura de Errores

**Componente:** `ErrorCollector`

**Tipos de errores capturados:**
1. **Errores de predicción:** Casos donde MAE > threshold
2. **Fallos de integración:** Excepciones durante inferencia
3. **Violaciones de invarianzas:** Modelo no respeta invarianzas
4. **Casos adversariales:** Inputs que engañan al modelo

**Código clave:**
```python
class ErrorCollector:
    def collect_prediction_errors(self, model, test_loader, threshold=0.5):
        """
        Recopila casos con error > threshold.
        
        Returns:
            List de (input, prediction, target, error)
        """
        errors = []
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                predictions = model(inputs)
                
                # Calcular error por muestra
                batch_errors = torch.abs(predictions - targets)
                
                # Filtrar errores grandes
                for i in range(len(inputs)):
                    if batch_errors[i].max() > threshold:
                        errors.append({
                            'input': inputs[i].cpu().numpy(),
                            'prediction': predictions[i].cpu().numpy(),
                            'target': targets[i].cpu().numpy(),
                            'error': batch_errors[i].cpu().numpy()
                        })
        
        return errors
```

#### 9.2. Análisis de Patrones de Error

**Componente:** `ErrorPatternAnalyzer`

**Análisis:**
1. **Distribución de errores:** Histogram
2. **Errores por rango de input:** ¿Modelo falla en problemas grandes?
3. **Correlación error-features:** ¿Qué features causan más error?
4. **Clustering de errores:** ¿Hay grupos de errores similares?

**Código clave:**
```python
class ErrorPatternAnalyzer:
    def analyze_error_distribution(self, errors):
        """Analiza distribución de errores."""
        error_magnitudes = [e['error'].max() for e in errors]
        
        # Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(error_magnitudes, bins=50, edgecolor='black')
        plt.xlabel('Error Magnitude')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.show()
        
        # Estadísticas
        print(f"Mean error: {np.mean(error_magnitudes):.4f}")
        print(f"Median error: {np.median(error_magnitudes):.4f}")
        print(f"95th percentile: {np.percentile(error_magnitudes, 95):.4f}")
    
    def analyze_error_by_input_range(self, errors):
        """Analiza errores por rango de input."""
        # Agrupar por tamaño de problema (feature 0 = num_variables)
        small_errors = [e for e in errors if e['input'][0] < 20]
        medium_errors = [e for e in errors if 20 <= e['input'][0] < 50]
        large_errors = [e for e in errors if e['input'][0] >= 50]
        
        # Comparar
        print(f"Small problems: {len(small_errors)} errors, avg {np.mean([e['error'].max() for e in small_errors]):.4f}")
        print(f"Medium problems: {len(medium_errors)} errors, avg {np.mean([e['error'].max() for e in medium_errors]):.4f}")
        print(f"Large problems: {len(large_errors)} errors, avg {np.mean([e['error'].max() for e in large_errors]):.4f}")
```

#### 9.3. Sugerencias de Corrección

**Componente:** `CorrectionSuggester`

**Sugerencias automáticas:**
1. **Más datos:** Si error correlaciona con escasez de datos en región
2. **Augmentation adicional:** Si modelo no generaliza a transformaciones
3. **Arquitectura más compleja:** Si underfitting
4. **Regularización:** Si overfitting
5. **Feature engineering:** Si features no capturan información relevante

**Código clave:**
```python
class CorrectionSuggester:
    def suggest_corrections(self, errors, model, training_data):
        """
        Genera sugerencias de corrección.
        
        Returns:
            List de sugerencias con prioridad
        """
        suggestions = []
        
        # Análisis 1: Escasez de datos
        error_regions = self._identify_error_regions(errors)
        data_density = self._compute_data_density(training_data, error_regions)
        
        if data_density < 0.1:  # Menos de 10% de datos en región de error
            suggestions.append({
                'priority': 'HIGH',
                'type': 'data_collection',
                'message': 'Collect more training data in error-prone regions',
                'details': f'Only {data_density*100:.1f}% of training data covers error regions'
            })
        
        # Análisis 2: Underfitting vs Overfitting
        train_error = self._compute_train_error(model, training_data)
        test_error = np.mean([e['error'].max() for e in errors])
        
        if train_error > 0.2 and test_error > 0.2:
            suggestions.append({
                'priority': 'HIGH',
                'type': 'model_capacity',
                'message': 'Increase model capacity (underfitting)',
                'details': f'Both train ({train_error:.3f}) and test ({test_error:.3f}) errors are high'
            })
        elif train_error < 0.1 and test_error > 0.3:
            suggestions.append({
                'priority': 'MEDIUM',
                'type': 'regularization',
                'message': 'Add regularization (overfitting)',
                'details': f'Train error ({train_error:.3f}) << test error ({test_error:.3f})'
            })
        
        # Análisis 3: Feature importance
        important_features = self._analyze_feature_importance(model, errors)
        if len(important_features) < 5:
            suggestions.append({
                'priority': 'MEDIUM',
                'type': 'feature_engineering',
                'message': 'Add more informative features',
                'details': f'Only {len(important_features)} features are highly predictive'
            })
        
        return sorted(suggestions, key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']])
```

#### 9.4. Reportes de Errores

**Formato:** Markdown + HTML interactivo

**Contenido:**
```markdown
# Error Analysis Report - CostPredictor

## Summary
- Total test samples: 1,000
- Errors (MAE > 0.5): 87 (8.7%)
- Mean error: 0.123
- 95th percentile error: 0.678

## Error Distribution
[Histogram]

## Error Patterns
- Small problems (< 20 vars): 2.3% error rate
- Medium problems (20-50 vars): 7.1% error rate
- Large problems (> 50 vars): 15.4% error rate

**Finding:** Model struggles with large problems

## Top 10 Worst Cases
1. Input: [100, 250, 8.5, ...], Predicted: 1234 ms, Actual: 3456 ms, Error: 180%
2. ...

## Suggestions
### HIGH PRIORITY
1. **Collect more training data for large problems**
   - Current coverage: 8.2% of training data
   - Recommended: Generate 500 more large problem instances

### MEDIUM PRIORITY
2. **Increase model capacity**
   - Current: 3,395 parameters
   - Suggested: Try 10K-20K parameters

## Next Steps
1. Generate additional training data for large problems
2. Retrain with increased capacity
3. Re-evaluate on same test set
```

---

## 10. Flujo de Trabajo Completo {#flujo-trabajo}

### Flujo Recomendado

```
1. Control Center (Notebook 01)
   ↓
2. Dataset Generation (Notebook 02)
   - Generar 1,000 problemas sintéticos
   - Ejecutar solver con logging
   - Aplicar augmentation (5x)
   - Purificar datos
   - Exportar train/val/test
   ↓
3. Training (Notebook 03)
   - Configurar hiperparámetros
   - Entrenar con early stopping
   - Monitorear en tiempo real
   - Checkpointing automático
   ↓
4. Validation & Benchmarks (Notebook 04)
   - Tests de correctitud
   - Benchmarks de precisión
   - Benchmarks de speedup
   - Benchmarks de overhead
   - Generar reporte
   ↓
5. Optimization (Notebook 05)
   - Cuantizar modelo
   - Exportar a ONNX
   - Pruning (opcional)
   - Comparar pre/post
   ↓
6. Model Explorer (Notebook 06)
   - Cargar modelo optimizado
   - Inferencia interactiva
   - Visualizar embeddings
   ↓
7. Error Analysis (Notebook 07)
   - Analizar errores
   - Identificar patrones
   - Generar sugerencias
   ↓
   Si errores > threshold:
     Volver a paso 2 (más datos)
   Else:
     ✅ Modelo listo para deployment
```

### Tiempo Estimado

**Por modelo (e.g., CostPredictor):**
- Dataset generation: 20 min
- Training: 30-60 min (dependiendo de GPU)
- Validation: 10 min
- Optimization: 5 min
- Exploration: Manual
- Error analysis: 5 min

**Total por modelo:** ~1.5 horas

**Para suite completa (6 modelos):** ~9 horas (paralelizable)

---

## 11. Captura de Métricas {#metricas}

### Métricas Capturadas por Notebook

#### Notebook 02: Dataset Generation

```json
{
  "generation": {
    "num_problems": 1000,
    "problem_sizes": {"small": 200, "medium": 600, "large": 200},
    "generation_time_sec": 138.5,
    "avg_time_per_problem_ms": 138.5
  },
  "solver_execution": {
    "total_time_sec": 942.3,
    "avg_time_per_problem_ms": 942.3,
    "success_rate": 0.987,
    "avg_steps": 1234
  },
  "augmentation": {
    "original_samples": 1000,
    "augmented_samples": 5200,
    "expansion_factor": 5.2,
    "augmentation_time_sec": 72.1
  },
  "purification": {
    "input_samples": 5200,
    "valid_samples": 4904,
    "valid_rate": 0.943,
    "class_balance": {"positive": 0.512, "negative": 0.488},
    "purification_time_sec": 48.3
  },
  "total_time_sec": 1201.2
}
```

#### Notebook 03: Training

```json
{
  "config": {
    "model_name": "CostPredictor",
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "early_stopping_patience": 10
  },
  "training": {
    "total_epochs": 87,
    "total_time_sec": 1834.5,
    "avg_epoch_time_sec": 21.1,
    "best_epoch": 77,
    "best_val_loss": 0.0234,
    "final_train_loss": 0.0198,
    "final_val_loss": 0.0234
  },
  "convergence": {
    "epochs_to_90pct_best": 52,
    "early_stopped": true,
    "patience_triggered_at": 87
  },
  "resources": {
    "peak_gpu_memory_mb": 1247.3,
    "avg_gpu_utilization": 0.78,
    "total_gpu_hours": 0.51
  }
}
```

#### Notebook 04: Validation

```json
{
  "correctness": {
    "dimensions": "PASS",
    "value_range": "PASS",
    "invariances": "PASS",
    "determinism": "PASS"
  },
  "precision": {
    "mae": 0.123,
    "rmse": 0.187,
    "r2": 0.912,
    "mape": 8.7
  },
  "speedup": {
    "avg_speedup": 1.67,
    "time_baseline_ms": 523.4,
    "time_ml_ms": 313.2,
    "node_reduction_pct": 32.5
  },
  "overhead": {
    "inference_time_ms": 0.020,
    "throughput_per_sec": 50000,
    "model_memory_mb": 0.013,
    "overhead_pct": 3.2
  }
}
```

#### Notebook 05: Optimization

```json
{
  "original": {
    "size_kb": 13.26,
    "inference_ms": 0.020,
    "throughput": 50000,
    "mae": 0.123
  },
  "quantized": {
    "size_kb": 3.31,
    "inference_ms": 0.015,
    "throughput": 66667,
    "mae": 0.125,
    "size_reduction": 4.0,
    "speedup": 1.33
  },
  "onnx": {
    "size_kb": 3.50,
    "inference_ms": 0.003,
    "throughput": 333333,
    "mae": 0.123,
    "speedup": 6.67
  },
  "pruned": {
    "size_kb": 9.28,
    "inference_ms": 0.018,
    "throughput": 55556,
    "mae": 0.128,
    "params_removed_pct": 30.0,
    "speedup": 1.11
  }
}
```

### Exportación de Métricas

**Formatos:**
- JSON (machine-readable)
- CSV (para análisis en Excel/Pandas)
- Markdown (human-readable)
- TensorBoard logs (para visualización)

**Ubicación:**
```
/content/lattice-weaver/metrics/
├── dataset_generation_20251013_143022.json
├── training_CostPredictor_20251013_150134.json
├── validation_CostPredictor_20251013_153045.json
└── optimization_CostPredictor_20251013_154512.json
```

---

## 12. Manejo de Errores {#errores}

### Estrategia de Manejo

**Principios:**
1. **Fail-safe:** Nunca perder progreso
2. **Informative:** Mensajes de error claros
3. **Recoverable:** Sugerencias de recuperación
4. **Logged:** Todos los errores registrados

### Errores Comunes y Soluciones

#### Error 1: Out of Memory (OOM)

**Causa:** Batch size muy grande o modelo muy grande

**Detección:**
```python
try:
    loss.backward()
except RuntimeError as e:
    if "out of memory" in str(e):
        # OOM detected
```

**Solución automática:**
1. Reducir batch size a la mitad
2. Limpiar cache de GPU
3. Reintentar
4. Si persiste, usar gradient accumulation

**Código:**
```python
def handle_oom(config, model, optimizer):
    print(f"⚠️  OOM detected")
    print(f"   Reducing batch size: {config.batch_size} → {config.batch_size // 2}")
    
    config.batch_size = config.batch_size // 2
    torch.cuda.empty_cache()
    
    # Recrear dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    print("✅ Recovered, continuing training")
    return train_loader, val_loader
```

#### Error 2: NaN Loss

**Causa:** Learning rate muy alto, gradientes explotando, datos corruptos

**Detección:**
```python
if torch.isnan(loss):
    # NaN detected
```

**Solución automática:**
1. Cargar último checkpoint válido
2. Reducir learning rate 10x
3. Aplicar gradient clipping más agresivo
4. Reintentar

**Código:**
```python
def handle_nan_loss(model, optimizer, config):
    print("⚠️  NaN loss detected")
    
    # Cargar último checkpoint
    checkpoint = load_last_valid_checkpoint(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Reducir LR
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Reduced LR to {param_group['lr']}")
    
    return model, optimizer, checkpoint['epoch']
```

#### Error 3: Dataset Corrupto

**Causa:** Archivos dañados, formato incorrecto

**Detección:**
```python
try:
    dataset = torch.load('train.pt')
except Exception as e:
    # Dataset corrupto
```

**Solución:**
1. Intentar cargar backup
2. Si no hay backup, regenerar dataset
3. Validar integridad antes de usar

**Código:**
```python
def load_dataset_safe(path):
    try:
        dataset = torch.load(path)
        # Validar
        assert 'features' in dataset
        assert 'labels' in dataset
        assert len(dataset['features']) == len(dataset['labels'])
        return dataset
    except Exception as e:
        print(f"⚠️  Failed to load {path}: {e}")
        
        # Intentar backup
        backup_path = path.replace('.pt', '_backup.pt')
        if Path(backup_path).exists():
            print(f"   Loading backup from {backup_path}")
            return torch.load(backup_path)
        else:
            print("   No backup found, regenerating dataset...")
            return regenerate_dataset()
```

#### Error 4: Modelo No Converge

**Causa:** Arquitectura inadecuada, hiperparámetros malos, datos insuficientes

**Detección:**
```python
if epoch > 50 and val_loss > initial_val_loss * 0.9:
    # No convergencia después de 50 epochs
```

**Solución:**
1. Revisar hiperparámetros
2. Aumentar capacidad del modelo
3. Verificar calidad de datos
4. Generar reporte de diagnóstico

**Código:**
```python
def diagnose_non_convergence(metrics_history, config):
    print("⚠️  Model not converging")
    
    # Análisis
    initial_loss = metrics_history[0]['val_loss']
    current_loss = metrics_history[-1]['val_loss']
    improvement = (initial_loss - current_loss) / initial_loss
    
    suggestions = []
    
    if improvement < 0.1:
        suggestions.append("Try increasing learning rate (current: {config.learning_rate})")
    
    if config.model_name in ['CostPredictor'] and improvement < 0.2:
        suggestions.append("Try increasing model capacity (add more layers/neurons)")
    
    # Generar reporte
    report = f"""
    # Non-Convergence Diagnostic Report
    
    ## Metrics
    - Initial val loss: {initial_loss:.4f}
    - Current val loss: {current_loss:.4f}
    - Improvement: {improvement*100:.1f}%
    
    ## Suggestions
    """
    for i, sug in enumerate(suggestions, 1):
        report += f"{i}. {sug}\n"
    
    print(report)
    return suggestions
```

### Logging de Errores

**Todos los errores se registran en:**
```
/content/lattice-weaver/logs/errors/
├── error_20251013_143022.log
├── error_20251013_150134.log
└── ...
```

**Formato:**
```
[2025-10-13 14:30:22] ERROR in Notebook 03 (Training)
Type: RuntimeError (Out of Memory)
Message: CUDA out of memory. Tried to allocate 256.00 MiB
Context:
  - Model: CostPredictor
  - Batch size: 64
  - Epoch: 23
  - Step: 1234
Action taken:
  - Reduced batch size to 32
  - Cleared GPU cache
  - Resumed training
Status: RECOVERED
```

---

## 13. Reportes Automáticos {#reportes}

### Tipos de Reportes

#### 13.1. Reporte de Progreso

**Generado por:** Notebook 01 (Control Center)

**Frecuencia:** On-demand

**Contenido:**
- Estado general (modelos implementados, entrenados, optimizados)
- Progreso por suite
- Próximos pasos

**Formato:** Markdown

#### 13.2. Reporte de Dataset

**Generado por:** Notebook 02 (Dataset Generation)

**Frecuencia:** Después de cada generación

**Contenido:**
- Número de problemas generados
- Distribución de tamaños
- Factor de augmentation
- Métricas de calidad
- Tiempo de generación

**Formato:** Markdown + JSON

#### 13.3. Reporte de Entrenamiento

**Generado por:** Notebook 03 (Training)

**Frecuencia:** Después de cada entrenamiento

**Contenido:**
- Configuración de entrenamiento
- Métricas finales (train/val loss, accuracy, etc.)
- Curvas de aprendizaje (gráficas)
- Tiempo de entrenamiento
- Uso de recursos

**Formato:** Markdown + PNG (gráficas)

#### 13.4. Reporte de Validación

**Generado por:** Notebook 04 (Validation)

**Frecuencia:** Después de cada validación

**Contenido:**
- Tests de correctitud (PASS/FAIL)
- Métricas de precisión
- Benchmarks de speedup
- Overhead
- Conclusión (ready for deployment o no)

**Formato:** Markdown + PDF

#### 13.5. Reporte de Optimización

**Generado por:** Notebook 05 (Optimization)

**Frecuencia:** Después de cada optimización

**Contenido:**
- Comparación pre/post optimización
- Tabla de métricas
- Recomendación (qué versión usar)

**Formato:** Markdown

#### 13.6. Reporte de Errores

**Generado por:** Notebook 07 (Error Analysis)

**Frecuencia:** Después de análisis de errores

**Contenido:**
- Distribución de errores
- Patrones identificados
- Top 10 peores casos
- Sugerencias de corrección
- Próximos pasos

**Formato:** Markdown + HTML interactivo

### Exportación de Reportes

**Ubicación:**
```
/content/lattice-weaver/reports/
├── progress/
│   └── progress_20251013_143022.md
├── datasets/
│   └── dataset_csp_20251013_144530.md
├── training/
│   ├── training_CostPredictor_20251013_150134.md
│   └── training_CostPredictor_20251013_150134_curves.png
├── validation/
│   └── validation_CostPredictor_20251013_153045.pdf
├── optimization/
│   └── optimization_CostPredictor_20251013_154512.md
└── errors/
    └── errors_CostPredictor_20251013_155623.html
```

### Agregación de Reportes

**Reporte global:** Combina todos los reportes de una suite

**Generado por:** Script de agregación

**Contenido:**
- Resumen ejecutivo
- Estado de cada modelo
- Métricas comparativas
- Recomendaciones globales

**Formato:** PDF profesional (con gráficas, tablas, etc.)

---

## Conclusión

Esta suite de notebooks proporciona una **plataforma completa y automatizada** para el desarrollo, entrenamiento, validación y optimización de las 120 mini-IAs de LatticeWeaver.

**Características clave:**
- ✅ Automatización end-to-end
- ✅ Documentación exhaustiva
- ✅ Captura completa de métricas
- ✅ Manejo robusto de errores
- ✅ Interfaz interactiva
- ✅ Reportes automáticos
- ✅ Reproducibilidad total

**Próximos pasos:**
1. Ejecutar Notebook 01 (Control Center)
2. Seguir flujo recomendado
3. Entrenar primera suite (Costos y Memoización)
4. Validar y optimizar
5. Continuar con siguientes suites

---

**Documentación creada:** 13 de Octubre, 2025  
**Versión:** 1.0  
**Autor:** Sistema automatizado de LatticeWeaver ML

