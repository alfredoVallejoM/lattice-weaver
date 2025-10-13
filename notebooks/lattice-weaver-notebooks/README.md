# LatticeWeaver ML Notebooks

Suite completa de notebooks para entrenamiento, validación y optimización de mini-IAs.

## 📚 Notebooks Disponibles

### 01. Control Center (`01_Control_Center.ipynb`)
- Dashboard interactivo con estado del proyecto
- Menú principal de navegación
- Generador de reportes de progreso
- **Formato:** Jupyter Notebook (.ipynb)

### 02. Dataset Generation (`02_Dataset_Generation.py`)
- Generación de problemas sintéticos
- Data augmentation (5x expansión)
- Purificación y normalización
- Train/val/test split
- **Líneas:** 405

### 03. Training (`03_Training.py`)
- Entrenamiento automatizado con early stopping
- Monitoreo en tiempo real
- Checkpointing automático
- Evaluación en test set
- **Líneas:** 374

### 04. Validation & Benchmarks (`04_Validation_Benchmarks.py`)
- Validación de precisión (MAE, RMSE, R²)
- Benchmarks de velocidad de inferencia
- Análisis de overhead
- Estimación de speedup
- **Líneas:** 470

### 05. Optimization (`05_Optimization.py`)
- Cuantización dinámica (INT8)
- Export a ONNX
- Comparación pre/post optimización
- **Líneas:** 456

### 06. Model Explorer (`06_Model_Explorer.py`)
- Inferencia interactiva
- Análisis de sensibilidad
- Inspección de arquitectura
- Feature importance
- **Líneas:** 404

### 07. Error Analysis (`07_Error_Analysis.py`)
- Análisis de distribución de errores
- Detección de patrones
- Top 10 peores casos
- Sugerencias de mejora
- **Líneas:** 457

### Utilidades (`ml_utils.py`)
- Clases compartidas por todos los notebooks
- ModelRegistry, Trainer, Validator, etc.
- **Líneas:** 667

**Total:** ~3,233 líneas de código documentado

---

## 🚀 Uso Rápido

### En Google Colab

1. **Subir archivos:**
   ```python
   # En Colab, subir todos los archivos .py
   from google.colab import files
   uploaded = files.upload()
   ```

2. **Ejecutar notebooks en orden:**
   ```bash
   !python 02_Dataset_Generation.py
   !python 03_Training.py
   !python 04_Validation_Benchmarks.py
   !python 05_Optimization.py
   !python 06_Model_Explorer.py
   !python 07_Error_Analysis.py
   ```

3. **O convertir a notebooks:**
   ```bash
   !pip install jupytext
   !jupytext --to notebook 02_Dataset_Generation.py
   # Repetir para cada archivo
   ```

### Localmente

```bash
# Ejecutar directamente
python 02_Dataset_Generation.py
python 03_Training.py
# ...

# O convertir a notebooks
jupytext --to notebook *.py
jupyter notebook
```

---

## 📋 Flujo de Trabajo Recomendado

```
1. Dataset Generation (20 min)
   ↓
2. Training (30-60 min)
   ↓
3. Validation & Benchmarks (10 min)
   ↓
4. Optimization (5 min)
   ↓
5. Model Explorer (interactivo)
   ↓
6. Error Analysis (5 min)
```

**Tiempo total:** ~1.5 horas por modelo

---

## ⚙️ Configuración

Todos los notebooks usan configuración centralizada. Editar al inicio de cada archivo:

```python
CONFIG = {
    'model_name': 'CostPredictor',
    'suite_name': 'costs_memoization',
    'batch_size': 32,
    'learning_rate': 1e-3,
    'num_epochs': 100,
    # ...
}
```

---

## 📊 Métricas Capturadas

Cada notebook genera métricas detalladas:

- **Dataset Generation:** Tiempo, tamaño, factor de augmentation
- **Training:** Loss curves, learning rate, epoch time
- **Validation:** MAE, RMSE, R², speedup, overhead
- **Optimization:** Size reduction, speedup, accuracy
- **Error Analysis:** Error distribution, patterns, suggestions

Todas las métricas se exportan a:
- JSON (machine-readable)
- Markdown (human-readable)
- PNG (visualizaciones)

---

## 📁 Estructura de Salida

```
/content/lattice-weaver/
├── datasets/
│   └── csp/
│       ├── train.pt
│       ├── val.pt
│       ├── test.pt
│       └── normalization_stats.json
├── models/
│   └── costs_memoization/
│       ├── CostPredictor_best.pt
│       ├── CostPredictor_quantized.pt
│       └── CostPredictor.onnx
├── reports/
│   ├── dataset_generation_*.md
│   ├── training_*.md
│   ├── validation_*.md
│   ├── optimization_*.md
│   └── error_analysis_*.md
└── logs/
    └── errors/
```

---

## 🛠️ Dependencias

```python
# Core
torch >= 1.10
numpy >= 1.20
matplotlib >= 3.3
seaborn >= 0.11

# Optional (para optimización)
onnx >= 1.10
onnxruntime >= 1.10

# LatticeWeaver (si disponible)
lattice_weaver
```

Instalación en Colab:
```bash
!pip install torch numpy matplotlib seaborn onnx onnxruntime
```

---

## 🔧 Troubleshooting

### Error: "Module not found"
```python
# Asegurarse de que ml_utils.py está en el mismo directorio
import sys
sys.path.insert(0, str(Path.cwd()))
```

### Error: "Dataset not found"
```python
# Ejecutar primero 02_Dataset_Generation.py
!python 02_Dataset_Generation.py
```

### Error: "CUDA out of memory"
```python
# Reducir batch size en CONFIG
CONFIG['batch_size'] = 16  # o menor
```

---

## 📖 Documentación Adicional

- **Documentación completa:** `docs/suite_notebooks_documentacion_completa.md` (2,137 líneas)
- **Análisis ML:** `docs/ML_VISION.md`
- **Roadmap:** `README.md` (sección ML)

---

## ✅ Checklist de Validación

Antes de considerar un modelo listo para producción:

- [ ] Dataset generado (> 1K samples)
- [ ] Modelo entrenado (val loss < 0.05)
- [ ] Validación pasada (MAE < 0.2, R² > 0.8)
- [ ] Overhead aceptable (< 5%)
- [ ] Speedup positivo (> 1.5x)
- [ ] Modelo optimizado (ONNX exportado)
- [ ] Errores analizados (< 10% large errors)

---

## 🎯 Próximos Pasos

1. **Entrenar suite completa** (6 modelos de Costos y Memoización)
2. **Implementar suites adicionales** (Renormalización, TDA, etc.)
3. **Integrar en LatticeWeaver** (MLAugmentedArcEngine)
4. **Benchmarks en problemas reales**
5. **Deployment en producción**

---

## 📞 Soporte

Para preguntas o problemas:
1. Revisar documentación completa
2. Verificar logs de error en `/content/lattice-weaver/logs/`
3. Abrir issue en GitHub

---

**Creado:** 2025-10-13  
**Versión:** 1.0  
**Autor:** LatticeWeaver ML Team

