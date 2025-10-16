# Fibration Flow Benchmarks

Este directorio contiene benchmarks para evaluar la performance del Fibration Flow en diferentes escenarios y tamaños de problemas.

## Estado Actual

**Nota:** Algunos benchmarks tienen imports comentados temporalmente porque dependen de componentes que se integrarán en fases futuras (ArcEngine, solvers optimizados, etc.). Se irán activando progresivamente a medida que se completen las fases de reintegración.

Ver: `docs/FIBRATION_REINTEGRATION_TRACKING.md` para el estado de la reintegración.

## Benchmarks Disponibles

### Benchmarks Core (Funcionales)

Estos benchmarks usan solo componentes existentes y están completamente funcionales:

1. **`circuit_design_problem.py`**
   - Problema de diseño de circuitos
   - Usa: `FibrationSearchSolver`

2. **`complex_multiobjective_benchmark.py`**
   - Benchmark multi-objetivo complejo
   - Usa: `FibrationSearchSolver`

3. **`comprehensive_benchmark.py`**
   - Benchmark comprehensivo de funcionalidades
   - Usa: `FibrationSearchSolver`, `EnergyLandscapeOptimized`

4. **`fibration_optimized_benchmark.py`**
   - Benchmark de optimizaciones de fibration
   - Usa: `FibrationSearchSolver`

5. **`fibration_soft_optimization_benchmark.py`**
   - Benchmark de optimización con restricciones SOFT
   - Usa: `FibrationSearchSolver`

6. **`fibration_vs_baseline.py`**
   - Comparación con baseline
   - Usa: `FibrationSearchSolver`

7. **`final_soft_benchmark.py`**
   - Benchmark final de restricciones SOFT
   - Usa: `FibrationSearchSolver`

8. **`network_config_problem.py`**
   - Problema de configuración de red
   - Usa: `FibrationSearchSolver`

9. **`scalability_benchmark.py`**
   - Benchmark de escalabilidad
   - Usa: `FibrationSearchSolver`

10. **`soft_constraints_benchmark.py`**
    - Benchmark de restricciones SOFT
    - Usa: `FibrationSearchSolver`

11. **`soft_optimization_benchmark.py`**
    - Benchmark de optimización SOFT
    - Usa: `FibrationSearchSolver`

### Benchmarks Parcialmente Funcionales

Estos benchmarks tienen algunos imports comentados pero pueden ejecutarse parcialmente:

12. **`adaptive_solver_benchmark.py`** ⚠️
    - Benchmark de solver adaptativo
    - **Pendiente:** `FibrationSearchSolverEnhanced`, `FibrationSearchSolverAdaptive`, `ArcEngine`
    - Se activará en: Fase 5 (ArcEngine) y Fase 6 (Solvers avanzados)

13. **`fibration_flow_performance.py`** ⚠️
    - Benchmark de performance general
    - **Pendiente:** `FibrationSearchSolverEnhanced`, `HacificationEngineOptimized`, `ArcEngine`
    - Se activará en: Fase 4 (HacificationEngine), Fase 5 (ArcEngine), Fase 6 (Solvers)

14. **`final_comprehensive_benchmark.py`** ⚠️
    - Benchmark comprehensivo final
    - **Pendiente:** `FibrationSearchSolverEnhanced`, `FibrationSearchSolverAdaptiveV2`, `ArcEngine`
    - Se activará en: Fase 5 (ArcEngine) y Fase 6 (Solvers avanzados)

15. **`hacification_benchmark.py`** ⚠️
    - Benchmark de HacificationEngine
    - **Pendiente:** `HacificationEngineOptimized`, `ArcEngine`
    - Se activará en: Fase 4 (HacificationEngine) y Fase 5 (ArcEngine)

16. **`job_shop_scheduling_benchmark.py`** ⚠️
    - Benchmark de Job Shop Scheduling
    - **Pendiente:** `FibrationSearchSolverEnhanced`, `ArcEngine`
    - Se activará en: Fase 5 (ArcEngine) y Fase 6 (Solvers avanzados)

17. **`state_of_the_art_comparison.py`** ⚠️
    - Comparación con state-of-the-art
    - **Pendiente:** `FibrationSearchSolverEnhanced`, `ArcEngine`
    - Se activará en: Fase 5 (ArcEngine) y Fase 6 (Solvers avanzados)

18. **`task_assignment_with_preferences.py`** ⚠️
    - Benchmark de asignación de tareas con preferencias
    - **Pendiente:** `FibrationSearchSolverEnhanced`, `ArcEngine`
    - Se activará en: Fase 5 (ArcEngine) y Fase 6 (Solvers avanzados)

## Uso

### Ejecutar un Benchmark Individual

```python
# Ejemplo: Benchmark de escalabilidad
python3.11 benchmarks/scalability_benchmark.py
```

### Ejecutar Todos los Benchmarks Funcionales

```bash
# Ejecutar solo los benchmarks que no tienen dependencias pendientes
for bench in circuit_design_problem.py complex_multiobjective_benchmark.py comprehensive_benchmark.py; do
    python3.11 benchmarks/$bench
done
```

## Estructura de un Benchmark

Cada benchmark típicamente incluye:

```python
from lattice_weaver.fibration.fibration_search_solver import FibrationSearchSolver
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy
# ... otros imports ...

def create_problem(size):
    """Crea un problema de tamaño dado"""
    # ... definir variables, dominios, restricciones ...
    return variables, domains, hierarchy

def run_benchmark(max_size=10):
    """Ejecuta el benchmark"""
    results = []
    for size in range(2, max_size + 1):
        variables, domains, hierarchy = create_problem(size)
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        
        start = time.time()
        solution = solver.solve()
        end = time.time()
        
        results.append({
            'size': size,
            'time': end - start,
            'solution_found': solution is not None
        })
    
    return results

if __name__ == '__main__':
    results = run_benchmark()
    print(json.dumps(results, indent=2))
```

## Métricas Típicas

Los benchmarks suelen medir:

- **Tiempo de ejecución** (wall-clock time)
- **Uso de memoria** (peak memory, allocations)
- **Número de backtracks**
- **Número de nodos explorados**
- **Calidad de soluciones** (energía final para problemas de optimización)
- **Escalabilidad** (cómo varía el tiempo con el tamaño del problema)

## Roadmap de Activación

| Fase | Componente | Benchmarks Activados |
|------|-----------|----------------------|
| **Fase 1** (actual) | Benchmarks core | 11 funcionales ✓ |
| **Fase 4** | HacificationEngine refactorizado | `hacification_benchmark.py`, `fibration_flow_performance.py` |
| **Fase 5** | ArcEngine opcional | Todos los benchmarks con ArcEngine |
| **Fase 6** | Solvers avanzados | Todos los benchmarks con solvers optimizados |

## Contribuir

Al añadir nuevos benchmarks:

1. Usar solo componentes existentes o marcar claramente las dependencias futuras
2. Incluir docstring explicando qué mide el benchmark
3. Seguir la estructura estándar (create_problem, run_benchmark, main)
4. Añadir entrada en este README

## Referencias

- **Análisis de problemas del merge original:** `/home/ubuntu/analisis_fibration_merge.md`
- **Plan de reintegración:** `/home/ubuntu/plan_reintegracion_fibration_flow.md`
- **Tracking de reintegración:** `docs/FIBRATION_REINTEGRATION_TRACKING.md`

