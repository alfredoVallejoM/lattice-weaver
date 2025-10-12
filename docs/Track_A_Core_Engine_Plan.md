# Track A: Core Engine - Plan de Implementación Detallado

**Responsable:** Dev A (Core Developer)  
**Duración:** 8 semanas  
**Dependencias:** Ninguna (track independiente)

---

## Objetivo

Resolver el Issue 1 de backtracking, implementar herramientas de trazado y análisis del espacio de búsqueda, y crear la suite de experimentación masiva.

---

## Semana 1-2: Resolución Issue 1 + SearchSpaceTracer

### Tareas

#### 1.1 Implementar Solución Híbrida al Issue 1 (3 días)

**Archivos a modificar:**
- `lattice_weaver/arc_weaver/adaptive_consistency.py`

**Cambios:**
```python
class AdaptiveConsistencyEngine:
    def __init__(self, ..., propagation_strategy='selective'):
        self.propagation_strategy = propagation_strategy
        
    def _propagate_assignment(self, var, value):
        if self.propagation_strategy == 'selective':
            # Solo propagar a vecinos directos
            neighbors = self.cg.get_neighbors(var)
            return self._ac3_solver.enforce_arc_consistency(
                variables=neighbors
            )
        elif self.propagation_strategy == 'full':
            # AC-3 completo (comportamiento actual)
            return self._ac3_solver.enforce_arc_consistency()
```

**Tests a crear:**
- `test_selective_propagation_nqueens_4()`
- `test_selective_propagation_nqueens_8()`
- `test_full_vs_selective_comparison()`

**Checkpoint 1.1:** Tests pasando, N-Reinas n=8 resuelve en <1s

---

#### 1.2 Implementar SearchSpaceTracer (4 días)

**Archivo nuevo:**
- `lattice_weaver/arc_weaver/tracing.py` (≈500 líneas)

**Clases:**
```python
@dataclass
class SearchEvent:
    timestamp: float
    event_type: str  # 'assignment', 'backtrack', 'propagation', etc.
    variable: Optional[str]
    value: Optional[Any]
    domains_snapshot: Dict[str, Set]
    metadata: Dict[str, Any]

class SearchSpaceTracer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.events: List[SearchEvent] = []
        
    def record_assignment(self, var, value, domains):
        if not self.enabled:
            return
        self.events.append(SearchEvent(...))
        
    def record_backtrack(self, var, domains):
        ...
        
    def record_propagation(self, affected_vars, domains):
        ...
        
    def to_csv(self, path: str):
        """Exportar a CSV para análisis"""
        
    def to_json(self, path: str):
        """Exportar a JSON para visualización"""
        
    def get_statistics(self) -> Dict:
        """Estadísticas agregadas"""
        return {
            'total_assignments': ...,
            'total_backtracks': ...,
            'avg_domain_size': ...,
            'branching_factor': ...
        }
```

**Integración en ACE:**
```python
class AdaptiveConsistencyEngine:
    def __init__(self, ..., tracer: Optional[SearchSpaceTracer] = None):
        self.tracer = tracer or SearchSpaceTracer(enabled=False)
        
    def _backtrack_search(self, ...):
        var = self._select_variable()
        for value in domain:
            self.tracer.record_assignment(var, value, self.cg.domains)
            # ... resto del código
            if not consistent:
                self.tracer.record_backtrack(var, self.cg.domains)
```

**Tests a crear:**
- `test_tracer_records_assignments()`
- `test_tracer_records_backtracks()`
- `test_tracer_csv_export()`
- `test_tracer_json_export()`
- `test_tracer_statistics()`
- `test_tracer_disabled_no_overhead()`

**Checkpoint 1.2:** Tracer funcional, exportación CSV/JSON, overhead <5%

---

### Entregable Semana 2

**Archivos:**
- `adaptive_consistency.py` (modificado)
- `tracing.py` (nuevo)
- `test_issue1_resolution.py` (nuevo, 10 tests)
- `test_tracing.py` (nuevo, 15 tests)

**Documentación:**
- `docs/TRACING_GUIDE.md` - Guía de uso del tracer

**Validación:**
```bash
pytest tests/unit/test_issue1_resolution.py -v
pytest tests/unit/test_tracing.py -v
python examples/trace_nqueens_8.py  # Genera trace.csv
```

---

## Semana 3-4: SearchSpaceVisualizer

### Tareas

#### 2.1 Implementar Visualizador (5 días)

**Archivo nuevo:**
- `lattice_weaver/arc_weaver/visualization.py` (≈400 líneas)

**Clases:**
```python
class SearchSpaceVisualizer:
    def __init__(self, trace_file: str):
        self.events = self._load_trace(trace_file)
        
    def plot_search_tree(self, output='search_tree.html'):
        """Árbol de búsqueda interactivo con Plotly"""
        
    def plot_domain_evolution(self, variables=None, output='domains.html'):
        """Evolución de dominios en el tiempo"""
        
    def plot_timeline(self, output='timeline.html'):
        """Línea de tiempo de eventos"""
        
    def generate_report(self, output_dir='report/'):
        """Reporte HTML completo"""
```

**Implementación de plot_search_tree:**
```python
def plot_search_tree(self, output='search_tree.html'):
    import plotly.graph_objects as go
    
    # Construir árbol desde eventos
    nodes = []
    edges = []
    current_path = []
    
    for event in self.events:
        if event.event_type == 'assignment':
            node_id = len(nodes)
            nodes.append({
                'id': node_id,
                'label': f"{event.variable}={event.value}",
                'depth': len(current_path)
            })
            if current_path:
                edges.append((current_path[-1], node_id))
            current_path.append(node_id)
            
        elif event.event_type == 'backtrack':
            current_path.pop()
    
    # Crear figura con Plotly
    fig = go.Figure(data=[
        go.Scatter(
            x=[node['depth'] for node in nodes],
            y=list(range(len(nodes))),
            mode='markers+text',
            text=[node['label'] for node in nodes],
            ...
        )
    ])
    
    fig.write_html(output)
```

**Tests a crear:**
- `test_visualizer_loads_trace()`
- `test_plot_search_tree_generates_html()`
- `test_plot_domain_evolution()`
- `test_plot_timeline()`
- `test_generate_full_report()`

**Checkpoint 2.1:** Visualizaciones HTML generadas correctamente

---

#### 2.2 Crear Ejemplos y Documentación (2 días)

**Ejemplos a crear:**
- `examples/trace_and_visualize_nqueens.py`
- `examples/compare_algorithms_visually.py`
- `examples/analyze_backtracking_patterns.py`

**Documentación:**
- `docs/VISUALIZATION_GUIDE.md`

**Checkpoint 2.2:** Ejemplos ejecutables, documentación completa

---

### Entregable Semana 4

**Archivos:**
- `visualization.py` (nuevo)
- `test_visualization.py` (nuevo, 12 tests)
- 3 ejemplos ejecutables
- 2 guías de documentación

**Validación:**
```bash
pytest tests/unit/test_visualization.py -v
python examples/trace_and_visualize_nqueens.py
# Abrir report/index.html en navegador
```

---

## Semana 5-6: ExperimentRunner (Minería Masiva)

### Tareas

#### 3.1 Implementar ExperimentRunner (6 días)

**Archivo nuevo:**
- `lattice_weaver/benchmarks/experiment_runner.py` (≈600 líneas)

**Clases:**
```python
@dataclass
class ExperimentConfig:
    problem_type: str
    param_grid: Dict[str, List[Any]]
    algorithms: List[str]
    timeout: float
    repetitions: int

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, n_jobs: int = -1):
        self.config = config
        self.n_jobs = n_jobs
        
    def run_grid_search(self) -> pd.DataFrame:
        """Ejecutar grid search completo"""
        from concurrent.futures import ProcessPoolExecutor, TimeoutError
        
        # Generar todas las combinaciones
        experiments = self._generate_experiments()
        
        # Ejecutar en paralelo
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self._run_single_experiment, exp)
                for exp in experiments
            ]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.config.timeout)
                    results.append(result)
                except TimeoutError:
                    results.append({'status': 'timeout', ...})
                    
        return pd.DataFrame(results)
        
    def _run_single_experiment(self, exp_config):
        """Ejecutar un experimento individual"""
        problem = self._create_problem(exp_config)
        solver = self._create_solver(exp_config)
        
        start = time.time()
        stats = solver.solve(problem)
        elapsed = time.time() - start
        
        return {
            'problem_type': exp_config['problem_type'],
            'algorithm': exp_config['algorithm'],
            **exp_config['params'],
            'time': elapsed,
            'nodes': stats.nodes_explored,
            'backtracks': stats.backtracks,
            'success': stats.solutions_found > 0
        }
        
    def visualize_results(self, df: pd.DataFrame, output_dir='results/'):
        """Generar visualizaciones de resultados"""
        # Heatmaps, scatter plots, etc.
```

**Tests a crear:**
- `test_experiment_runner_grid_search()`
- `test_experiment_runner_parallel_execution()`
- `test_experiment_runner_timeout_handling()`
- `test_experiment_runner_csv_export()`
- `test_experiment_runner_visualization()`

**Checkpoint 3.1:** ExperimentRunner funcional, ejecución paralela

---

#### 3.2 Crear Suite de Experimentos Predefinidos (2 días)

**Archivo nuevo:**
- `lattice_weaver/benchmarks/experiment_suites.py`

```python
def nqueens_scalability_suite() -> ExperimentConfig:
    """Suite para analizar escalabilidad en N-Reinas"""
    return ExperimentConfig(
        problem_type='nqueens',
        param_grid={
            'n': [4, 6, 8, 10, 12, 14, 16],
            'propagation_strategy': ['selective', 'full']
        },
        algorithms=['ace', 'backtracking', 'forward_checking'],
        timeout=60.0,
        repetitions=5
    )

def graph_coloring_density_suite() -> ExperimentConfig:
    """Suite para analizar impacto de densidad en coloreo"""
    return ExperimentConfig(
        problem_type='graph_coloring',
        param_grid={
            'n_nodes': [10, 20, 30],
            'density': [0.3, 0.5, 0.7, 0.9],
            'n_colors': [3, 4, 5]
        },
        algorithms=['ace', 'backtracking'],
        timeout=30.0,
        repetitions=3
    )
```

**Checkpoint 3.2:** Suites predefinidas, fáciles de ejecutar

---

### Entregable Semana 6

**Archivos:**
- `experiment_runner.py` (nuevo)
- `experiment_suites.py` (nuevo)
- `test_experiment_runner.py` (nuevo, 15 tests)

**Ejemplo de uso:**
```python
from lattice_weaver.benchmarks import ExperimentRunner, nqueens_scalability_suite

runner = ExperimentRunner(nqueens_scalability_suite(), n_jobs=8)
results = runner.run_grid_search()
results.to_csv('nqueens_scalability.csv')
runner.visualize_results(results, output_dir='results/')
```

**Validación:**
```bash
pytest tests/unit/test_experiment_runner.py -v
python examples/run_scalability_experiment.py
# Genera nqueens_scalability.csv y visualizaciones
```

---

## Semana 7-8: Integración y Optimización

### Tareas

#### 4.1 Optimizar Overhead del Tracer (2 días)

**Objetivo:** Reducir overhead a <2% cuando habilitado

**Técnicas:**
- Lazy evaluation de snapshots
- Sampling configurable (grabar 1 de cada N eventos)
- Buffer de eventos en memoria

**Checkpoint 4.1:** Benchmark muestra overhead <2%

---

#### 4.2 Crear Dashboard Integrado (3 días)

**Archivo nuevo:**
- `lattice_weaver/dashboard/app.py` (Flask app simple)

**Funcionalidad:**
- Subir archivo de trace
- Visualizar en tiempo real
- Comparar múltiples traces
- Exportar reportes

**Checkpoint 4.2:** Dashboard funcional localmente

---

#### 4.3 Documentación Final y Ejemplos (3 días)

**Documentos a crear:**
- `docs/CORE_ENGINE_GUIDE.md` - Guía completa
- `docs/EXPERIMENTATION_GUIDE.md` - Guía de experimentación
- `CHANGELOG_v4.2.md` - Changelog detallado

**Ejemplos a crear:**
- `examples/full_workflow_nqueens.py` - Workflow completo
- `examples/compare_strategies.py` - Comparación de estrategias
- `examples/massive_experiment.py` - Experimento masivo

**Checkpoint 4.3:** Documentación completa, ejemplos ejecutables

---

### Entregable Semana 8 (Final Track A)

**Resumen de Logros:**
- ✅ Issue 1 resuelto (N-Reinas n=8 en <1s)
- ✅ SearchSpaceTracer funcional (overhead <2%)
- ✅ SearchSpaceVisualizer con reportes HTML
- ✅ ExperimentRunner para minería masiva
- ✅ Dashboard simple para visualización
- ✅ Documentación completa
- ✅ 52+ tests nuevos

**Archivos entregados:**
- 5 módulos Python (≈2,000 líneas)
- 52+ tests unitarios
- 6+ ejemplos ejecutables
- 4 guías de documentación
- Dashboard Flask

**Métricas:**
- Speedup N-Reinas: 10-20x
- Overhead tracer: <2%
- Cobertura tests: >90%
- Experimentos paralelos: 8x speedup con 8 cores

---

## Puntos de Sincronización con Otros Tracks

### Sync Point 1 (Semana 2)
**Con Track D (Inference Engine):**
- Compartir formato de trace para que el inference engine pueda generar traces sintéticos

### Sync Point 2 (Semana 4)
**Con Track E (Web App):**
- Definir API REST para subir traces y obtener visualizaciones

### Sync Point 3 (Semana 6)
**Con Track C (Problem Families):**
- Integrar generadores de familias de problemas en ExperimentRunner

### Sync Point 4 (Semana 8)
**Con todos los tracks:**
- Integración final y validación cruzada

---

## Riesgos y Mitigaciones

### Riesgo 1: Overhead del Tracer
**Mitigación:** Implementar sampling y lazy evaluation desde el inicio

### Riesgo 2: Escalabilidad de Visualizaciones
**Mitigación:** Limitar número de eventos visualizados, ofrecer agregación

### Riesgo 3: Timeout en Experimentos Masivos
**Mitigación:** Timeout por experimento, manejo robusto de excepciones

---

## Checklist de Completitud

- [ ] Issue 1 resuelto (N-Reinas n=8 <1s)
- [ ] SearchSpaceTracer implementado
- [ ] Exportación CSV/JSON funcional
- [ ] SearchSpaceVisualizer implementado
- [ ] Reportes HTML generados
- [ ] ExperimentRunner implementado
- [ ] Ejecución paralela funcional
- [ ] Suites predefinidas creadas
- [ ] Dashboard Flask funcional
- [ ] 52+ tests pasando
- [ ] Documentación completa
- [ ] Ejemplos ejecutables
- [ ] Overhead <2%
- [ ] Cobertura >90%

---

**Estado:** ✅ PLAN COMPLETO Y DETALLADO

