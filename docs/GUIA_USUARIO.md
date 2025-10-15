# Guía de Usuario - Fibration Flow

**Versión**: 1.0  
**Fecha**: 15 de Octubre, 2025  
**Autor**: Agente Autónomo - Lattice Weaver

---

## Introducción

Bienvenido a Fibration Flow, un solver avanzado de problemas de satisfacción de restricciones (CSP) que soporta restricciones jerárquicas (HARD/SOFT), optimización multi-objetivo y búsqueda adaptativa. Esta guía te ayudará a comenzar a usar Fibration Flow y aprovechar sus capacidades avanzadas.

## Instalación

### Requisitos

- Python 3.11+
- NumPy
- Numba (opcional, para JIT compilation)
- psutil (opcional, para profiling automático)

### Instalación desde Código Fuente

```bash
# Clonar repositorio
git clone https://github.com/alfredoVallejoM/lattice-weaver.git
cd lattice-weaver

# Instalar en modo desarrollo
pip install -e .

# Instalar dependencias opcionales
pip install numba psutil
```

## Conceptos Básicos

### Problemas CSP

Un problema de satisfacción de restricciones (CSP) consiste en:
- **Variables**: Elementos que necesitan valores asignados
- **Dominios**: Conjuntos de valores posibles para cada variable
- **Restricciones**: Reglas que limitan las combinaciones válidas de valores

### Restricciones Jerárquicas

Fibration Flow soporta tres niveles de restricciones:

- **LOCAL**: Restricciones entre pares de variables (ej: `X != Y`)
- **PATTERN**: Restricciones sobre patrones de variables (ej: "todas diferentes")
- **GLOBAL**: Restricciones sobre todas las variables (ej: "minimizar costo total")

### Restricciones HARD vs SOFT

- **HARD**: Deben satisfacerse obligatoriamente
- **SOFT**: Preferencias que se intentan satisfacer pero pueden violarse

## Uso Básico

### Ejemplo 1: N-Queens

```python
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness
from lattice_weaver.fibration.hacification_engine_optimized import HacificationEngineOptimized
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
from lattice_weaver.arc_engine.core import ArcEngine

# Crear jerarquía de restricciones
n = 8
hierarchy = ConstraintHierarchy()

# Definir dominios: cada reina puede estar en cualquier fila
domains = {f"Q{i}": list(range(n)) for i in range(n)}

# Añadir restricciones: no dos reinas se atacan
for i in range(n):
    for j in range(i + 1, n):
        def no_attack(assignment, i=i, j=j):
            qi = assignment.get(f"Q{i}")
            qj = assignment.get(f"Q{j}")
            if qi is None or qj is None:
                return True
            # Misma fila, columna o diagonal
            return qi != qj and abs(qi - qj) != abs(i - j)
        
        hierarchy.add_local_constraint(
            f"Q{i}", f"Q{j}",
            no_attack,
            Hardness.HARD
        )

# Crear motor de hacificación
landscape = EnergyLandscapeOptimized(hierarchy)
arc_engine = ArcEngine()
engine = HacificationEngineOptimized(hierarchy, landscape, arc_engine)

# Resolver
result = engine.hacify(domains, assignment={})

if result:
    print("Solución encontrada:")
    for var, val in result.assignment.items():
        print(f"  {var} = {val}")
    print(f"Energía: {result.energy.total_energy}")
else:
    print("No se encontró solución")
```

### Ejemplo 2: Asignación con Preferencias

```python
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness

# Crear jerarquía
hierarchy = ConstraintHierarchy()

# Variables: tareas a trabajadores
domains = {
    "T1": ["W1", "W2", "W3"],
    "T2": ["W1", "W2", "W3"],
    "T3": ["W1", "W2", "W3"],
    "T4": ["W1", "W2", "W3"]
}

# Restricción HARD: T1 y T2 no pueden estar en el mismo trabajador
def different_workers(assignment):
    t1 = assignment.get("T1")
    t2 = assignment.get("T2")
    if t1 is None or t2 is None:
        return True
    return t1 != t2

hierarchy.add_pattern_constraint(
    ["T1", "T2"],
    different_workers,
    Hardness.HARD
)

# Restricción SOFT: Preferir que T1 esté en W1
def prefer_w1_for_t1(assignment):
    t1 = assignment.get("T1")
    if t1 is None:
        return True
    return t1 == "W1"

hierarchy.add_local_constraint(
    "T1", "T1",  # Restricción unaria
    prefer_w1_for_t1,
    Hardness.SOFT,
    weight=2.0
)

# Restricción SOFT: Balancear carga
def balanced_load(assignment):
    workers = ["W1", "W2", "W3"]
    counts = {w: 0 for w in workers}
    for task in ["T1", "T2", "T3", "T4"]:
        worker = assignment.get(task)
        if worker:
            counts[worker] += 1
    # Penalizar si algún trabajador tiene más de 2 tareas
    return all(count <= 2 for count in counts.values())

hierarchy.add_global_constraint(
    ["T1", "T2", "T3", "T4"],
    balanced_load,
    Hardness.SOFT,
    weight=1.0
)

# Resolver (mismo código que antes)
landscape = EnergyLandscapeOptimized(hierarchy)
arc_engine = ArcEngine()
engine = HacificationEngineOptimized(hierarchy, landscape, arc_engine)

result = engine.hacify(domains, assignment={})

if result:
    print("Solución encontrada:")
    for var, val in result.assignment.items():
        print(f"  {var} = {val}")
    print(f"Energía: {result.energy.total_energy}")
    print(f"  HARD: {result.energy.hard_energy}")
    print(f"  SOFT: {result.energy.soft_energy}")
```

## Uso Avanzado

### Solver Adaptativo

El solver adaptativo analiza automáticamente las características del problema y activa optimizaciones apropiadas:

```python
from lattice_weaver.fibration.fibration_search_solver_adaptive_v2 import FibrationSearchSolverAdaptiveV2

# Crear solver adaptativo
solver = FibrationSearchSolverAdaptiveV2(
    hierarchy=hierarchy,
    initial_domains=domains,
    homotopy_threshold=100,  # Activar HomotopyRules después de 100 backtracks
    enable_profiling=True     # Activar profiling automático
)

# Resolver
solution = solver.solve()

if solution:
    print("Solución encontrada:")
    print(f"  Asignación: {solution.assignment}")
    print(f"  Energía: {solution.energy}")
    
    # Ver estadísticas
    stats = solver.get_stats()
    print(f"  Backtracks: {stats.get('backtracks', 0)}")
    print(f"  Modo usado: {stats.get('mode', 'unknown')}")
```

### Restricciones Globales Especializadas

Fibration Flow incluye restricciones globales optimizadas:

```python
from lattice_weaver.fibration.global_constraints import (
    AllDifferentConstraint,
    CumulativeConstraint
)

# AllDifferent: todas las variables deben tener valores diferentes
all_diff = AllDifferentConstraint(
    variables=["Q0", "Q1", "Q2", "Q3"],
    hierarchy=hierarchy
)
all_diff.post()  # Añade la restricción a la jerarquía

# Cumulative: para scheduling con recursos limitados
cumulative = CumulativeConstraint(
    tasks=["T1", "T2", "T3"],
    durations=[2, 3, 1],
    resources=[1, 2, 1],
    capacity=3,
    hierarchy=hierarchy
)
cumulative.post()
```

### Búsqueda Híbrida

Combina búsqueda sistemática con búsqueda local:

```python
from lattice_weaver.fibration.hybrid_search import HybridSearchSolver, SearchStrategy

# Crear solver híbrido
hybrid_solver = HybridSearchSolver(
    hierarchy=hierarchy,
    initial_domains=domains,
    strategy=SearchStrategy.SIMULATED_ANNEALING,
    max_iterations=1000,
    temperature=100.0
)

# Resolver
solution = hybrid_solver.solve()
```

### Profiling Automático

El AutoProfiler analiza el problema y recomienda optimizaciones:

```python
from lattice_weaver.utils.auto_profiler import AutoProfiler

# Crear profiler
profiler = AutoProfiler(
    profile_duration=10,  # Perfilar durante 10 backtracks
    min_backtracks=5
)

# Analizar problema
profiler.start_profiling()

# ... ejecutar solver ...

profiler.stop_profiling()

# Obtener recomendaciones
recommendations = profiler.get_recommendations()
print("Recomendaciones:")
for rec in recommendations:
    print(f"  - {rec}")

# Ver estadísticas
stats = profiler.get_stats()
print(f"Características del problema:")
print(f"  Variables: {stats['num_variables']}")
print(f"  Restricciones: {stats['num_constraints']}")
print(f"  Tamaño promedio de dominio: {stats['avg_domain_size']:.1f}")
```

## Configuración de Optimizaciones

### JIT Compilation

Para problemas medianos/grandes, activar JIT compilation:

```python
from lattice_weaver.utils.jit_compiler import get_jit_compiler

# Obtener compilador JIT
jit_compiler = get_jit_compiler()

# Configurar
jit_compiler.enable()  # Activar compilación JIT
jit_compiler.set_parallel(True)  # Activar paralelización

# El compilador se usará automáticamente en operaciones críticas
```

### Object Pooling

Para reducir allocations en problemas grandes:

```python
from lattice_weaver.utils.object_pool import get_list_pool, get_dict_pool

# Obtener pools globales
list_pool = get_list_pool()
dict_pool = get_dict_pool()

# Usar pools en código crítico
with list_pool.acquire_context() as lst:
    # Usar lista del pool
    lst.extend(range(100))
    # ... operaciones ...
    # La lista se devuelve automáticamente al pool al salir del context
```

### Configuración Manual de Modo

Si el modo adaptativo no es adecuado, configurar manualmente:

```python
from lattice_weaver.fibration.fibration_search_solver_adaptive_v2 import (
    FibrationSearchSolverAdaptiveV2,
    SolverMode
)

# Forzar modo específico
solver = FibrationSearchSolverAdaptiveV2(
    hierarchy=hierarchy,
    initial_domains=domains,
    force_mode=SolverMode.FULL  # LITE, MEDIUM, o FULL
)
```

## Mejores Prácticas

### 1. Usar Restricciones Apropiadas

- **LOCAL** para restricciones binarias simples
- **PATTERN** para restricciones sobre subconjuntos de variables
- **GLOBAL** para restricciones que involucran todas las variables

### 2. Balancear HARD y SOFT

- Usar **HARD** solo para restricciones que deben satisfacerse
- Usar **SOFT** para preferencias y optimización
- Ajustar **weights** para priorizar restricciones SOFT

### 3. Optimizar Dominios

- Usar dominios más pequeños cuando sea posible
- Pre-procesar para eliminar valores imposibles
- Considerar usar restricciones globales especializadas

### 4. Configurar Optimizaciones

- Para problemas pequeños (<30 vars): Usar modo LITE o desactivar optimizaciones
- Para problemas medianos (30-100 vars): Usar modo MEDIUM con JIT
- Para problemas grandes (>100 vars): Usar modo FULL con todas las optimizaciones

### 5. Monitorear Rendimiento

- Usar AutoProfiler para entender características del problema
- Revisar estadísticas después de resolver
- Ajustar configuración basándose en resultados

## Troubleshooting

### Problema: Solver muy lento en problemas pequeños

**Solución**: Desactivar optimizaciones o usar modo LITE:

```python
solver = FibrationSearchSolverAdaptiveV2(
    hierarchy=hierarchy,
    initial_domains=domains,
    force_mode=SolverMode.LITE
)
```

### Problema: No encuentra solución

**Verificar**:
1. ¿Las restricciones HARD son satisfacibles?
2. ¿Los dominios iniciales son suficientemente grandes?
3. ¿Hay conflictos entre restricciones?

**Solución**: Usar modo debug para ver qué restricciones fallan:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Ejecutar solver con logging activado
result = engine.hacify(domains, assignment={})
```

### Problema: Solución subóptima

**Solución**: Ajustar weights de restricciones SOFT o usar búsqueda híbrida:

```python
# Aumentar weight de restricción importante
hierarchy.add_global_constraint(
    variables,
    predicate,
    Hardness.SOFT,
    weight=10.0  # Aumentar de 1.0 a 10.0
)

# O usar búsqueda híbrida
hybrid_solver = HybridSearchSolver(
    hierarchy=hierarchy,
    initial_domains=domains,
    strategy=SearchStrategy.SIMULATED_ANNEALING
)
```

## Ejemplos Adicionales

Ver el directorio `examples/` para más ejemplos:
- `examples/n_queens.py`: N-Queens completo
- `examples/graph_coloring.py`: Graph Coloring
- `examples/job_shop_scheduling.py`: Job Shop Scheduling
- `examples/sudoku.py`: Sudoku solver

## Referencias

- **Documentación de API**: Ver `docs/API.md`
- **Principios de Diseño**: Ver `docs/META_PRINCIPIOS_DISENO.md`
- **Análisis de Dependencias**: Ver `docs/ANALISIS_DEPENDENCIAS_ESTRUCTURA.md`
- **Benchmarks**: Ver `benchmarks/` para ejemplos de uso avanzado

## Soporte

Para preguntas, reportar bugs o contribuir:
- **GitHub**: https://github.com/alfredoVallejoM/lattice-weaver
- **Issues**: https://github.com/alfredoVallejoM/lattice-weaver/issues

---

**Versión**: 1.0  
**Última Actualización**: 15 de Octubre, 2025  
**Mantenido por**: Equipo de Desarrollo de Lattice Weaver

