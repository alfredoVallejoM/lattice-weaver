# Performance Utilities

Este directorio contiene utilidades de performance que pueden ser usadas opcionalmente para optimizar el rendimiento de los solvers de Lattice Weaver.

**Filosofía:** Todas estas utilidades son **opt-in** - los solvers funcionan perfectamente sin ellas, pero pueden proporcionar mejoras de rendimiento en casos específicos.

---

## Utilidades Disponibles

### 1. `auto_profiler.py` - Profiling Automático

**Propósito:** Medir características del problema y tomar decisiones sobre qué optimizaciones activar.

**Características:**
- Recolecta métricas del problema (variables, restricciones, densidad)
- Mide tiempos de ejecución por componente
- Recomienda nivel de optimización basado en características
- Detecta cuellos de botella automáticamente

**Uso:**
```python
from lattice_weaver.utils.auto_profiler import AutoProfiler, OptimizationLevel

profiler = AutoProfiler()
metrics = profiler.profile_problem(variables, domains, hierarchy)
level = profiler.recommend_optimization_level(metrics)

if level >= OptimizationLevel.MEDIUM:
    # Activar optimizaciones pesadas
    pass
```

**Cuándo usar:**
- Problemas de tamaño desconocido
- Cuando se necesita adaptación automática
- Para benchmarking y análisis

---

### 2. `jit_compiler.py` - Compilación JIT

**Propósito:** Compilar funciones críticas usando Numba para acelerar código Python.

**Características:**
- Decoradores para compilación JIT
- Caché de funciones compiladas
- Fallback automático si Numba no está disponible
- Compilación lazy (solo cuando se usa)

**Uso:**
```python
from lattice_weaver.utils.jit_compiler import jit_compile, jit_if_available

@jit_if_available(nopython=True, cache=True)
def compute_energy(assignment, constraints):
    # Código crítico que se ejecuta muchas veces
    total = 0.0
    for constraint in constraints:
        total += evaluate_constraint(constraint, assignment)
    return total
```

**Cuándo usar:**
- Funciones que se ejecutan miles de veces
- Código con loops intensivos
- Operaciones numéricas pesadas

**Nota:** Requiere `numba` instalado. Si no está disponible, funciona sin JIT.

---

### 3. `lazy_init.py` - Inicialización Lazy

**Propósito:** Retrasar la creación de objetos costosos hasta que realmente se necesiten.

**Características:**
- Decorador `@lazy_property` para propiedades lazy
- Clase `LazyObject` para objetos completos lazy
- Thread-safe
- Caché automático después de primera inicialización

**Uso:**
```python
from lattice_weaver.utils.lazy_init import lazy_property, LazyObject

class Solver:
    @lazy_property
    def arc_engine(self):
        # Solo se crea si se usa
        return ArcEngine(self.variables, self.domains)
    
    @lazy_property
    def energy_landscape(self):
        # Solo se crea si se usa
        return EnergyLandscape(self.hierarchy)
```

**Cuándo usar:**
- Objetos costosos de crear
- Objetos que no siempre se usan
- Para reducir tiempo de inicialización

---

### 4. `metrics.py` - Sistema de Métricas

**Propósito:** Recolectar y reportar métricas de rendimiento de manera estructurada.

**Características:**
- Contadores, timers, histogramas
- Agregación automática
- Export a JSON/dict
- Métricas jerárquicas (por componente)

**Uso:**
```python
from lattice_weaver.utils.metrics import MetricsCollector

metrics = MetricsCollector()

with metrics.timer('search'):
    solution = search()

metrics.increment('backtracks')
metrics.record('domain_size', len(domain))

print(metrics.summary())
```

**Cuándo usar:**
- Para debugging de performance
- Benchmarking
- Análisis de comportamiento de solvers

---

### 5. `numpy_vectorization.py` - Vectorización con NumPy

**Propósito:** Usar operaciones vectorizadas de NumPy para acelerar cálculos sobre conjuntos de datos.

**Características:**
- Conversión automática Python ↔ NumPy
- Operaciones vectorizadas comunes
- Fallback a Python puro si NumPy no disponible
- Utilidades para dominios y restricciones

**Uso:**
```python
from lattice_weaver.utils.numpy_vectorization import (
    vectorize_domains,
    intersect_domains_vectorized,
    evaluate_constraints_vectorized
)

# Convertir dominios a arrays NumPy
np_domains = vectorize_domains(domains)

# Operaciones vectorizadas (mucho más rápidas)
intersections = intersect_domains_vectorized(np_domains, other_domains)
```

**Cuándo usar:**
- Problemas con muchas variables (>100)
- Operaciones sobre todos los dominios
- Cálculos de energía sobre muchas asignaciones

**Nota:** Requiere `numpy` instalado (ya está en requirements.txt).

---

### 6. `object_pool.py` - Object Pooling

**Propósito:** Reutilizar objetos en lugar de crear y destruir constantemente para reducir presión en GC.

**Características:**
- Pool genérico para cualquier tipo de objeto
- Límite de tamaño configurable
- Reset automático de objetos al devolver al pool
- Thread-safe

**Uso:**
```python
from lattice_weaver.utils.object_pool import ObjectPool

# Crear pool de asignaciones
assignment_pool = ObjectPool(
    factory=lambda: {},
    reset=lambda obj: obj.clear(),
    max_size=1000
)

# Obtener objeto del pool
assignment = assignment_pool.acquire()
assignment['x'] = 1

# Devolver al pool (se limpiará automáticamente)
assignment_pool.release(assignment)
```

**Cuándo usar:**
- Objetos que se crean/destruyen frecuentemente
- Problemas con muchos backtracks
- Para reducir tiempo en garbage collection

---

### 7. `persistence.py` - Persistencia de Estado

**Propósito:** Guardar y cargar estado de solvers para reanudar búsquedas o cachear resultados.

**Características:**
- Serialización de problemas CSP
- Guardado incremental (checkpoints)
- Compresión automática
- Versionado de formato

**Uso:**
```python
from lattice_weaver.utils.persistence import (
    save_problem,
    load_problem,
    save_checkpoint,
    load_checkpoint
)

# Guardar problema
save_problem('problem.pkl', variables, domains, hierarchy)

# Cargar problema
variables, domains, hierarchy = load_problem('problem.pkl')

# Checkpoint durante búsqueda
save_checkpoint('search_state.pkl', solver.get_state())

# Reanudar
state = load_checkpoint('search_state.pkl')
solver.restore_state(state)
```

**Cuándo usar:**
- Problemas que tardan mucho en resolver
- Para cachear problemas generados
- Debugging de estados específicos

---

### 8. `sparse_set.py` - Sparse Set

**Propósito:** Estructura de datos eficiente para conjuntos densos de enteros (como dominios).

**Características:**
- O(1) para add, remove, contains
- O(1) para iterar sobre elementos activos
- Muy eficiente en memoria para dominios grandes
- Operaciones de conjunto optimizadas

**Uso:**
```python
from lattice_weaver.utils.sparse_set import SparseSet

# Crear dominio como sparse set
domain = SparseSet(max_value=100)
domain.add_range(0, 100)  # Dominio [0..99]

# Operaciones O(1)
domain.remove(50)
if 42 in domain:
    print("42 está en el dominio")

# Iterar solo sobre valores activos (muy rápido)
for value in domain:
    print(value)
```

**Cuándo usar:**
- Dominios grandes (>50 valores)
- Muchas operaciones de add/remove
- Cuando se necesita iterar frecuentemente

---

### 9. `state_manager.py` - Gestión de Estado

**Propósito:** Gestionar estado de búsqueda con soporte para backtracking eficiente.

**Características:**
- Stack de estados con copy-on-write
- Backtracking O(1)
- Detección de cambios automática
- Soporte para undo/redo

**Uso:**
```python
from lattice_weaver.utils.state_manager import StateManager

state_mgr = StateManager()

# Guardar estado actual
state_mgr.push_state({'x': 1, 'y': 2})

# Modificar
state_mgr.set('x', 5)

# Backtrack (restaura estado anterior)
state_mgr.pop_state()
```

**Cuándo usar:**
- Búsqueda con backtracking
- Cuando se necesita deshacer cambios frecuentemente
- Para implementar búsqueda especulativa

---

## Guía de Uso

### Principio General: Opt-In

**Todas estas utilidades son opcionales.** Los solvers deben funcionar perfectamente sin ellas.

```python
# ❌ MAL: Dependencia obligatoria
class Solver:
    def __init__(self):
        self.profiler = AutoProfiler()  # Siempre creado
        
# ✅ BIEN: Opt-in
class Solver:
    def __init__(self, enable_profiling=False):
        self.profiler = AutoProfiler() if enable_profiling else None
```

### Cuándo Usar Cada Utilidad

| Utilidad | Tamaño Problema | Tipo de Problema | Beneficio Esperado |
|----------|-----------------|------------------|-------------------|
| **auto_profiler** | Cualquiera | Desconocido | Adaptación automática |
| **jit_compiler** | Grande (>100 vars) | Loops intensivos | 2-10x más rápido |
| **lazy_init** | Cualquiera | Objetos costosos | Menor tiempo init |
| **metrics** | Cualquiera | Debugging/benchmark | Visibilidad |
| **numpy_vectorization** | Grande (>100 vars) | Operaciones masivas | 5-50x más rápido |
| **object_pool** | Cualquiera | Muchos backtracks | Menos GC pressure |
| **persistence** | Muy grande (>1000 vars) | Búsquedas largas | Reanudar búsqueda |
| **sparse_set** | Dominios grandes (>50) | Muchas operaciones | 2-5x más rápido |
| **state_manager** | Cualquiera | Backtracking complejo | Código más limpio |

### Ejemplo de Integración Completa

```python
from lattice_weaver.fibration.fibration_search_solver import FibrationSearchSolver
from lattice_weaver.utils.auto_profiler import AutoProfiler, OptimizationLevel
from lattice_weaver.utils.lazy_init import lazy_property
from lattice_weaver.utils.metrics import MetricsCollector

class OptimizedSolver(FibrationSearchSolver):
    def __init__(self, variables, domains, hierarchy, enable_optimizations=True):
        super().__init__(variables, domains, hierarchy)
        
        if enable_optimizations:
            # Profiling automático
            self.profiler = AutoProfiler()
            metrics = self.profiler.profile_problem(variables, domains, hierarchy)
            self.opt_level = self.profiler.recommend_optimization_level(metrics)
            
            # Métricas
            self.metrics = MetricsCollector()
        else:
            self.profiler = None
            self.metrics = None
    
    @lazy_property
    def energy_landscape(self):
        # Solo se crea si se usa
        from lattice_weaver.fibration.energy_landscape import EnergyLandscape
        return EnergyLandscape(self.hierarchy)
    
    def solve(self):
        if self.metrics:
            with self.metrics.timer('total_solve'):
                solution = super().solve()
                print(self.metrics.summary())
                return solution
        else:
            return super().solve()
```

---

## Testing

Todas las utilidades tienen tests unitarios en `tests/unit/test_utils/`.

```bash
# Ejecutar tests de utilidades
python3.11 -m pytest tests/unit/test_utils/ -v
```

---

## Dependencias Opcionales

Algunas utilidades tienen dependencias opcionales:

- **jit_compiler:** `numba` (opcional, fallback a Python puro)
- **numpy_vectorization:** `numpy` (ya en requirements.txt)

Si una dependencia no está disponible, la utilidad funciona en modo degradado o lanza una advertencia clara.

---

## Roadmap

Futuras utilidades planeadas:

- **parallel_executor:** Ejecutar búsquedas en paralelo
- **adaptive_heuristics:** Heurísticas que se adaptan durante la búsqueda
- **memory_profiler:** Profiling de uso de memoria
- **distributed_search:** Búsqueda distribuida con Ray

---

## Referencias

- **Plan de reintegración:** `/home/ubuntu/plan_reintegracion_fibration_flow.md`
- **Lecciones aprendidas:** `/home/ubuntu/lecciones_aprendidas_fibration_merge.md`
- **Tracking:** `docs/FIBRATION_REINTEGRATION_TRACKING.md`

---

**Última actualización:** 16 de octubre de 2025 - Fase 2

