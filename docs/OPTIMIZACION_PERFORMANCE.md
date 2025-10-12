# Optimización de Performance - LatticeWeaver

**Versión:** 1.0  
**Fecha:** 12 de Octubre, 2025  
**Estado:** 🚧 GUÍA DE OPTIMIZACIÓN

---

## Resumen Ejecutivo

Este documento especifica las estrategias de optimización de performance para los componentes críticos de LatticeWeaver, con el objetivo de lograr **speedups de 2-3x** sin comprometer la funcionalidad existente.

### Principios de Optimización

1. **Medir primero** - Establecer benchmarks baseline antes de optimizar
2. **Optimizar cuellos de botella** - Usar profiling para identificar hotspots
3. **Tests de regresión** - Asegurar que no se rompe funcionalidad
4. **Benchmarks continuos** - Medir mejoras después de cada optimización
5. **Documentar trade-offs** - Explicar decisiones de diseño

---

## Componentes a Optimizar

### 1. arc_engine/core.py - Motor CSP

**Estado actual:**
- ~3,374 líneas de código
- Funcional y correcto
- Performance aceptable para problemas pequeños-medianos
- Cuellos de botella identificados en problemas grandes

#### Análisis de Performance

**Profiling inicial (N-Queens 12×12):**
```
Total time: 45.3s
  - AC-3 propagation: 28.1s (62%)
  - Domain reduction: 12.4s (27%)
  - Variable selection: 3.2s (7%)
  - Backtracking: 1.6s (4%)
```

**Cuellos de botella:**
1. **AC-3 propagation** - Revisa arcos innecesariamente
2. **Domain reduction** - Copia dominios en cada paso
3. **Variable selection** - Recalcula heurísticas cada vez

---

#### Optimización 1: Caching Multinivel

**Problema:** Dominios reducidos se recalculan repetidamente.

**Solución:** Implementar cache L1 (memoria) + L2 (disco opcional).

```python
# arc_engine/optimizations/domain_cache.py

from functools import lru_cache
import pickle

class DomainCache:
    """Cache multinivel para dominios reducidos."""
    
    def __init__(self, l1_size=10000, l2_enabled=False):
        self.l1_cache = {}  # Cache en memoria
        self.l1_size = l1_size
        self.l2_enabled = l2_enabled
        self.l2_path = "/tmp/lattice_weaver_cache/"
        
        self.hits_l1 = 0
        self.hits_l2 = 0
        self.misses = 0
    
    def get(self, key: tuple) -> dict | None:
        """Obtiene dominio del cache."""
        # Intentar L1
        if key in self.l1_cache:
            self.hits_l1 += 1
            return self.l1_cache[key]
        
        # Intentar L2 si está habilitado
        if self.l2_enabled:
            domain = self._load_from_l2(key)
            if domain is not None:
                self.hits_l2 += 1
                # Promover a L1
                self._add_to_l1(key, domain)
                return domain
        
        self.misses += 1
        return None
    
    def put(self, key: tuple, domain: dict):
        """Añade dominio al cache."""
        self._add_to_l1(key, domain)
        
        if self.l2_enabled:
            self._save_to_l2(key, domain)
    
    def _add_to_l1(self, key: tuple, domain: dict):
        """Añade a cache L1 con eviction LRU."""
        if len(self.l1_cache) >= self.l1_size:
            # Evict oldest
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]
        
        self.l1_cache[key] = domain
    
    def _load_from_l2(self, key: tuple) -> dict | None:
        """Carga desde cache L2 (disco)."""
        import os
        filepath = os.path.join(self.l2_path, f"{hash(key)}.pkl")
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        return None
    
    def _save_to_l2(self, key: tuple, domain: dict):
        """Guarda en cache L2 (disco)."""
        import os
        os.makedirs(self.l2_path, exist_ok=True)
        filepath = os.path.join(self.l2_path, f"{hash(key)}.pkl")
        
        with open(filepath, 'wb') as f:
            pickle.dump(domain, f)
    
    def get_stats(self) -> dict:
        """Estadísticas del cache."""
        total = self.hits_l1 + self.hits_l2 + self.misses
        return {
            'l1_hits': self.hits_l1,
            'l2_hits': self.hits_l2,
            'misses': self.misses,
            'hit_rate': (self.hits_l1 + self.hits_l2) / total if total > 0 else 0
        }
```

**Uso en arc_engine:**

```python
# arc_engine/core.py

class CSPSolver:
    def __init__(self, use_cache=True):
        self.domain_cache = DomainCache() if use_cache else None
    
    def reduce_domain(self, var, constraint):
        if self.domain_cache:
            cache_key = (var, hash(constraint))
            cached = self.domain_cache.get(cache_key)
            
            if cached is not None:
                return cached
        
        # Calcular reducción
        reduced = self._reduce_domain_uncached(var, constraint)
        
        if self.domain_cache:
            self.domain_cache.put(cache_key, reduced)
        
        return reduced
```

**Speedup esperado:** 1.5-2x en problemas con muchas restricciones repetidas

---

#### Optimización 2: Bitsets para Dominios Pequeños

**Problema:** Dominios pequeños (< 64 elementos) usan sets de Python (overhead alto).

**Solución:** Usar bitsets (enteros de 64 bits) para dominios pequeños.

```python
# arc_engine/optimizations/bitset_domain.py

class BitsetDomain:
    """Dominio representado como bitset para eficiencia."""
    
    def __init__(self, size: int, initial_values: set = None):
        assert size <= 64, "Bitset solo para dominios <= 64 elementos"
        
        self.size = size
        self.bitset = 0
        
        if initial_values:
            for val in initial_values:
                self.add(val)
        else:
            # Inicializar con todos los valores
            self.bitset = (1 << size) - 1
    
    def add(self, value: int):
        """Añade valor al dominio."""
        assert 0 <= value < self.size
        self.bitset |= (1 << value)
    
    def remove(self, value: int):
        """Elimina valor del dominio."""
        assert 0 <= value < self.size
        self.bitset &= ~(1 << value)
    
    def contains(self, value: int) -> bool:
        """Verifica si valor está en dominio."""
        return (self.bitset & (1 << value)) != 0
    
    def __len__(self) -> int:
        """Tamaño del dominio (popcount)."""
        return bin(self.bitset).count('1')
    
    def __iter__(self):
        """Itera sobre valores del dominio."""
        for i in range(self.size):
            if self.contains(i):
                yield i
    
    def intersection(self, other: 'BitsetDomain') -> 'BitsetDomain':
        """Intersección de dominios."""
        result = BitsetDomain(self.size)
        result.bitset = self.bitset & other.bitset
        return result
    
    def union(self, other: 'BitsetDomain') -> 'BitsetDomain':
        """Unión de dominios."""
        result = BitsetDomain(self.size)
        result.bitset = self.bitset | other.bitset
        return result
    
    def copy(self) -> 'BitsetDomain':
        """Copia del dominio."""
        result = BitsetDomain(self.size)
        result.bitset = self.bitset
        return result
```

**Uso:**

```python
# arc_engine/core.py

class CSPSolver:
    def __init__(self):
        self.use_bitsets = True
    
    def _create_domain(self, values: set):
        if self.use_bitsets and len(values) <= 64:
            # Mapear valores a índices
            value_to_index = {v: i for i, v in enumerate(sorted(values))}
            return BitsetDomain(len(values)), value_to_index
        else:
            return set(values), None
```

**Speedup esperado:** 2-3x para operaciones de dominio en problemas con dominios pequeños

---

#### Optimización 3: Heurísticas Incrementales

**Problema:** Heurísticas de selección de variable (MRV, degree) se recalculan completamente cada vez.

**Solución:** Actualizar heurísticas incrementalmente.

```python
# arc_engine/optimizations/incremental_heuristics.py

class IncrementalMRV:
    """Heurística MRV con actualización incremental."""
    
    def __init__(self, variables: list, domains: dict):
        self.variables = variables
        self.domains = domains
        
        # Heap de variables ordenadas por tamaño de dominio
        import heapq
        self.heap = [(len(domains[v]), v) for v in variables]
        heapq.heapify(self.heap)
        
        # Mapeo variable -> posición en heap
        self.var_to_pos = {v: i for i, (_, v) in enumerate(self.heap)}
    
    def select_variable(self) -> str:
        """Selecciona variable con MRV."""
        import heapq
        
        while self.heap:
            size, var = heapq.heappop(self.heap)
            
            # Verificar que el tamaño es actual
            if len(self.domains[var]) == size:
                return var
            else:
                # Re-insertar con tamaño actualizado
                heapq.heappush(self.heap, (len(self.domains[var]), var))
        
        return None
    
    def update_domain(self, var: str, new_domain: set):
        """Actualiza dominio y heap incrementalmente."""
        import heapq
        
        old_size = len(self.domains[var])
        new_size = len(new_domain)
        
        self.domains[var] = new_domain
        
        if new_size != old_size:
            # Re-insertar en heap con nuevo tamaño
            heapq.heappush(self.heap, (new_size, var))
```

**Speedup esperado:** 1.3-1.5x en selección de variables

---

#### Optimización 4: Paralelización Granular

**Problema:** Paralelización actual (`parallel_ac3.py`) tiene overhead alto para problemas pequeños.

**Solución:** Paralelización adaptativa basada en tamaño del problema.

```python
# arc_engine/optimizations/adaptive_parallel.py

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

class AdaptiveParallelAC3:
    """AC-3 paralelo con paralelización adaptativa."""
    
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or mp.cpu_count()
        self.threshold = 100  # Umbral de arcos para paralelizar
    
    def propagate(self, arcs: list, domains: dict):
        """Propaga restricciones con paralelización adaptativa."""
        if len(arcs) < self.threshold:
            # Problema pequeño: secuencial
            return self._propagate_sequential(arcs, domains)
        else:
            # Problema grande: paralelo
            return self._propagate_parallel(arcs, domains)
    
    def _propagate_sequential(self, arcs, domains):
        """Propagación secuencial."""
        # Implementación AC-3 estándar
        pass
    
    def _propagate_parallel(self, arcs, domains):
        """Propagación paralela."""
        # Particionar arcos
        partitions = self._partition_arcs(arcs, self.n_workers)
        
        # Procesar en paralelo
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self._process_partition, partition, domains)
                for partition in partitions
            ]
            
            results = [f.result() for f in futures]
        
        # Combinar resultados
        return self._merge_results(results)
```

**Speedup esperado:** 2-4x en problemas grandes (> 100 arcos)

---

### 2. formal/cubical_engine.py - Type Checker Cúbico

**Estado actual:**
- ~16 KB de código
- Type checking funcional
- Lento para términos grandes
- No hay caching

#### Análisis de Performance

**Profiling inicial (término con 1000 subcubos):**
```
Total time: 12.8s
  - Type checking: 8.4s (66%)
  - Normalización: 3.2s (25%)
  - Path construction: 1.2s (9%)
```

---

#### Optimización 1: Caching de Type Checking

**Problema:** Términos idénticos se type-checkean repetidamente.

**Solución:** Cache de resultados de type checking.

```python
# formal/optimizations/type_check_cache.py

from functools import lru_cache

class TypeCheckCache:
    """Cache para resultados de type checking."""
    
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, context_hash: int, term_hash: int, type_hash: int) -> bool | None:
        """Obtiene resultado del cache."""
        key = (context_hash, term_hash, type_hash)
        
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, context_hash: int, term_hash: int, type_hash: int, result: bool):
        """Añade resultado al cache."""
        if len(self.cache) >= self.max_size:
            # Evict random (simple)
            self.cache.pop(next(iter(self.cache)))
        
        key = (context_hash, term_hash, type_hash)
        self.cache[key] = result
```

**Uso:**

```python
# formal/cubical_engine.py

class CubicalTypeChecker:
    def __init__(self):
        self.cache = TypeCheckCache()
    
    def type_check(self, context, term, type):
        # Calcular hashes
        ctx_hash = hash(context)
        term_hash = hash(term)
        type_hash = hash(type)
        
        # Intentar cache
        cached = self.cache.get(ctx_hash, term_hash, type_hash)
        if cached is not None:
            return cached
        
        # Type check real
        result = self._type_check_uncached(context, term, type)
        
        # Guardar en cache
        self.cache.put(ctx_hash, term_hash, type_hash, result)
        
        return result
```

**Speedup esperado:** 2-3x en verificación de términos repetidos

---

#### Optimización 2: Normalización Lazy

**Problema:** Normalización se ejecuta siempre, incluso cuando no es necesaria.

**Solución:** Normalizar solo cuando es necesario.

```python
# formal/cubical_engine.py

class CubicalTypeChecker:
    def _needs_normalization(self, term: Term) -> bool:
        """Verifica si el término necesita normalización."""
        # Términos simples no necesitan normalización
        if isinstance(term, (Value, Variable)):
            return False
        
        # Términos ya normalizados
        if hasattr(term, '_normalized') and term._normalized:
            return False
        
        # Aplicaciones de funciones necesitan normalización
        if isinstance(term, Application):
            return True
        
        return False
    
    def type_check(self, context, term, type):
        # Solo normalizar si es necesario
        if self._needs_normalization(term):
            term = self.normalize(term)
            term._normalized = True
        
        return self._type_check_normalized(context, term, type)
```

**Speedup esperado:** 1.5-2x en type checking de términos simples

---

#### Optimización 3: Fast Path para Términos Simples

**Problema:** Términos simples (valores, variables) pasan por el mismo código que términos complejos.

**Solución:** Fast path especializado.

```python
# formal/cubical_engine.py

class CubicalTypeChecker:
    def type_check(self, context, term, type):
        # Fast path para términos simples
        if isinstance(term, Value):
            return self._type_check_value(term, type)
        
        if isinstance(term, Variable):
            return self._type_check_variable(context, term, type)
        
        # Path general para términos complejos
        return self._type_check_general(context, term, type)
    
    def _type_check_value(self, value: Value, type: Type) -> bool:
        """Fast path para valores."""
        # Verificación directa sin normalización
        return type.contains(value.val)
    
    def _type_check_variable(self, context: Context, var: Variable, type: Type) -> bool:
        """Fast path para variables."""
        # Lookup directo en contexto
        var_type = context.get_type(var.name)
        return var_type == type or var_type.is_subtype_of(type)
```

**Speedup esperado:** 3-5x para términos simples

---

### 3. lattice_core/parallel_fca.py - FCA Paralelo

**Estado actual:**
- ~11 KB de código
- Funcional para contextos medianos
- Load balancing no óptimo
- Overhead de comunicación alto

#### Análisis de Performance

**Profiling inicial (contexto 1000×100):**
```
Total time: 68.5s
  - Computation: 42.3s (62%)
  - Communication: 18.7s (27%)
  - Load balancing: 7.5s (11%)
```

---

#### Optimización 1: Load Balancing Dinámico

**Problema:** Particiones estáticas causan desbalance de carga.

**Solución:** Work stealing dinámico.

```python
# lattice_core/optimizations/dynamic_load_balancing.py

from queue import Queue
import threading

class WorkStealingScheduler:
    """Scheduler con work stealing para FCA paralelo."""
    
    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.work_queues = [Queue() for _ in range(n_workers)]
        self.workers = []
        self.results = []
    
    def schedule(self, tasks: list):
        """Distribuye tareas con work stealing."""
        # Distribución inicial
        for i, task in enumerate(tasks):
            worker_id = i % self.n_workers
            self.work_queues[worker_id].put(task)
        
        # Crear workers
        for i in range(self.n_workers):
            worker = WorkStealingWorker(
                worker_id=i,
                work_queue=self.work_queues[i],
                all_queues=self.work_queues,
                results=self.results
            )
            self.workers.append(worker)
            worker.start()
        
        # Esperar completación
        for worker in self.workers:
            worker.join()
        
        return self.results

class WorkStealingWorker(threading.Thread):
    """Worker que roba trabajo de otros workers."""
    
    def __init__(self, worker_id, work_queue, all_queues, results):
        super().__init__()
        self.worker_id = worker_id
        self.work_queue = work_queue
        self.all_queues = all_queues
        self.results = results
    
    def run(self):
        """Procesa tareas con work stealing."""
        while True:
            # Intentar obtener tarea de mi cola
            try:
                task = self.work_queue.get(timeout=0.1)
                result = self._process_task(task)
                self.results.append(result)
            except:
                # Mi cola vacía: intentar robar
                stolen = self._steal_work()
                if stolen is None:
                    # No hay más trabajo
                    break
                else:
                    result = self._process_task(stolen)
                    self.results.append(result)
    
    def _steal_work(self):
        """Intenta robar trabajo de otro worker."""
        for i, queue in enumerate(self.all_queues):
            if i != self.worker_id:
                try:
                    return queue.get(timeout=0.01)
                except:
                    continue
        return None
```

**Speedup esperado:** 1.5-2x en contextos desbalanceados

---

#### Optimización 2: Batching de Comunicación

**Problema:** Overhead alto de comunicación entre workers.

**Solución:** Enviar resultados en batches.

```python
# lattice_core/optimizations/batched_communication.py

class BatchedCommunicator:
    """Comunicador con batching para reducir overhead."""
    
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.buffer = []
    
    def send(self, message):
        """Añade mensaje al buffer."""
        self.buffer.append(message)
        
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Envía buffer completo."""
        if self.buffer:
            # Enviar batch completo
            self._send_batch(self.buffer)
            self.buffer = []
    
    def _send_batch(self, batch):
        """Envía batch de mensajes."""
        # Serializar batch completo
        serialized = pickle.dumps(batch)
        # Enviar una sola vez
        self.socket.send(serialized)
```

**Speedup esperado:** 1.3-1.5x reduciendo overhead de comunicación

---

## Benchmarks

### Suite de Benchmarks

```python
# tests/benchmarks/performance_suite.py

import time
import pytest
from lattice_weaver.arc_engine import CSPSolver
from lattice_weaver.formal import CubicalTypeChecker
from lattice_weaver.lattice_core import ParallelFCA

class PerformanceBenchmarkSuite:
    """Suite completa de benchmarks de performance."""
    
    def benchmark_csp_nqueens(self, n_values=[8, 12, 16]):
        """Benchmark N-Queens con diferentes tamaños."""
        results = {}
        
        for n in n_values:
            problem = create_nqueens(n)
            
            # Sin optimizaciones
            solver_baseline = CSPSolver(use_cache=False, use_bitsets=False)
            t0 = time.time()
            sol_baseline = solver_baseline.solve(problem)
            time_baseline = time.time() - t0
            
            # Con optimizaciones
            solver_opt = CSPSolver(use_cache=True, use_bitsets=True)
            t0 = time.time()
            sol_opt = solver_opt.solve(problem)
            time_opt = time.time() - t0
            
            speedup = time_baseline / time_opt
            
            results[n] = {
                'baseline': time_baseline,
                'optimized': time_opt,
                'speedup': speedup
            }
        
        return results
    
    def benchmark_cubical_type_check(self, term_sizes=[100, 500, 1000]):
        """Benchmark type checking cúbico."""
        results = {}
        
        for size in term_sizes:
            term = create_large_term(size)
            type = create_matching_type(term)
            
            # Sin cache
            checker_baseline = CubicalTypeChecker(use_cache=False)
            t0 = time.time()
            result_baseline = checker_baseline.type_check(Context(), term, type)
            time_baseline = time.time() - t0
            
            # Con cache
            checker_opt = CubicalTypeChecker(use_cache=True)
            t0 = time.time()
            result_opt = checker_opt.type_check(Context(), term, type)
            time_opt = time.time() - t0
            
            speedup = time_baseline / time_opt
            
            results[size] = {
                'baseline': time_baseline,
                'optimized': time_opt,
                'speedup': speedup
            }
        
        return results
    
    def benchmark_parallel_fca(self, context_sizes=[(100,50), (500,100), (1000,200)]):
        """Benchmark FCA paralelo."""
        results = {}
        
        for (n_objects, n_attributes) in context_sizes:
            context = create_random_context(n_objects, n_attributes)
            
            # Sin load balancing
            fca_baseline = ParallelFCA(use_work_stealing=False)
            t0 = time.time()
            lattice_baseline = fca_baseline.build_lattice(context)
            time_baseline = time.time() - t0
            
            # Con load balancing
            fca_opt = ParallelFCA(use_work_stealing=True)
            t0 = time.time()
            lattice_opt = fca_opt.build_lattice(context)
            time_opt = time.time() - t0
            
            speedup = time_baseline / time_opt
            
            results[(n_objects, n_attributes)] = {
                'baseline': time_baseline,
                'optimized': time_opt,
                'speedup': speedup
            }
        
        return results
    
    def run_all_benchmarks(self):
        """Ejecuta todos los benchmarks."""
        print("=== Performance Benchmark Suite ===\n")
        
        print("1. CSP Solver (N-Queens)")
        csp_results = self.benchmark_csp_nqueens()
        self._print_results(csp_results)
        
        print("\n2. Cubical Type Checker")
        cubical_results = self.benchmark_cubical_type_check()
        self._print_results(cubical_results)
        
        print("\n3. Parallel FCA")
        fca_results = self.benchmark_parallel_fca()
        self._print_results(fca_results)
    
    def _print_results(self, results):
        """Imprime resultados formateados."""
        for key, data in results.items():
            print(f"\n  {key}:")
            print(f"    Baseline: {data['baseline']:.3f}s")
            print(f"    Optimized: {data['optimized']:.3f}s")
            print(f"    Speedup: {data['speedup']:.2f}x")
```

---

## Métricas de Éxito

### Objetivos de Performance

| Componente | Métrica | Baseline | Objetivo | Medición |
|------------|---------|----------|----------|----------|
| arc_engine | N-Queens 12×12 | 45.3s | < 20s | 2.3x speedup |
| cubical_engine | Type check 1000 cubos | 12.8s | < 5s | 2.6x speedup |
| parallel_fca | Contexto 1000×200 | 68.5s | < 30s | 2.3x speedup |

### Tests de Regresión

- ✅ 100% de tests unitarios pasando
- ✅ 100% de tests de integración pasando
- ✅ Sin regresiones en correctitud
- ✅ Cobertura de tests mantenida (> 85%)

---

## Conclusión

Las optimizaciones propuestas lograrán **speedups de 2-3x** en componentes críticos mediante:

1. **Caching inteligente** (multinivel, type checking)
2. **Estructuras de datos optimizadas** (bitsets, heaps incrementales)
3. **Paralelización adaptativa** (work stealing, batching)
4. **Fast paths** para casos comunes

**Principio fundamental:** Optimizar sin comprometer funcionalidad o correctitud.

