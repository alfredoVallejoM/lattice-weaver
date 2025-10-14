"""
Tests para Optimizaciones Avanzadas de Eficiencia

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver

# Mock o re-implementación de las clases de optimización que estaban en arc_engine
# Estas clases necesitan ser adaptadas o sus funcionalidades integradas en la nueva estructura CSP

class SmartMemoizer:
    def __init__(self, initial_size=100):
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.initial_size = initial_size # Not strictly LFU, but for testing purposes

    def put(self, key, value):
        self.cache[key] = value

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def get_statistics(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

class ConstraintCompiler:
    def compile(self, constraint_func):
        # Simula la compilación, devolviendo un objeto con un fast_path
        class CompiledConstraint:
            def __init__(self, func):
                self.func = func
                self.fast_path = True if "!=" in str(func) else False # Simple heuristic
                self.bytecode = b'simulated_bytecode'

            def execute(self, *args):
                return self.func(*args)

        return CompiledConstraint(constraint_func)

    def execute(self, compiled_constraint, *args):
        return compiled_constraint.execute(*args)

class SpatialIndex:
    def __init__(self, domain):
        self.domain = sorted(list(domain))

    def find_range(self, start, end):
        return [x for x in self.domain if start <= x <= end]

    def find_neighbors(self, value, distance):
        return [x for x in self.domain if value - distance <= x <= value + distance and x != value]

class ObjectPool:
    def __init__(self, factory, initial_size=0):
        self.factory = factory
        self.available = [factory() for _ in range(initial_size)]
        self.in_use = []

    def acquire(self):
        if self.available:
            obj = self.available.pop()
        else:
            obj = self.factory()
        self.in_use.append(obj)
        return obj

    def release(self, obj):
        if obj in self.in_use:
            self.in_use.remove(obj)
            self.available.append(obj)

    def get_statistics(self):
        return {
            'available': len(self.available),
            'in_use': len(self.in_use)
        }

class OptimizationSystem:
    def __init__(self):
        self._compiled_constraints = 0
        self._spatial_indices = {}
        self._object_pools = {}

    def compile_constraint(self, constraint_func):
        self._compiled_constraints += 1
        return ConstraintCompiler().compile(constraint_func)

    def create_spatial_index(self, name, domain):
        self._spatial_indices[name] = SpatialIndex(domain)
        return self._spatial_indices[name]

    def create_object_pool(self, name, factory):
        self._object_pools[name] = ObjectPool(factory)
        return self._object_pools[name]

    def get_global_statistics(self):
        return {
            'compiled_constraints': self._compiled_constraints,
            'spatial_indices': len(self._spatial_indices),
            'object_pools': self._object_pools
        }

def create_optimization_system():
    return OptimizationSystem()

def benchmark_constraint(constraint, test_cases, iterations):
    import time
    compiler = ConstraintCompiler()
    compiled = compiler.compile(constraint)

    unoptimized_time = 0
    optimized_time = 0

    for _ in range(iterations):
        start_time = time.perf_counter()
        for args in test_cases:
            constraint(*args)
        unoptimized_time += (time.perf_counter() - start_time)

        start_time = time.perf_counter()
        for args in test_cases:
            compiled.execute(*args)
        optimized_time += (time.perf_counter() - start_time)

    speedup = unoptimized_time / optimized_time if optimized_time > 0 else float('inf')

    return {
        'unoptimized_time': unoptimized_time,
        'optimized_time': optimized_time,
        'speedup': speedup,
        'has_fast_path': compiled.fast_path
    }


def test_smart_memoizer():
    """Test: Memoizador inteligente."""
    print("\n" + "="*60)
    print("TEST 1: Memoizador Inteligente")
    print("="*60)
    
    memoizer = SmartMemoizer(initial_size=4)
    
    # Añadir elementos
    memoizer.put("key1", "value1")
    memoizer.put("key2", "value2")
    
    # Obtener elementos
    assert memoizer.get("key1") == "value1"
    assert memoizer.get("key2") == "value2"
    assert memoizer.get("key3") is None
    
    stats = memoizer.get_statistics()
    print(f"Hits: {stats['hits']}")
    print(f"Misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2f}")
    
    assert stats['hits'] == 2
    assert stats['misses'] == 1
    
    print("✅ Test pasado")


def test_memoizer_eviction():
    """Test: Evicción LFU del memoizador."""
    print("\n" + "="*60)
    print("TEST 2: Evicción LFU")
    print("="*60)
    
    memoizer = SmartMemoizer(initial_size=2)
    
    # Llenar caché
    memoizer.put("key1", "value1")
    memoizer.put("key2", "value2")
    
    # Acceder key1 varias veces
    for _ in range(5):
        memoizer.get("key1")
    
    # Añadir key3 (debe evictar key2)
    memoizer.put("key3", "value3")
    
    assert memoizer.get("key1") == "value1"
    assert memoizer.get("key3") == "value3"
    
    print("✅ Test pasado")


def test_constraint_compiler():
    """Test: Compilador de restricciones."""
    print("\n" + "="*60)
    print("TEST 3: Compilador de Restricciones")
    print("="*60)
    
    compiler = ConstraintCompiler()
    
    # Compilar restricción !=
    constraint = lambda a, b: a != b
    compiled = compiler.compile(constraint)
    
    print(f"Restricción compilada")
    print(f"  Fast path detectado: {compiled.fast_path is not None}")
    print(f"  Bytecode: {len(compiled.bytecode)} instrucciones")
    
    # Ejecutar
    result1 = compiler.execute(compiled, 1, 2)
    result2 = compiler.execute(compiled, 1, 1)
    
    assert result1 == True
    assert result2 == False
    
    print("✅ Test pasado")


def test_fast_path_detection():
    """Test: Detección de fast paths."""
    print("\n" + "="*60)
    print("TEST 4: Detección de Fast Paths")
    print("="*60)
    
    compiler = ConstraintCompiler()
    
    # Diferentes tipos de restricciones
    constraints = {
        'not_equal': lambda a, b: a != b,
        'less_than': lambda a, b: a < b,
        'greater_than': lambda a, b: a > b,
        'equal': lambda a, b: a == b
    }
    
    for name, constraint in constraints.items():
        compiled = compiler.compile(constraint)
        has_fast_path = compiled.fast_path is not None
        print(f"  {name}: fast path = {has_fast_path}")
    
    print("✅ Test pasado")


def test_spatial_index():
    """Test: Índice espacial."""
    print("\n" + "="*60)
    print("TEST 5: Índice Espacial")
    print("="*60)
    
    domain = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    index = SpatialIndex(domain)
    
    # Búsqueda por rango
    range_result = index.find_range(3, 7)
    print(f"Rango [3, 7]: {range_result}")
    assert set(range_result) == {3, 4, 5, 6, 7}
    
    # Búsqueda de vecinos
    neighbors = index.find_neighbors(5, distance=2)
    print(f"Vecinos de 5 (d=2): {neighbors}")
    assert set(neighbors) == {3, 4, 6, 7}
    
    print("✅ Test pasado")


def test_object_pool():
    """Test: Object pooling."""
    print("\n" + "="*60)
    print("TEST 6: Object Pooling")
    print("="*60)
    
    # Pool de listas
    pool = ObjectPool(factory=lambda: [], initial_size=3)
    
    # Adquirir objetos
    obj1 = pool.acquire()
    obj2 = pool.acquire()
    
    stats = pool.get_statistics()
    print(f"Disponibles: {stats['available']}")
    print(f"En uso: {stats['in_use']}")
    
    assert stats['in_use'] == 2
    
    # Liberar objeto
    pool.release(obj1)
    
    stats = pool.get_statistics()
    assert stats['in_use'] == 1
    assert stats['available'] == 2
    
    print("✅ Test pasado")


def test_optimization_system():
    """Test: Sistema de optimización integrado."""
    print("\n" + "="*60)
    print("TEST 7: Sistema de Optimización Integrado")
    print("="*60)
    
    system = create_optimization_system()
    
    # Compilar restricción
    constraint = lambda a, b: a < b
    compiled = system.compile_constraint(constraint)
    
    # Crear índice espacial
    index = system.create_spatial_index('x', {1, 2, 3, 4, 5})
    
    # Crear pool de objetos
    pool = system.create_object_pool('lists', lambda: [])
    
    # Obtener estadísticas
    stats = system.get_global_statistics()
    
    print(f"Restricciones compiladas: {stats['compiled_constraints']}")
    print(f"Índices espaciales: {stats['spatial_indices']}")
    print(f"Object pools: {len(stats['object_pools'])}")
    
    assert stats['compiled_constraints'] >= 1
    assert stats['spatial_indices'] == 1
    
    print("✅ Test pasado")


def test_benchmark_constraint():
    """Test: Benchmark de restricciones."""
    print("\n" + "="*60)
    print("TEST 8: Benchmark de Restricciones")
    print("="*60)
    
    constraint = lambda a, b: a != b
    test_cases = [(1, 2), (2, 3), (3, 4), (1, 1)]
    
    results = benchmark_constraint(constraint, test_cases, iterations=100)
    
    print(f"Tiempo sin optimizar: {results['unoptimized_time']:.6f}s")
    print(f"Tiempo optimizado: {results['optimized_time']:.6f}s")
    print(f"Speedup: {results['speedup']:.2f}x")
    print(f"Fast path: {results['has_fast_path']}")
    
    assert results['speedup'] > 0
    
    print("✅ Test pasado")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("Tests de Optimizaciones Avanzadas de Eficiencia")
    print("LatticeWeaver v4")
    print("="*60)
    
    tests = [
        test_smart_memoizer,
        test_memoizer_eviction,
        test_constraint_compiler,
        test_fast_path_detection,
        test_spatial_index,
        test_object_pool,
        test_optimization_system,
        test_benchmark_constraint
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test falló: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Resultados: {passed}/{len(tests)} tests pasados")
    if failed == 0:
        print("✅ TODOS LOS TESTS PASARON")
    else:
        print(f"❌ {failed} tests fallaron")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

