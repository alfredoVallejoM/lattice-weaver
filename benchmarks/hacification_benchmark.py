"""
Benchmark Simplificado de HacificationEngine

Mide las mejoras de rendimiento entre HacificationEngine original y optimizado.

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import time
import tracemalloc
import gc
from typing import Dict, Any

from lattice_weaver.fibration.hacification_engine import HacificationEngine
from lattice_weaver.fibration.hacification_engine_optimized import HacificationEngineOptimized
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
from lattice_weaver.arc_engine.core import ArcEngine


def create_test_hierarchy(n: int) -> ConstraintHierarchy:
    """Crea una jerarquía de restricciones de prueba (N-Queens simplificado)."""
    hierarchy = ConstraintHierarchy()
    
    # Añadir restricciones binarias
    for i in range(n):
        for j in range(i + 1, n):
            # No misma columna
            def ne_constraint(assignment, i=i, j=j):
                qi = f"Q{i}"
                qj = f"Q{j}"
                if qi in assignment and qj in assignment:
                    return assignment[qi] != assignment[qj]
                return True
            
            hierarchy.add_local_constraint(
                var1=f"Q{i}",
                var2=f"Q{j}",
                predicate=ne_constraint,
                hardness=Hardness.HARD,
                metadata={"name": f"Q{i}_ne_Q{j}"}
            )
            
            # No misma diagonal
            def no_diagonal(assignment, i=i, j=j):
                qi = f"Q{i}"
                qj = f"Q{j}"
                if qi in assignment and qj in assignment:
                    return abs(assignment[qi] - assignment[qj]) != abs(i - j)
                return True
            
            hierarchy.add_local_constraint(
                var1=f"Q{i}",
                var2=f"Q{j}",
                predicate=no_diagonal,
                hardness=Hardness.HARD,
                metadata={"name": f"Q{i}_nodiag_Q{j}"}
            )
    
    return hierarchy


def generate_test_assignments(n: int, count: int):
    """Genera asignaciones de prueba."""
    import random
    assignments = []
    
    for _ in range(count):
        assignment = {}
        for i in range(n):
            assignment[f"Q{i}"] = random.randint(0, n-1)
        assignments.append(assignment)
    
    return assignments


def benchmark_hacification_engine(engine, assignments, name):
    """Ejecuta benchmark de un HacificationEngine."""
    print(f"\n  Benchmarking {name}...")
    
    # Limpiar memoria
    gc.collect()
    
    # Iniciar medición de memoria
    tracemalloc.start()
    
    # Ejecutar llamadas a hacify
    start_time = time.time()
    for assignment in assignments:
        engine.hacify(assignment, strict=True)
    wall_time = time.time() - start_time
    
    # Obtener estadísticas de memoria
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Obtener estadísticas del engine
    stats = engine.get_statistics()
    
    print(f"    Tiempo total: {wall_time:.4f}s")
    print(f"    Tiempo promedio por llamada: {wall_time/len(assignments)*1000:.2f}ms")
    print(f"    Memoria pico: {peak/(1024*1024):.2f}MB")
    print(f"    Llamadas a hacify: {stats.get('performance', {}).get('hacify_calls', len(assignments))}")
    
    return {
        'wall_time': wall_time,
        'avg_time_per_call': wall_time / len(assignments),
        'peak_memory_mb': peak / (1024 * 1024),
        'stats': stats
    }


def main():
    """Función principal del benchmark."""
    print("=" * 80)
    print("HACIFICATION ENGINE PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # Configuración del benchmark
    test_sizes = [
        (4, 100),   # 4-Queens, 100 llamadas
        (4, 500),   # 4-Queens, 500 llamadas
        (4, 1000),  # 4-Queens, 1000 llamadas
        (6, 100),   # 6-Queens, 100 llamadas
    ]
    
    results = []
    
    for n, n_calls in test_sizes:
        print(f"\n{'='*80}")
        print(f"Test: {n}-Queens, {n_calls} llamadas a hacify()")
        print(f"{'='*80}")
        
        # Crear jerarquía y landscape
        hierarchy = create_test_hierarchy(n)
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Generar asignaciones de prueba
        assignments = generate_test_assignments(n, n_calls)
        
        # Benchmark versión ORIGINAL
        print("\n[1] Versión ORIGINAL (HacificationEngine)")
        arc_engine_orig = ArcEngine(use_tms=False, parallel=False)
        engine_orig = HacificationEngine(hierarchy, landscape, arc_engine_orig)
        result_orig = benchmark_hacification_engine(engine_orig, assignments, "Original")
        
        # Benchmark versión OPTIMIZADA
        print("\n[2] Versión OPTIMIZADA (HacificationEngineOptimized)")
        arc_engine_opt = ArcEngine(use_tms=False, parallel=False)
        engine_opt = HacificationEngineOptimized(hierarchy, landscape, arc_engine_opt)
        result_opt = benchmark_hacification_engine(engine_opt, assignments, "Optimized")
        
        # Calcular mejoras
        time_speedup = result_orig['wall_time'] / result_opt['wall_time']
        memory_reduction = result_orig['peak_memory_mb'] / result_opt['peak_memory_mb']
        
        print(f"\n{'='*80}")
        print(f"MEJORAS OBTENIDAS:")
        print(f"  Speedup de tiempo:    {time_speedup:.2f}x")
        print(f"  Reducción de memoria: {memory_reduction:.2f}x")
        print(f"  Tiempo por llamada:   {result_orig['avg_time_per_call']*1000:.2f}ms → {result_opt['avg_time_per_call']*1000:.2f}ms")
        print(f"{'='*80}")
        
        results.append({
            'n': n,
            'n_calls': n_calls,
            'original': result_orig,
            'optimized': result_opt,
            'time_speedup': time_speedup,
            'memory_reduction': memory_reduction
        })
    
    # Resumen final
    print(f"\n\n{'='*80}")
    print("RESUMEN FINAL")
    print(f"{'='*80}")
    
    avg_time_speedup = sum(r['time_speedup'] for r in results) / len(results)
    avg_memory_reduction = sum(r['memory_reduction'] for r in results) / len(results)
    
    print(f"\nPromedios generales:")
    print(f"  Speedup de tiempo promedio:      {avg_time_speedup:.2f}x")
    print(f"  Reducción de memoria promedio:   {avg_memory_reduction:.2f}x")
    
    print(f"\nDetalle por test:")
    print(f"{'Test':<20} {'Speedup':<15} {'Reducción Mem':<20}")
    print(f"{'-'*55}")
    for r in results:
        test_name = f"{r['n']}-Queens, {r['n_calls']} calls"
        print(f"{test_name:<20} {r['time_speedup']:>6.2f}x{'':<8} {r['memory_reduction']:>6.2f}x")
    
    print(f"\n{'='*80}")
    print("✓ Benchmark completado exitosamente")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

