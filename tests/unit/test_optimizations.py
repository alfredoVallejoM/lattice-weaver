#!/usr/bin/env python3
# test_optimizations.py

"""
Tests para Optimizaciones de Rendimiento

Valida caché, ordenamiento, filtrado y monitoreo.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_weaver.arc_engine import (
    ArcEngine, NE, LT,
    ArcRevisionCache, ArcOrderingStrategy, RedundantArcDetector,
    PerformanceMonitor, OptimizedAC3, create_optimized_ac3
)


def test_arc_revision_cache():
    """Test: Caché de revisiones de arcos."""
    print("=" * 60)
    print("TEST 1: Caché de Revisiones de Arcos")
    print("=" * 60)
    
    cache = ArcRevisionCache(max_size=100)
    
    # Simular revisiones
    cache.put("X", "Y", "C1", hash(frozenset([1, 2, 3])), hash(frozenset([2, 3, 4])),
             True, [1])
    
    # Recuperar
    result = cache.get("X", "Y", "C1", hash(frozenset([1, 2, 3])), hash(frozenset([2, 3, 4])))
    
    assert result is not None
    revised, removed = result
    assert revised == True
    assert removed == [1]
    
    # Estadísticas
    stats = cache.get_statistics()
    print(f"\nEstadísticas del caché:")
    print(f"  Tamaño: {stats['size']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    
    assert stats['size'] == 1
    assert stats['hits'] >= 1
    
    print("\n✅ Test pasado")
    return True


def test_arc_ordering_by_domain_size():
    """Test: Ordenamiento de arcos por tamaño de dominio."""
    print("\n" + "=" * 60)
    print("TEST 2: Ordenamiento por Tamaño de Dominio")
    print("=" * 60)
    
    engine = ArcEngine()
    
    # Variables con diferentes tamaños de dominio
    engine.add_variable("X", [1, 2, 3, 4, 5])  # 5 valores
    engine.add_variable("Y", [1, 2])            # 2 valores
    engine.add_variable("Z", [1, 2, 3])         # 3 valores
    
    engine.add_constraint("X", "Y", NE(), cid="C1")
    engine.add_constraint("X", "Z", NE(), cid="C2")
    engine.add_constraint("Y", "Z", NE(), cid="C3")
    
    # Arcos
    arcs = [("X", "Y", "C1"), ("X", "Z", "C2"), ("Y", "Z", "C3")]
    
    # Ordenar
    ordered = ArcOrderingStrategy.order_by_domain_size(arcs, engine)
    
    print(f"\nArcos originales: {arcs}")
    print(f"Arcos ordenados: {ordered}")
    
    # Verificar que Y (2 valores) está primero
    assert ordered[0][0] == "Y" or ordered[0][1] == "Y"
    
    print("\n✅ Test pasado")
    return True


def test_redundant_arc_detection():
    """Test: Detección de arcos redundantes."""
    print("\n" + "=" * 60)
    print("TEST 3: Detección de Arcos Redundantes")
    print("=" * 60)
    
    engine = ArcEngine()
    
    # Variable con singleton
    engine.add_variable("X", [1])
    engine.add_variable("Y", [1, 2, 3])
    engine.add_constraint("X", "Y", NE(), cid="C1")
    
    # Verificar redundancia
    is_redundant = RedundantArcDetector.is_redundant("X", "Y", "C1", engine)
    
    print(f"\nDominio X: {list(engine.variables['X'].get_values())}")
    print(f"Dominio Y: {list(engine.variables['Y'].get_values())}")
    print(f"Arco (X, Y, C1) es redundante: {is_redundant}")
    
    assert is_redundant == True  # X es singleton
    
    print("\n✅ Test pasado")
    return True


def test_performance_monitor():
    """Test: Monitor de rendimiento."""
    print("\n" + "=" * 60)
    print("TEST 4: Monitor de Rendimiento")
    print("=" * 60)
    
    monitor = PerformanceMonitor()
    
    monitor.start()
    
    # Simular actividad
    monitor.record_iteration()
    monitor.record_revision(True, 2)
    monitor.record_revision(False, 0)
    monitor.record_arc_evaluation()
    monitor.record_cache_hit()
    monitor.record_cache_miss()
    
    monitor.end()
    
    # Estadísticas
    stats = monitor.get_statistics()
    
    print(f"\nEstadísticas:")
    print(f"  Iteraciones: {stats['iterations']}")
    print(f"  Revisiones: {stats['revisions']}")
    print(f"  Revisiones exitosas: {stats['successful_revisions']}")
    print(f"  Reducciones: {stats['domain_reductions']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    
    assert stats['iterations'] == 1
    assert stats['revisions'] == 2
    assert stats['successful_revisions'] == 1
    assert stats['domain_reductions'] == 2
    
    print("\n✅ Test pasado")
    return True


def test_optimized_ac3_basic():
    """Test: AC-3 optimizado básico."""
    print("\n" + "=" * 60)
    print("TEST 5: AC-3 Optimizado Básico")
    print("=" * 60)
    
    engine = ArcEngine()
    
    # Problema simple
    engine.add_variable("X", [1, 2, 3])
    engine.add_variable("Y", [1, 2, 3])
    engine.add_variable("Z", [1, 2, 3])
    
    engine.add_constraint("X", "Y", NE(), cid="C1")
    engine.add_constraint("X", "Z", NE(), cid="C2")
    engine.add_constraint("Y", "Z", NE(), cid="C3")
    
    print(f"\nProblema: AllDifferent(X, Y, Z)")
    
    # Crear AC-3 optimizado
    opt_ac3 = create_optimized_ac3(engine)
    
    # Ejecutar
    consistent = opt_ac3.enforce_arc_consistency_optimized()
    
    print(f"Consistente: {consistent}")
    print(f"Dominios:")
    for var in ["X", "Y", "Z"]:
        print(f"  {var}: {sorted(list(engine.variables[var].get_values()))}")
    
    # Estadísticas
    opt_ac3.print_statistics()
    
    assert consistent
    
    print("\n✅ Test pasado")
    return True


def test_optimized_ac3_with_cache():
    """Test: AC-3 optimizado con caché."""
    print("\n" + "=" * 60)
    print("TEST 6: AC-3 Optimizado con Caché")
    print("=" * 60)
    
    engine = ArcEngine()
    
    # Problema más grande
    n = 5
    for i in range(n):
        engine.add_variable(f"V{i}", list(range(1, n + 1)))
    
    for i in range(n):
        for j in range(i + 1, n):
            engine.add_constraint(f"V{i}", f"V{j}", NE(), cid=f"C{i}_{j}")
    
    print(f"\nProblema: AllDifferent con {n} variables")
    
    # Con caché
    opt_ac3 = create_optimized_ac3(engine, use_cache=True)
    consistent = opt_ac3.enforce_arc_consistency_optimized()
    
    print(f"Consistente: {consistent}")
    
    # Estadísticas
    stats = opt_ac3.get_statistics()
    
    if 'cache' in stats:
        cache_stats = stats['cache']
        print(f"\nCaché:")
        print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"  Hits: {cache_stats['hits']}")
        print(f"  Misses: {cache_stats['misses']}")
    
    assert consistent
    
    print("\n✅ Test pasado")
    return True


def test_comparison_normal_vs_optimized():
    """Test: Comparación AC-3 normal vs optimizado."""
    print("\n" + "=" * 60)
    print("TEST 7: Comparación Normal vs Optimizado")
    print("=" * 60)
    
    import time
    
    # Problema
    n = 6
    
    # Engine normal
    engine_normal = ArcEngine()
    for i in range(n):
        engine_normal.add_variable(f"V{i}", list(range(1, n + 2)))
    
    for i in range(n):
        for j in range(i + 1, n):
            engine_normal.add_constraint(f"V{i}", f"V{j}", NE(), cid=f"C{i}_{j}")
    
    # Engine optimizado
    engine_opt = ArcEngine()
    for i in range(n):
        engine_opt.add_variable(f"V{i}", list(range(1, n + 2)))
    
    for i in range(n):
        for j in range(i + 1, n):
            engine_opt.add_constraint(f"V{i}", f"V{j}", NE(), cid=f"C{i}_{j}")
    
    print(f"\nProblema: AllDifferent con {n} variables, dominio {n+1}")
    
    # Normal
    start = time.time()
    consistent_normal = engine_normal.enforce_arc_consistency()
    time_normal = time.time() - start
    
    print(f"\nNormal:")
    print(f"  Tiempo: {time_normal:.4f}s")
    print(f"  Consistente: {consistent_normal}")
    
    # Optimizado
    opt_ac3 = create_optimized_ac3(engine_opt, use_cache=True, use_ordering=True,
                                   use_redundancy_filter=True)
    start = time.time()
    consistent_opt = opt_ac3.enforce_arc_consistency_optimized()
    time_opt = time.time() - start
    
    print(f"\nOptimizado:")
    print(f"  Tiempo: {time_opt:.4f}s")
    print(f"  Consistente: {consistent_opt}")
    
    # Speedup
    if time_opt > 0:
        speedup = time_normal / time_opt
        print(f"\nSpeedup: {speedup:.2f}x")
    
    assert consistent_normal == consistent_opt
    
    print("\n✅ Test pasado")
    return True


def test_all_optimizations_combined():
    """Test: Todas las optimizaciones combinadas."""
    print("\n" + "=" * 60)
    print("TEST 8: Todas las Optimizaciones Combinadas")
    print("=" * 60)
    
    engine = ArcEngine()
    
    # Problema N-Reinas 4x4
    n = 4
    for i in range(n):
        engine.add_variable(f"Q{i}", list(range(n)))
    
    from lattice_weaver.arc_engine import NoAttackQueensConstraint
    
    for i in range(n):
        for j in range(i + 1, n):
            col_diff = j - i
            constraint = NoAttackQueensConstraint(col_diff)
            engine.add_constraint(f"Q{i}", f"Q{j}", constraint, cid=f"C{i}_{j}")
    
    print(f"\nProblema: {n}-Reinas")
    
    # AC-3 con todas las optimizaciones
    opt_ac3 = create_optimized_ac3(
        engine,
        use_cache=True,
        use_ordering=True,
        use_redundancy_filter=True,
        use_monitoring=True
    )
    
    consistent = opt_ac3.enforce_arc_consistency_optimized()
    
    print(f"Consistente: {consistent}")
    print(f"Dominios:")
    for i in range(n):
        domain = sorted(list(engine.variables[f"Q{i}"].get_values()))
        print(f"  Q{i}: {domain}")
    
    # Estadísticas completas
    opt_ac3.print_statistics()
    
    assert consistent
    
    print("\n✅ Test pasado")
    return True


def main():
    """Ejecuta todos los tests."""
    print("Tests de Optimizaciones de Rendimiento")
    print("LatticeWeaver v4\n")
    
    try:
        test_arc_revision_cache()
        test_arc_ordering_by_domain_size()
        test_redundant_arc_detection()
        test_performance_monitor()
        test_optimized_ac3_basic()
        test_optimized_ac3_with_cache()
        test_comparison_normal_vs_optimized()
        test_all_optimizations_combined()
        
        print("\n" + "=" * 60)
        print("TODOS LOS TESTS COMPLETADOS EXITOSAMENTE")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

