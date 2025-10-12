#!/usr/bin/env python3
# heyting_optimized_demo.py

"""
Demostración de Álgebra de Heyting Optimizada - Fase 11

Ejemplo de uso de OptimizedHeytingAlgebra con caché y precomputación
para operaciones lógicas eficientes.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_weaver.formal import HeytingElement, OptimizedHeytingAlgebra


def demo_power_set_algebra():
    """Demostración con álgebra de conjuntos potencia."""
    print("=" * 70)
    print("DEMO: Álgebra de Heyting Optimizada - Conjuntos Potencia")
    print("=" * 70)
    
    # Crear álgebra optimizada
    algebra = OptimizedHeytingAlgebra("PowerSet({1,2,3})")
    
    # Elementos: subconjuntos de {1, 2, 3}
    subsets = {
        "∅": frozenset(),
        "{1}": frozenset({1}),
        "{2}": frozenset({2}),
        "{3}": frozenset({3}),
        "{1,2}": frozenset({1, 2}),
        "{1,3}": frozenset({1, 3}),
        "{2,3}": frozenset({2, 3}),
        "{1,2,3}": frozenset({1, 2, 3})
    }
    
    # Crear elementos
    elements = {}
    for name, value in subsets.items():
        elem = HeytingElement(name, value)
        elements[name] = elem
        algebra.add_element(elem)
    
    # Definir orden (inclusión de conjuntos)
    algebra.set_bottom(elements["∅"])
    algebra.set_top(elements["{1,2,3}"])
    
    for name1, set1 in subsets.items():
        for name2, set2 in subsets.items():
            if set1.issubset(set2):
                algebra.add_order(elements[name1], elements[name2])
    
    print(f"\nÁlgebra creada con {len(algebra.elements)} elementos")
    
    # Precomputar meets frecuentes
    print("\nPrecomputando meets frecuentes...")
    algebra.precompute_frequent_meets()
    
    stats = algebra.get_cache_statistics()
    print(f"Meets precomputados: {stats['meet_cache_size']}")
    
    # Operaciones lógicas
    print("\n--- Operaciones Lógicas ---")
    
    # Conjunción (meet)
    a = elements["{1,2}"]
    b = elements["{2,3}"]
    meet_result = algebra.meet(a, b)
    print(f"\n{a} ∧ {b} = {meet_result}")
    print(f"  (Intersección: {a.value} ∩ {b.value} = {meet_result.value})")
    
    # Disyunción (join)
    join_result = algebra.join(a, b)
    print(f"\n{a} ∨ {b} = {join_result}")
    print(f"  (Unión: {a.value} ∪ {b.value} = {join_result.value})")
    
    # Implicación
    impl_result = algebra.implies(a, b)
    print(f"\n{a} → {b} = {impl_result}")
    
    # Negación
    neg_a = algebra.neg(a)
    print(f"\n¬{a} = {neg_a}")
    
    # Meet múltiple
    elements_to_meet = [elements["{1,2}"], elements["{2,3}"], elements["{1,2,3}"]]
    multi_meet = algebra.meet_multiple(elements_to_meet)
    print(f"\nMeet de {[str(e) for e in elements_to_meet]}:")
    print(f"  Resultado: {multi_meet}")
    
    # Estadísticas finales
    print("\n--- Estadísticas de Caché ---")
    final_stats = algebra.get_cache_statistics()
    print(f"Meet hits: {final_stats['meet_hits']}")
    print(f"Meet misses: {final_stats['meet_misses']}")
    print(f"Hit rate: {final_stats['meet_hit_rate']}%")
    print(f"Join hits: {final_stats['join_hits']}")
    print(f"Join misses: {final_stats['join_misses']}")
    
    print("\n" + "=" * 70)


def demo_logic_propositions():
    """Demostración con proposiciones lógicas."""
    print("\n" + "=" * 70)
    print("DEMO: Lógica Proposicional Intuicionista")
    print("=" * 70)
    
    algebra = OptimizedHeytingAlgebra("Propositions")
    
    # Crear proposiciones
    false = HeytingElement("⊥")
    p = HeytingElement("p")
    q = HeytingElement("q")
    r = HeytingElement("r")
    p_and_q = HeytingElement("p∧q")
    p_or_q = HeytingElement("p∨q")
    q_or_r = HeytingElement("q∨r")
    p_or_q_or_r = HeytingElement("p∨q∨r")
    true = HeytingElement("⊤")
    
    propositions = [false, p, q, r, p_and_q, p_or_q, q_or_r, p_or_q_or_r, true]
    
    for prop in propositions:
        algebra.add_element(prop)
    
    algebra.set_bottom(false)
    algebra.set_top(true)
    
    # Definir orden lógico
    # ⊥ < p, q, r
    for prop in [p, q, r]:
        algebra.add_order(false, prop)
    
    # p∧q < p, q
    algebra.add_order(p_and_q, p)
    algebra.add_order(p_and_q, q)
    algebra.add_order(false, p_and_q)
    
    # p, q < p∨q
    algebra.add_order(p, p_or_q)
    algebra.add_order(q, p_or_q)
    
    # q, r < q∨r
    algebra.add_order(q, q_or_r)
    algebra.add_order(r, q_or_r)
    
    # p∨q, r < p∨q∨r
    algebra.add_order(p_or_q, p_or_q_or_r)
    algebra.add_order(r, p_or_q_or_r)
    algebra.add_order(p, p_or_q_or_r)
    algebra.add_order(q, p_or_q_or_r)
    
    # Todo < ⊤
    for prop in propositions[:-1]:
        algebra.add_order(prop, true)
    
    print(f"\nÁlgebra de proposiciones con {len(algebra.elements)} elementos")
    
    # Precomputar
    algebra.precompute_frequent_meets()
    
    # Razonamiento lógico
    print("\n--- Razonamiento Lógico ---")
    
    # p ∧ q
    meet_pq = algebra.meet(p, q)
    print(f"\np ∧ q = {meet_pq}")
    
    # p ∨ q
    join_pq = algebra.join(p, q)
    print(f"p ∨ q = {join_pq}")
    
    # (p ∨ q) ∧ r
    meet_pqr = algebra.meet(p_or_q, r)
    print(f"(p ∨ q) ∧ r = {meet_pqr}")
    
    # p → q
    impl_pq = algebra.implies(p, q)
    print(f"p → q = {impl_pq}")
    
    # ¬p
    neg_p = algebra.neg(p)
    print(f"¬p = {neg_p}")
    
    # Estadísticas
    stats = algebra.get_cache_statistics()
    print(f"\nOperaciones totales: {stats['meet_hits'] + stats['meet_misses'] + stats['join_hits'] + stats['join_misses']}")
    print(f"Eficiencia de caché: {(stats['meet_hit_rate'] + stats['join_hit_rate']) / 2:.1f}%")
    
    print("\n" + "=" * 70)


def demo_performance_comparison():
    """Demostración de mejora de rendimiento."""
    print("\n" + "=" * 70)
    print("DEMO: Comparación de Rendimiento")
    print("=" * 70)
    
    import time
    
    # Crear álgebra grande
    algebra = OptimizedHeytingAlgebra("Performance")
    
    # Generar elementos
    n = 20
    elements = []
    for i in range(n):
        elem = HeytingElement(f"e{i}", frozenset({i}))
        elements.append(elem)
        algebra.add_element(elem)
    
    bottom = HeytingElement("⊥", frozenset())
    top = HeytingElement("⊤", frozenset(range(n)))
    
    algebra.add_element(bottom)
    algebra.add_element(top)
    algebra.set_bottom(bottom)
    algebra.set_top(top)
    
    # Orden
    for elem in elements:
        algebra.add_order(bottom, elem)
        algebra.add_order(elem, top)
    
    print(f"\nÁlgebra con {len(algebra.elements)} elementos")
    
    # Sin precomputación
    algebra.clear_cache()
    start = time.time()
    for i in range(min(10, n-1)):
        algebra.meet(elements[i], elements[i+1])
    time_no_precomp = time.time() - start
    
    stats_no_precomp = algebra.get_cache_statistics()
    
    # Con precomputación
    algebra.clear_cache()
    algebra.precompute_frequent_meets()
    
    start = time.time()
    for i in range(min(10, n-1)):
        algebra.meet(elements[i], elements[i+1])
    time_with_precomp = time.time() - start
    
    stats_with_precomp = algebra.get_cache_statistics()
    
    # Resultados
    print("\n--- Sin Precomputación ---")
    print(f"Tiempo: {time_no_precomp*1000:.2f} ms")
    print(f"Hit rate: {stats_no_precomp['meet_hit_rate']}%")
    
    print("\n--- Con Precomputación ---")
    print(f"Tiempo: {time_with_precomp*1000:.2f} ms")
    print(f"Hit rate: {stats_with_precomp['meet_hit_rate']}%")
    print(f"Meets precomputados: {stats_with_precomp['meet_cache_size']}")
    
    if time_no_precomp > 0:
        speedup = time_no_precomp / time_with_precomp if time_with_precomp > 0 else float('inf')
        print(f"\nSpeedup: {speedup:.2f}x")
    
    print("\n" + "=" * 70)


def main():
    """Ejecuta las demostraciones."""
    print("\nDemostraciones de Álgebra de Heyting Optimizada")
    print("LatticeWeaver v4 - Fase 11\n")
    
    try:
        demo_power_set_algebra()
        demo_logic_propositions()
        demo_performance_comparison()
        
        print("\n✅ Demostraciones completadas exitosamente\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

