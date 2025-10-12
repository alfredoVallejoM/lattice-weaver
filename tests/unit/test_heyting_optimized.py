#!/usr/bin/env python3
# test_heyting_optimized.py

"""
Tests para Álgebra de Heyting Optimizada - Fase 11

Valida la implementación de OptimizedHeytingAlgebra con caché y precomputación.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_weaver.formal import HeytingElement, OptimizedHeytingAlgebra


def test_basic_optimized_operations():
    """Test básico: Operaciones meet y join optimizadas."""
    print("=" * 60)
    print("TEST 1: Operaciones Básicas Optimizadas")
    print("=" * 60)
    
    # Crear álgebra
    algebra = OptimizedHeytingAlgebra("Test")
    
    # Crear elementos
    bottom = HeytingElement("⊥", frozenset())
    a = HeytingElement("a", frozenset({1}))
    b = HeytingElement("b", frozenset({2}))
    ab = HeytingElement("a∧b", frozenset())
    ab_join = HeytingElement("a∨b", frozenset({1, 2}))
    top = HeytingElement("⊤", frozenset({1, 2, 3}))
    
    # Añadir elementos
    for elem in [bottom, a, b, ab, ab_join, top]:
        algebra.add_element(elem)
    
    # Definir orden
    algebra.set_bottom(bottom)
    algebra.set_top(top)
    
    algebra.add_order(bottom, a)
    algebra.add_order(bottom, b)
    algebra.add_order(bottom, ab)
    algebra.add_order(ab, a)
    algebra.add_order(ab, b)
    algebra.add_order(a, ab_join)
    algebra.add_order(b, ab_join)
    algebra.add_order(ab_join, top)
    algebra.add_order(a, top)
    algebra.add_order(b, top)
    
    # Test meet
    result_meet = algebra.meet(a, b)
    print(f"\nMeet: {a} ∧ {b} = {result_meet}")
    assert result_meet == ab or result_meet == bottom
    
    # Test join
    result_join = algebra.join(a, b)
    print(f"Join: {a} ∨ {b} = {result_join}")
    assert result_join == ab_join
    
    # Test caché (segunda llamada)
    result_meet2 = algebra.meet(a, b)
    assert result_meet2 == result_meet
    
    print("\n✅ Test básico pasado")
    return True


def test_cache_statistics():
    """Test: Estadísticas de caché."""
    print("\n" + "=" * 60)
    print("TEST 2: Estadísticas de Caché")
    print("=" * 60)
    
    algebra = OptimizedHeytingAlgebra("Cache")
    
    # Crear elementos simples
    bottom = HeytingElement("⊥")
    a = HeytingElement("a")
    b = HeytingElement("b")
    c = HeytingElement("c")
    top = HeytingElement("⊤")
    
    for elem in [bottom, a, b, c, top]:
        algebra.add_element(elem)
    
    algebra.set_bottom(bottom)
    algebra.set_top(top)
    
    # Orden lineal: ⊥ < a < b < c < ⊤
    algebra.add_order(bottom, a)
    algebra.add_order(a, b)
    algebra.add_order(b, c)
    algebra.add_order(c, top)
    
    # Primera llamada (miss)
    algebra.meet(a, b)
    
    # Segunda llamada (hit)
    algebra.meet(a, b)
    algebra.meet(b, a)  # Conmutatividad
    
    # Estadísticas
    stats = algebra.get_cache_statistics()
    print(f"\nEstadísticas de caché:")
    print(f"  Meet hits: {stats['meet_hits']}")
    print(f"  Meet misses: {stats['meet_misses']}")
    print(f"  Meet hit rate: {stats['meet_hit_rate']}%")
    print(f"  Tamaño caché meet: {stats['meet_cache_size']}")
    
    assert stats['meet_hits'] >= 2, "Debe tener al menos 2 hits"
    assert stats['meet_misses'] >= 1, "Debe tener al menos 1 miss"
    
    print("\n✅ Test de caché pasado")
    return True


def test_precomputation():
    """Test: Precomputación de meets frecuentes."""
    print("\n" + "=" * 60)
    print("TEST 3: Precomputación de Meets")
    print("=" * 60)
    
    algebra = OptimizedHeytingAlgebra("Precomp")
    
    # Crear retículo más grande
    bottom = HeytingElement("⊥")
    elements = [HeytingElement(f"e{i}") for i in range(1, 6)]
    top = HeytingElement("⊤")
    
    algebra.add_element(bottom)
    for elem in elements:
        algebra.add_element(elem)
    algebra.add_element(top)
    
    algebra.set_bottom(bottom)
    algebra.set_top(top)
    
    # Orden: ⊥ < e1, e2, e3, e4, e5 < ⊤
    for elem in elements:
        algebra.add_order(bottom, elem)
        algebra.add_order(elem, top)
    
    # Precomputar
    print("\nPrecomputando meets...")
    algebra.precompute_frequent_meets()
    
    stats = algebra.get_cache_statistics()
    print(f"Meets precomputados: {stats['meet_cache_size']}")
    
    # Verificar que se precomputaron algunos
    assert stats['meet_cache_size'] > 0, "Debe haber precomputado algunos meets"
    
    # Usar meets precomputados
    algebra.meet(elements[0], elements[1])
    
    stats_after = algebra.get_cache_statistics()
    print(f"Hits después de usar precomputados: {stats_after['meet_hits']}")
    
    print("\n✅ Test de precomputación pasado")
    return True


def test_meet_multiple():
    """Test: Meet de múltiples elementos."""
    print("\n" + "=" * 60)
    print("TEST 4: Meet de Múltiples Elementos")
    print("=" * 60)
    
    algebra = OptimizedHeytingAlgebra("Multiple")
    
    # Crear elementos con valores de conjunto
    bottom = HeytingElement("⊥", frozenset())
    e1 = HeytingElement("e1", frozenset({1, 2, 3}))
    e2 = HeytingElement("e2", frozenset({2, 3, 4}))
    e3 = HeytingElement("e3", frozenset({3, 4, 5}))
    e12 = HeytingElement("e1∧e2", frozenset({2, 3}))
    e23 = HeytingElement("e2∧e3", frozenset({3, 4}))
    e123 = HeytingElement("e1∧e2∧e3", frozenset({3}))
    top = HeytingElement("⊤", frozenset({1, 2, 3, 4, 5}))
    
    for elem in [bottom, e1, e2, e3, e12, e23, e123, top]:
        algebra.add_element(elem)
    
    algebra.set_bottom(bottom)
    algebra.set_top(top)
    
    # Definir orden basado en inclusión de conjuntos
    all_elems = [bottom, e1, e2, e3, e12, e23, e123, top]
    for a in all_elems:
        for b in all_elems:
            if a.value is not None and b.value is not None:
                if a.value.issubset(b.value):
                    algebra.add_order(a, b)
    
    # Meet de múltiples elementos
    elements_to_meet = [e1, e2, e3]
    result = algebra.meet_multiple(elements_to_meet)
    
    print(f"\nMeet de {[str(e) for e in elements_to_meet]}:")
    print(f"  Resultado: {result}")
    print(f"  Valor: {result.value}")
    
    # Verificar que el resultado es la intersección
    expected_value = frozenset({3})
    assert result.value == expected_value, f"Esperado {expected_value}, obtenido {result.value}"
    
    # Estadísticas
    stats = algebra.get_cache_statistics()
    print(f"\nEstadísticas:")
    print(f"  Meet calls: {stats['meet_hits'] + stats['meet_misses']}")
    print(f"  Cache hit rate: {stats['meet_hit_rate']}%")
    
    print("\n✅ Test de meet múltiple pasado")
    return True


def test_join_multiple():
    """Test: Join de múltiples elementos."""
    print("\n" + "=" * 60)
    print("TEST 5: Join de Múltiples Elementos")
    print("=" * 60)
    
    algebra = OptimizedHeytingAlgebra("JoinMultiple")
    
    # Crear elementos
    bottom = HeytingElement("⊥", frozenset())
    e1 = HeytingElement("e1", frozenset({1}))
    e2 = HeytingElement("e2", frozenset({2}))
    e3 = HeytingElement("e3", frozenset({3}))
    e12 = HeytingElement("e1∨e2", frozenset({1, 2}))
    e23 = HeytingElement("e2∨e3", frozenset({2, 3}))
    e123 = HeytingElement("e1∨e2∨e3", frozenset({1, 2, 3}))
    top = HeytingElement("⊤", frozenset({1, 2, 3, 4}))
    
    for elem in [bottom, e1, e2, e3, e12, e23, e123, top]:
        algebra.add_element(elem)
    
    algebra.set_bottom(bottom)
    algebra.set_top(top)
    
    # Orden basado en inclusión
    all_elems = [bottom, e1, e2, e3, e12, e23, e123, top]
    for a in all_elems:
        for b in all_elems:
            if a.value is not None and b.value is not None:
                if a.value.issubset(b.value):
                    algebra.add_order(a, b)
    
    # Join de múltiples elementos
    elements_to_join = [e1, e2, e3]
    result = algebra.join_multiple(elements_to_join)
    
    print(f"\nJoin de {[str(e) for e in elements_to_join]}:")
    print(f"  Resultado: {result}")
    print(f"  Valor: {result.value}")
    
    # Verificar que el resultado es la unión
    expected_value = frozenset({1, 2, 3})
    assert result.value == expected_value, f"Esperado {expected_value}, obtenido {result.value}"
    
    print("\n✅ Test de join múltiple pasado")
    return True


def main():
    """Ejecuta todos los tests."""
    print("Tests de Álgebra de Heyting Optimizada - Fase 11")
    print("LatticeWeaver v4\n")
    
    try:
        test_basic_optimized_operations()
        test_cache_statistics()
        test_precomputation()
        test_meet_multiple()
        test_join_multiple()
        
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

