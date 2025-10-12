#!/usr/bin/env python3
# test_parallel_fca_logic.py

"""
Tests de lógica para FCA Paralelo - Fase 10

Valida la lógica de ParallelFCABuilder sin usar multiprocessing real.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_weaver.lattice_core import FormalContext, ParallelFCABuilder


def test_serializable_context():
    """Test: Conversión de contexto a formato serializable."""
    print("=" * 60)
    print("TEST 1: Contexto Serializable")
    print("=" * 60)
    
    context = FormalContext()
    
    # Crear contexto simple
    context.add_object('a')
    context.add_object('b')
    context.add_attribute('x')
    context.add_attribute('y')
    context.add_incidence('a', 'x')
    context.add_incidence('b', 'y')
    
    # Convertir a serializable
    builder = ParallelFCABuilder(num_workers=1)
    ser_context = builder._make_serializable_context(context)
    
    print(f"Objetos: {ser_context['objects']}")
    print(f"Atributos: {ser_context['attributes']}")
    print(f"Incidencias: {len(ser_context['incidence'])}")
    print(f"Mapeo obj→attrs: {ser_context['obj_to_attrs']}")
    print(f"Mapeo attr→objs: {ser_context['attr_to_objs']}")
    
    assert 'objects' in ser_context
    assert 'attributes' in ser_context
    assert 'obj_to_attrs' in ser_context
    assert 'attr_to_objs' in ser_context
    
    print("\n✅ Test pasado")
    return True


def test_intent_extent_computation():
    """Test: Cálculo de intent y extent."""
    print("\n" + "=" * 60)
    print("TEST 2: Cálculo de Intent y Extent")
    print("=" * 60)
    
    context = FormalContext()
    
    # Animales
    context.add_object('perro')
    context.add_object('gato')
    context.add_object('pez')
    
    # Atributos
    context.add_attribute('patas')
    context.add_attribute('mamifero')
    context.add_attribute('agua')
    
    # Incidencias
    context.add_incidence('perro', 'patas')
    context.add_incidence('perro', 'mamifero')
    context.add_incidence('gato', 'patas')
    context.add_incidence('gato', 'mamifero')
    context.add_incidence('pez', 'agua')
    
    builder = ParallelFCABuilder(num_workers=1)
    ser_context = builder._make_serializable_context(context)
    
    # Test 1: Intent de {perro, gato}
    extent1 = frozenset(['perro', 'gato'])
    intent1 = builder._compute_intent(extent1, ser_context)
    print(f"\nIntent de {set(extent1)}: {set(intent1)}")
    assert 'patas' in intent1
    assert 'mamifero' in intent1
    assert 'agua' not in intent1
    
    # Test 2: Extent de {patas}
    intent2 = frozenset(['patas'])
    extent2 = builder._compute_extent(intent2, ser_context)
    print(f"Extent de {set(intent2)}: {set(extent2)}")
    assert 'perro' in extent2
    assert 'gato' in extent2
    assert 'pez' not in extent2
    
    # Test 3: Concepto formal
    is_formal = builder._is_formal_concept(extent2, intent2, ser_context)
    print(f"\n¿({set(extent2)}, {set(intent2)}) es concepto formal? {is_formal}")
    
    print("\n✅ Test pasado")
    return True


def test_closure_computation():
    """Test: Cálculo de cierre de conceptos."""
    print("\n" + "=" * 60)
    print("TEST 3: Cálculo de Cierre")
    print("=" * 60)
    
    context = FormalContext()
    
    # Contexto simple
    for i in range(1, 4):
        context.add_object(f"o{i}")
        context.add_attribute(f"a{i}")
    
    # Incidencias diagonales
    for i in range(1, 4):
        context.add_incidence(f"o{i}", f"a{i}")
    
    builder = ParallelFCABuilder(num_workers=1)
    ser_context = builder._make_serializable_context(context)
    
    # Conceptos iniciales
    initial_concepts = set()
    initial_concepts.add((frozenset(), frozenset(context.attributes)))
    initial_concepts.add((frozenset(context.objects), frozenset()))
    
    print(f"Conceptos iniciales: {len(initial_concepts)}")
    
    # Calcular cierre
    closed = builder._compute_closure(initial_concepts, ser_context)
    
    print(f"Conceptos después del cierre: {len(closed)}")
    
    for extent, intent in sorted(closed, key=lambda x: len(x[0])):
        print(f"  ({set(extent)}, {set(intent)})")
    
    assert len(closed) >= len(initial_concepts)
    
    print("\n✅ Test pasado")
    return True


def test_chunk_processing_logic():
    """Test: Lógica de procesamiento de chunks."""
    print("\n" + "=" * 60)
    print("TEST 4: Lógica de Procesamiento de Chunks")
    print("=" * 60)
    
    from lattice_weaver.lattice_core.parallel_fca import _compute_concepts_for_chunk
    
    context = FormalContext()
    
    # Crear contexto
    context.add_object('x')
    context.add_object('y')
    context.add_attribute('p')
    context.add_attribute('q')
    context.add_incidence('x', 'p')
    context.add_incidence('y', 'q')
    
    builder = ParallelFCABuilder(num_workers=1)
    ser_context = builder._make_serializable_context(context)
    
    # Procesar chunk
    chunk = ['x', 'y']
    concepts = _compute_concepts_for_chunk(chunk, ser_context)
    
    print(f"\nConceptos encontrados en chunk: {len(concepts)}")
    for extent, intent in sorted(concepts, key=lambda x: len(x[0])):
        print(f"  ({set(extent)}, {set(intent)})")
    
    assert len(concepts) > 0
    
    print("\n✅ Test pasado")
    return True


def main():
    """Ejecuta todos los tests."""
    print("Tests de Lógica para FCA Paralelo - Fase 10")
    print("LatticeWeaver v4\n")
    
    try:
        test_serializable_context()
        test_intent_extent_computation()
        test_closure_computation()
        test_chunk_processing_logic()
        
        print("\n" + "=" * 60)
        print("TODOS LOS TESTS DE LÓGICA COMPLETADOS EXITOSAMENTE")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

