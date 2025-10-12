#!/usr/bin/env python3
# test_parallel_fca.py

"""
Tests para FCA Paralelo - Fase 10

Valida la implementación de ParallelFCABuilder.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_weaver.lattice_core import FormalContext, ParallelFCABuilder, LatticeBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_basic_parallel_fca():
    """
    Test básico: Contexto pequeño con FCA paralelo.
    """
    logger.info("=" * 60)
    logger.info("TEST 1: FCA Paralelo Básico")
    logger.info("=" * 60)
    
    # Crear contexto simple
    context = FormalContext()
    
    # Objetos: animales
    animals = ['perro', 'gato', 'pez']
    for animal in animals:
        context.add_object(animal)
    
    # Atributos: características
    attrs = ['tiene_patas', 'vive_agua', 'mamifero']
    for attr in attrs:
        context.add_attribute(attr)
    
    # Incidencias
    context.add_incidence('perro', 'tiene_patas')
    context.add_incidence('perro', 'mamifero')
    context.add_incidence('gato', 'tiene_patas')
    context.add_incidence('gato', 'mamifero')
    context.add_incidence('pez', 'vive_agua')
    
    logger.info(f"Contexto: {context.get_statistics()}")
    
    # Construir con FCA paralelo
    parallel_builder = ParallelFCABuilder(num_workers=2)
    concepts = parallel_builder.build_lattice_parallel(context)
    
    logger.info(f"\nConceptos encontrados: {len(concepts)}")
    
    for i, (extent, intent) in enumerate(sorted(concepts, key=lambda x: len(x[0])), 1):
        logger.info(f"  Concepto {i}: extent={set(extent)}, intent={set(intent)}")
    
    assert len(concepts) > 0, "Debe encontrar al menos un concepto"
    
    logger.info("\n✅ Test básico pasado")
    return True


def test_comparison_sequential_parallel():
    """
    Test de comparación: FCA secuencial vs paralelo.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Comparación Secuencial vs Paralelo")
    logger.info("=" * 60)
    
    # Crear contexto más grande
    context = FormalContext()
    
    # Objetos: números del 1 al 10
    for i in range(1, 11):
        context.add_object(f"n{i}")
    
    # Atributos: propiedades
    attrs = ['par', 'impar', 'primo', 'menor_5', 'mayor_5']
    for attr in attrs:
        context.add_attribute(attr)
    
    # Definir incidencias
    primos = [2, 3, 5, 7]
    for i in range(1, 11):
        obj = f"n{i}"
        
        if i % 2 == 0:
            context.add_incidence(obj, 'par')
        else:
            context.add_incidence(obj, 'impar')
        
        if i in primos:
            context.add_incidence(obj, 'primo')
        
        if i < 5:
            context.add_incidence(obj, 'menor_5')
        else:
            context.add_incidence(obj, 'mayor_5')
    
    logger.info(f"Contexto: {context.get_statistics()}")
    
    # FCA secuencial
    logger.info("\nEjecutando FCA secuencial...")
    sequential_builder = LatticeBuilder()
    sequential_builder.context = context
    sequential_concepts = sequential_builder.build_concept_lattice()
    
    logger.info(f"Conceptos (secuencial): {len(sequential_concepts)}")
    
    # FCA paralelo
    logger.info("\nEjecutando FCA paralelo...")
    parallel_builder = ParallelFCABuilder(num_workers=4)
    parallel_concepts = parallel_builder.build_lattice_parallel(context)
    
    logger.info(f"Conceptos (paralelo): {len(parallel_concepts)}")
    
    # Comparar resultados
    logger.info(f"\nDiferencia en número de conceptos: {abs(len(sequential_concepts) - len(parallel_concepts))}")
    
    # Verificar que ambos encuentran conceptos
    assert len(sequential_concepts) > 0, "FCA secuencial debe encontrar conceptos"
    assert len(parallel_concepts) > 0, "FCA paralelo debe encontrar conceptos"
    
    logger.info("\n✅ Test de comparación pasado")
    return True


def test_empty_context():
    """
    Test de caso borde: Contexto vacío.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Contexto Vacío")
    logger.info("=" * 60)
    
    context = FormalContext()
    
    parallel_builder = ParallelFCABuilder()
    concepts = parallel_builder.build_lattice_parallel(context)
    
    logger.info(f"Conceptos en contexto vacío: {len(concepts)}")
    
    assert len(concepts) == 0, "Contexto vacío debe retornar 0 conceptos"
    
    logger.info("\n✅ Test de contexto vacío pasado")
    return True


def test_large_context():
    """
    Test de rendimiento: Contexto grande.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Contexto Grande")
    logger.info("=" * 60)
    
    import time
    
    # Crear contexto grande
    context = FormalContext()
    
    # 20 objetos
    for i in range(1, 21):
        context.add_object(f"obj{i}")
    
    # 8 atributos
    for i in range(1, 9):
        context.add_attribute(f"attr{i}")
    
    # Incidencias semi-aleatorias (determinísticas)
    for i in range(1, 21):
        for j in range(1, 9):
            if (i + j) % 3 == 0:
                context.add_incidence(f"obj{i}", f"attr{j}")
    
    logger.info(f"Contexto: {context.get_statistics()}")
    
    # Medir tiempo FCA paralelo
    start = time.time()
    parallel_builder = ParallelFCABuilder(num_workers=4)
    concepts = parallel_builder.build_lattice_parallel(context)
    elapsed = time.time() - start
    
    logger.info(f"\nConceptos encontrados: {len(concepts)}")
    logger.info(f"Tiempo: {elapsed:.3f} segundos")
    
    assert len(concepts) > 0, "Debe encontrar conceptos"
    assert elapsed < 60, "Debe completar en menos de 60 segundos"
    
    logger.info("\n✅ Test de contexto grande pasado")
    return True


def main():
    """Ejecuta todos los tests."""
    logger.info("Iniciando suite de tests para FCA Paralelo")
    logger.info("LatticeWeaver v4 - Fase 10\n")
    
    try:
        test_basic_parallel_fca()
        test_comparison_sequential_parallel()
        test_empty_context()
        test_large_context()
        
        logger.info("\n" + "=" * 60)
        logger.info("TODOS LOS TESTS COMPLETADOS EXITOSAMENTE")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"\nError durante las pruebas: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

