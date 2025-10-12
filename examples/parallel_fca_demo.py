#!/usr/bin/env python3
# parallel_fca_demo.py

"""
Demostración de FCA Paralelo - Fase 10

Ejemplo de uso de ParallelFCABuilder para construir retículos de conceptos
de forma paralela.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_weaver.lattice_core import FormalContext, ParallelFCABuilder


def demo_animals():
    """Demostración con contexto de animales."""
    print("=" * 70)
    print("DEMO: FCA Paralelo - Clasificación de Animales")
    print("=" * 70)
    
    # Crear contexto
    context = FormalContext()
    
    # Objetos: animales
    animals = {
        'perro': ['mamifero', 'terrestre', 'carnivoro', 'domestico'],
        'gato': ['mamifero', 'terrestre', 'carnivoro', 'domestico'],
        'leon': ['mamifero', 'terrestre', 'carnivoro', 'salvaje'],
        'caballo': ['mamifero', 'terrestre', 'herbivoro', 'domestico'],
        'delfin': ['mamifero', 'acuatico', 'carnivoro', 'salvaje'],
        'tiburon': ['pez', 'acuatico', 'carnivoro', 'salvaje'],
        'salmon': ['pez', 'acuatico', 'carnivoro', 'salvaje'],
        'aguila': ['ave', 'volador', 'carnivoro', 'salvaje'],
        'paloma': ['ave', 'volador', 'herbivoro', 'domestico']
    }
    
    # Añadir objetos
    for animal in animals.keys():
        context.add_object(animal)
    
    # Añadir atributos
    all_attrs = set()
    for attrs in animals.values():
        all_attrs.update(attrs)
    
    for attr in all_attrs:
        context.add_attribute(attr)
    
    # Añadir incidencias
    for animal, attrs in animals.items():
        for attr in attrs:
            context.add_incidence(animal, attr)
    
    # Mostrar estadísticas
    stats = context.get_statistics()
    print(f"\nEstadísticas del contexto:")
    print(f"  Objetos (animales): {stats['num_objects']}")
    print(f"  Atributos: {stats['num_attributes']}")
    print(f"  Incidencias: {stats['num_incidences']}")
    print(f"  Densidad: {stats['density']:.2%}")
    
    # Construir retículo con FCA paralelo
    print(f"\nConstruyendo retículo de conceptos (paralelo, 4 workers)...")
    
    builder = ParallelFCABuilder(num_workers=4)
    concepts = builder.build_lattice_parallel(context)
    
    print(f"\nConceptos formales encontrados: {len(concepts)}")
    
    # Mostrar algunos conceptos interesantes
    print("\nAlgunos conceptos interesantes:")
    
    for extent, intent in sorted(concepts, key=lambda x: (len(x[0]), len(x[1]))):
        if len(extent) > 0 and len(intent) > 0:
            print(f"\n  Concepto:")
            print(f"    Extent (objetos): {set(extent)}")
            print(f"    Intent (atributos): {set(intent)}")
            
            # Limitar a primeros 5 conceptos no triviales
            if len([c for c in concepts if len(c[0]) > 0 and len(c[1]) > 0]) > 5:
                break
    
    print("\n" + "=" * 70)


def demo_numbers():
    """Demostración con contexto de números."""
    print("\n" + "=" * 70)
    print("DEMO: FCA Paralelo - Propiedades de Números")
    print("=" * 70)
    
    # Crear contexto
    context = FormalContext()
    
    # Objetos: números del 1 al 15
    for i in range(1, 16):
        context.add_object(i)
    
    # Atributos: propiedades
    attrs = ['par', 'impar', 'primo', 'cuadrado', 'menor_10', 'mayor_10', 'divisible_3']
    for attr in attrs:
        context.add_attribute(attr)
    
    # Definir propiedades
    primos = [2, 3, 5, 7, 11, 13]
    cuadrados = [1, 4, 9]
    
    for i in range(1, 16):
        # Par/Impar
        if i % 2 == 0:
            context.add_incidence(i, 'par')
        else:
            context.add_incidence(i, 'impar')
        
        # Primo
        if i in primos:
            context.add_incidence(i, 'primo')
        
        # Cuadrado
        if i in cuadrados:
            context.add_incidence(i, 'cuadrado')
        
        # Menor/Mayor 10
        if i < 10:
            context.add_incidence(i, 'menor_10')
        else:
            context.add_incidence(i, 'mayor_10')
        
        # Divisible por 3
        if i % 3 == 0:
            context.add_incidence(i, 'divisible_3')
    
    # Estadísticas
    stats = context.get_statistics()
    print(f"\nEstadísticas del contexto:")
    print(f"  Objetos (números): {stats['num_objects']}")
    print(f"  Atributos: {stats['num_attributes']}")
    print(f"  Densidad: {stats['density']:.2%}")
    
    # Construir retículo
    print(f"\nConstruyendo retículo de conceptos (paralelo)...")
    
    builder = ParallelFCABuilder(num_workers=2)
    concepts = builder.build_lattice_parallel(context)
    
    print(f"\nConceptos formales encontrados: {len(concepts)}")
    
    # Buscar concepto de números primos impares menores a 10
    print("\nBuscando concepto: números primos impares menores a 10...")
    
    target_intent = frozenset(['primo', 'impar', 'menor_10'])
    
    for extent, intent in concepts:
        if target_intent.issubset(intent):
            print(f"  Encontrado: extent={set(extent)}, intent={set(intent)}")
    
    print("\n" + "=" * 70)


def main():
    """Ejecuta las demostraciones."""
    print("\nDemostraciones de FCA Paralelo - LatticeWeaver v4")
    print("Fase 10: Paralelización del Análisis Formal de Conceptos\n")
    
    try:
        demo_animals()
        demo_numbers()
        
        print("\n✅ Demostraciones completadas exitosamente\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

