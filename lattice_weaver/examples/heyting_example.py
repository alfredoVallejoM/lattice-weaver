"""
Ejemplo de Uso: Álgebra de Heyting en LatticeWeaver

Este ejemplo demuestra cómo usar el álgebra de Heyting para razonamiento
lógico intuicionista sobre problemas CSP.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
import os

# Añadir el directorio padre al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lattice_weaver.formal import HeytingAlgebra, HeytingElement, create_power_set_algebra
from lattice_weaver.formal import lattice_to_heyting, heyting_to_logic_table
from lattice_weaver.lattice_core import FormalContext, LatticeBuilder


def example_1_power_set():
    """Ejemplo 1: Álgebra de Heyting desde conjunto potencia."""
    print("="*70)
    print("EJEMPLO 1: ÁLGEBRA DE HEYTING DESDE CONJUNTO POTENCIA")
    print("="*70)
    
    # Crear álgebra desde el conjunto {1, 2, 3}
    base = {1, 2, 3}
    algebra = create_power_set_algebra(base, "P({1,2,3})")
    
    print(f"\nÁlgebra creada: {algebra}")
    print(f"Número de elementos: {len(algebra.elements)}")
    print(f"⊥ = {algebra.bottom}")
    print(f"⊤ = {algebra.top}")
    
    # Encontrar algunos elementos
    set1 = None
    set2 = None
    set12 = None
    
    for elem in algebra.elements:
        if elem.value == frozenset({1}):
            set1 = elem
        elif elem.value == frozenset({2}):
            set2 = elem
        elif elem.value == frozenset({1, 2}):
            set12 = elem
    
    print(f"\nOperaciones lógicas:")
    print(f"  {set1} ∧ {set2} = {algebra.meet(set1, set2)}")
    print(f"  {set1} ∨ {set2} = {algebra.join(set1, set2)}")
    print(f"  {set1} → {set2} = {algebra.implies(set1, set2)}")
    print(f"  ¬{set1} = {algebra.neg(set1)}")
    
    print(f"\nValidación del álgebra: {algebra.is_valid()}")


def example_2_lattice_to_heyting():
    """Ejemplo 2: Convertir retículo de conceptos a álgebra de Heyting."""
    print("\n" + "="*70)
    print("EJEMPLO 2: RETÍCULO DE CONCEPTOS → ÁLGEBRA DE HEYTING")
    print("="*70)
    
    # Crear un contexto formal
    context = FormalContext()
    
    # Añadir objetos (animales)
    context.add_object("perro")
    context.add_object("gato")
    context.add_object("pez")
    context.add_object("canario")
    
    # Añadir atributos
    context.add_attribute("mamífero")
    context.add_attribute("vuela")
    context.add_attribute("nada")
    context.add_attribute("mascota")
    
    # Añadir incidencias
    context.add_incidence("perro", "mamífero")
    context.add_incidence("perro", "mascota")
    
    context.add_incidence("gato", "mamífero")
    context.add_incidence("gato", "mascota")
    
    context.add_incidence("pez", "nada")
    context.add_incidence("pez", "mascota")
    
    context.add_incidence("canario", "vuela")
    context.add_incidence("canario", "mascota")
    
    print("\nContexto formal creado:")
    print(f"  Objetos: {len(context.objects)}")
    print(f"  Atributos: {len(context.attributes)}")
    
    # Construir retículo
    builder = LatticeBuilder(context)
    concepts = builder.build_lattice()
    
    print(f"\nRetículo construido:")
    print(f"  Conceptos: {len(concepts)}")
    
    # Convertir a álgebra de Heyting
    algebra = lattice_to_heyting(builder, "H_Animales")
    
    print(f"\nÁlgebra de Heyting:")
    print(f"  Elementos: {len(algebra.elements)}")
    print(f"  ⊥ = {algebra.bottom}")
    print(f"  ⊤ = {algebra.top}")
    
    print(f"\nValidación: {algebra.is_valid()}")
    
    # Mostrar algunos conceptos
    print(f"\nAlgunos conceptos:")
    for i, (extent, intent) in enumerate(concepts[:5]):
        print(f"  C{i}: extent={set(extent)}, intent={set(intent)}")


def example_3_logic_operations():
    """Ejemplo 3: Operaciones lógicas intuicionistas."""
    print("\n" + "="*70)
    print("EJEMPLO 3: OPERACIONES LÓGICAS INTUICIONISTAS")
    print("="*70)
    
    # Crear un álgebra simple
    base = {1, 2}
    algebra = create_power_set_algebra(base, "P({1,2})")
    
    # Generar tabla de verdad
    table = heyting_to_logic_table(algebra)
    print(table)
    
    print("\n" + "="*70)
    print("OBSERVACIONES SOBRE LÓGICA INTUICIONISTA")
    print("="*70)
    
    print("\n1. NEGACIÓN CONSTRUCTIVA:")
    print("   En lógica clásica: ¬¬a = a (doble negación)")
    print("   En lógica intuicionista: ¬¬a ≥ a (pero no necesariamente igual)")
    
    print("\n2. TERCIO EXCLUSO:")
    print("   En lógica clásica: a ∨ ¬a = ⊤ (siempre)")
    print("   En lógica intuicionista: a ∨ ¬a puede no ser ⊤")
    
    print("\n3. IMPLICACIÓN:")
    print("   La implicación a → b es el mayor c tal que a ∧ c ≤ b")
    print("   Esto captura la idea constructiva de \"cómo transformar a en b\"")


def main():
    """Función principal."""
    print("\n" + "="*70)
    print("ÁLGEBRA DE HEYTING EN LATTICEWEAVER")
    print("Lógica Intuicionista para Análisis de CSP")
    print("="*70)
    
    example_1_power_set()
    example_2_lattice_to_heyting()
    example_3_logic_operations()
    
    print("\n" + "="*70)
    print("CONCLUSIÓN")
    print("="*70)
    
    print("\nEl álgebra de Heyting proporciona a LatticeWeaver:")
    print("  1. Una semántica lógica para el retículo de conceptos")
    print("  2. Operaciones lógicas constructivas sobre conceptos")
    print("  3. Una base formal para razonamiento intuicionista")
    print("  4. Conexión con teoría de tipos y HoTT (futuro)")
    
    print("\nEsto permite razonar sobre el espacio de soluciones de un CSP")
    print("de manera lógicamente rigurosa y constructiva.")
    print()


if __name__ == '__main__':
    main()

