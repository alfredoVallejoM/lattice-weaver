"""
Ejemplo Completo: Motor Cúbico (CubicalEngine) en LatticeWeaver

Este ejemplo demuestra la integración completa del sistema formal de HoTT
con el motor de resolución de CSP.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
import os

# Añadir el directorio padre al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lattice_weaver.formal import *


def print_separator(title=""):
    """Imprime un separador visual."""
    print("\n" + "="*70)
    if title:
        print(title)
        print("="*70)


def example_1_proof_construction():
    """Ejemplo 1: Construcción de pruebas."""
    print_separator("EJEMPLO 1: CONSTRUCCIÓN DE PRUEBAS")
    
    # Crear motor
    engine = create_engine()
    
    # Crear contexto: A : Type, x : A
    ctx = Context()
    a = TypeVar("A")
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("x", a)
    
    x = Var("x")
    
    # Probar reflexividad: x = x
    print("\nProbando reflexividad: x = x")
    refl_proof = engine.prove_reflexivity(ctx, x)
    print(f"Prueba: {refl_proof}")
    
    # Verificar la prueba
    is_valid = engine.verify_proof(refl_proof)
    print(f"¿Es válida? {is_valid}")
    
    # Probar simetría: si x = x, entonces x = x
    print("\nProbando simetría")
    sym_proof = engine.prove_symmetry(ctx, refl_proof)
    print(f"Prueba: {sym_proof}")
    print(f"¿Es válida? {engine.verify_proof(sym_proof)}")


def example_2_automatic_proof_search():
    """Ejemplo 2: Búsqueda automática de pruebas."""
    print_separator("EJEMPLO 2: BÚSQUEDA AUTOMÁTICA DE PRUEBAS")
    
    engine = create_engine()
    
    # Contexto: A : Type, x : A
    ctx = Context()
    a = TypeVar("A")
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("x", a)
    
    x = Var("x")
    
    # Meta: probar que x = x
    goal_type = identity_type(a, x, x)
    goal = ProofGoal(goal_type, ctx, "reflexivity_goal")
    
    print(f"\nMeta: {goal}")
    print("Buscando prueba automáticamente...")
    
    proof = engine.search_proof(goal, max_depth=3)
    
    if proof:
        print(f"✓ Prueba encontrada: {proof}")
        print(f"✓ Verificación: {engine.verify_proof(proof)}")
    else:
        print("✗ No se encontró prueba")


def example_3_function_proofs():
    """Ejemplo 3: Pruebas con funciones."""
    print_separator("EJEMPLO 3: PRUEBAS CON FUNCIONES")
    
    engine = create_engine()
    
    # Contexto: A, B : Type
    ctx = Context()
    a = TypeVar("A")
    b = TypeVar("B")
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("B", Universe(0))
    
    # Meta: probar A → A (función identidad)
    goal_type = arrow_type(a, a)
    goal = ProofGoal(goal_type, ctx, "identity_function")
    
    print(f"\nMeta: probar {goal_type}")
    print("Buscando prueba...")
    
    proof = engine.search_proof(goal, max_depth=5)
    
    if proof:
        print(f"✓ Prueba encontrada: {proof.term}")
        print(f"✓ Tipo: {proof.type_}")
        print(f"✓ Verificación: {engine.verify_proof(proof)}")
    else:
        print("✗ No se encontró prueba")


def example_4_csp_integration():
    """Ejemplo 4: Integración con CSP."""
    print_separator("EJEMPLO 4: INTEGRACIÓN CSP-HOTT")
    
    # Crear puente CSP-HoTT
    bridge = create_bridge()
    
    # Crear problema CSP simple
    problem = simple_csp_example()
    print(f"\nProblema CSP:")
    print(f"  Variables: {problem.variables}")
    print(f"  Dominios: {problem.domains}")
    print(f"  Restricciones: {len(problem.constraints)}")
    
    # Convertir a contexto HoTT
    ctx = bridge.csp_to_context(problem)
    print(f"\nContexto HoTT: {ctx}")
    
    # Crear solución
    solution = simple_solution_example()
    print(f"\nSolución CSP: {solution.assignment}")
    
    # Convertir solución a prueba
    proof = bridge.solution_to_proof(solution, problem)
    
    if proof:
        print(f"✓ Solución convertida a prueba")
        print(f"  Término: {proof.term}")
        print(f"  Tipo: {proof.type_}")
    else:
        print("✗ No se pudo convertir la solución")


def example_5_proof_equality():
    """Ejemplo 5: Igualdad de pruebas."""
    print_separator("EJEMPLO 5: IGUALDAD DE PRUEBAS")
    
    engine = create_engine()
    
    # Contexto
    ctx = Context()
    a = TypeVar("A")
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("x", a)
    
    x = Var("x")
    
    # Dos pruebas de x = x
    proof1 = engine.prove_reflexivity(ctx, x)
    proof2 = engine.prove_reflexivity(ctx, x)
    
    print("\nPrueba 1:", proof1.term)
    print("Prueba 2:", proof2.term)
    
    # Verificar igualdad
    are_equal = engine.proofs_equal(proof1, proof2)
    print(f"\n¿Son iguales? {are_equal}")
    
    print("\n" + "-"*70)
    print("OBSERVACIÓN: En HoTT, dos pruebas de la misma proposición")
    print("pueden ser iguales (definitionally equal) o diferentes.")
    print("-"*70)


def example_6_normalization():
    """Ejemplo 6: Normalización de pruebas."""
    print_separator("EJEMPLO 6: NORMALIZACIÓN DE PRUEBAS")
    
    engine = create_engine()
    
    # Contexto: A : Type
    ctx = Context()
    a = TypeVar("A")
    ctx = ctx.extend("A", Universe(0))
    
    # Crear una prueba compleja: (λx. x) ((λy. y) z)
    ctx = ctx.extend("z", a)
    z = Var("z")
    
    id1 = identity_function(a)
    id2 = identity_function(a)
    complex_term = App(id1, App(id2, z))
    
    proof = ProofTerm(complex_term, a, ctx)
    
    print(f"\nPrueba original: {proof.term}")
    
    # Normalizar
    normalized = engine.normalize_proof(proof)
    print(f"Prueba normalizada: {normalized.term}")
    
    print("\n" + "-"*70)
    print("OBSERVACIÓN: La normalización reduce una prueba a su forma")
    print("más simple, aplicando todas las reducciones posibles.")
    print("-"*70)


def main():
    """Función principal."""
    print("\n" + "="*70)
    print("MOTOR CÚBICO (CUBICALENGINE) EN LATTICEWEAVER")
    print("Integración Completa CSP-HoTT")
    print("="*70)
    
    example_1_proof_construction()
    example_2_automatic_proof_search()
    example_3_function_proofs()
    example_4_csp_integration()
    example_5_proof_equality()
    example_6_normalization()
    
    print_separator("CONCLUSIÓN")
    
    print("\nEl CubicalEngine proporciona:")
    print("  1. Construcción manual de pruebas formales")
    print("  2. Búsqueda automática de pruebas simples")
    print("  3. Verificación de correctitud de pruebas")
    print("  4. Integración con problemas CSP")
    print("  5. Normalización y comparación de pruebas")
    
    print("\nEsto completa la implementación del sistema formal de")
    print("LatticeWeaver, dotándolo de fundamentos homotópicos sólidos.")
    print()


if __name__ == '__main__':
    main()

