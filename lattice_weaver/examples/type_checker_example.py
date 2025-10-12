"""
Ejemplo de Uso: Verificador de Tipos para HoTT en LatticeWeaver

Este ejemplo demuestra cómo usar el verificador de tipos para validar
la correctitud de términos y pruebas en la Teoría de Tipos Homotópica.

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


def example_1_basic_type_checking():
    """Ejemplo 1: Verificación básica de tipos."""
    print_separator("EJEMPLO 1: VERIFICACIÓN BÁSICA DE TIPOS")
    
    # Crear contexto
    ctx = Context()
    a = TypeVar("A")
    ctx = ctx.extend("A", Universe(0))
    
    # Función identidad: λ(x : A). x
    id_func = identity_function(a)
    print(f"\nTérmino: {id_func}")
    
    # Inferir tipo
    checker = TypeChecker()
    inferred_type = checker.infer_type(ctx, id_func)
    print(f"Tipo inferido: {inferred_type}")
    
    # Verificar tipo
    expected_type = arrow_type(a, a)
    try:
        checker.check_type(ctx, id_func, expected_type)
        print(f"✓ Verificación exitosa: el término tiene tipo {expected_type}")
    except TypeCheckError as e:
        print(f"✗ Error de tipos: {e}")


def example_2_function_application():
    """Ejemplo 2: Aplicación de funciones."""
    print_separator("EJEMPLO 2: APLICACIÓN DE FUNCIONES")
    
    # Contexto: A : Type, x : A
    ctx = Context()
    a = TypeVar("A")
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("x", a)
    
    # Función: λ(y : A). y
    id_func = identity_function(a)
    
    # Argumento: x
    x = Var("x")
    
    # Aplicación: (λ(y : A). y) x
    app = App(id_func, x)
    print(f"\nTérmino: {app}")
    
    # Inferir tipo
    checker = TypeChecker()
    inferred_type = checker.infer_type(ctx, app)
    print(f"Tipo inferido: {inferred_type}")
    
    # Verificar
    try:
        checker.check_type(ctx, app, a)
        print(f"✓ Verificación exitosa: el término tiene tipo {a}")
    except TypeCheckError as e:
        print(f"✗ Error de tipos: {e}")


def example_3_pairs_and_projections():
    """Ejemplo 3: Pares y proyecciones."""
    print_separator("EJEMPLO 3: PARES Y PROYECCIONES")
    
    # Contexto: A, B : Type, x : A, y : B
    ctx = Context()
    a = TypeVar("A")
    b = TypeVar("B")
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("B", Universe(0))
    ctx = ctx.extend("x", a)
    ctx = ctx.extend("y", b)
    
    # Par: (x, y)
    x = Var("x")
    y = Var("y")
    pair = Pair(x, y)
    print(f"\nPar: {pair}")
    
    checker = TypeChecker()
    pair_type = checker.infer_type(ctx, pair)
    print(f"Tipo del par: {pair_type}")
    
    # Primera proyección: fst (x, y)
    fst_term = Fst(pair)
    fst_type = checker.infer_type(ctx, fst_term)
    print(f"\nPrimera proyección: {fst_term}")
    print(f"Tipo: {fst_type}")
    
    # Segunda proyección: snd (x, y)
    snd_term = Snd(pair)
    snd_type = checker.infer_type(ctx, snd_term)
    print(f"\nSegunda proyección: {snd_term}")
    print(f"Tipo: {snd_type}")


def example_4_equality_and_paths():
    """Ejemplo 4: Igualdad y caminos."""
    print_separator("EJEMPLO 4: IGUALDAD Y CAMINOS (HOTT)")
    
    # Contexto: A : Type, x : A
    ctx = Context()
    a = TypeVar("A")
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("x", a)
    
    # Reflexividad: refl x
    x = Var("x")
    refl_x = Refl(x)
    print(f"\nReflexividad: {refl_x}")
    
    checker = TypeChecker()
    refl_type = checker.infer_type(ctx, refl_x)
    print(f"Tipo: {refl_type}")
    
    # Verificar que refl x : Path A x x
    expected_type = identity_type(a, x, x)
    try:
        checker.check_type(ctx, refl_x, expected_type)
        print(f"✓ Verificación exitosa: refl x : Path A x x")
    except TypeCheckError as e:
        print(f"✗ Error de tipos: {e}")
    
    print("\n" + "-"*70)
    print("OBSERVACIÓN: En HoTT, 'refl x' es la prueba de que x = x.")
    print("Esto es una construcción, no un axioma.")
    print("-"*70)


def example_5_type_errors():
    """Ejemplo 5: Detección de errores de tipos."""
    print_separator("EJEMPLO 5: DETECCIÓN DE ERRORES DE TIPOS")
    
    # Contexto: A, B : Type, x : A
    ctx = Context()
    a = TypeVar("A")
    b = TypeVar("B")
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("B", Universe(0))
    ctx = ctx.extend("x", a)
    
    x = Var("x")
    
    # Intentar verificar que x : B (error!)
    print(f"\nIntentando verificar que {x} : {b}")
    print(f"(Sabemos que x : {a})")
    
    checker = TypeChecker()
    try:
        checker.check_type(ctx, x, b)
        print("✓ Verificación exitosa")
    except TypeCheckError as e:
        print(f"✗ Error de tipos detectado: {e}")
    
    # Variable no en contexto
    z = Var("z")
    print(f"\nIntentando inferir el tipo de {z} (no está en el contexto)")
    
    try:
        checker.infer_type(ctx, z)
        print("✓ Tipo inferido")
    except TypeCheckError as e:
        print(f"✗ Error de tipos detectado: {e}")


def example_6_dependent_types():
    """Ejemplo 6: Tipos dependientes."""
    print_separator("EJEMPLO 6: TIPOS DEPENDIENTES")
    
    # Contexto básico
    ctx = Context()
    ctx = ctx.extend("Nat", Universe(0))
    ctx = ctx.extend("Vec", Universe(0))
    
    nat = TypeVar("Nat")
    vec = TypeVar("Vec")
    
    # Tipo Pi dependiente: Π(n : Nat). Vec
    # (Función que toma un número y devuelve un vector)
    dep_func_type = PiType("n", nat, vec)
    print(f"\nTipo Pi dependiente: {dep_func_type}")
    
    checker = TypeChecker()
    try:
        checker.check_type_formation(ctx, dep_func_type)
        print("✓ El tipo está bien formado")
    except TypeCheckError as e:
        print(f"✗ Error: {e}")
    
    # Tipo Sigma dependiente: Σ(n : Nat). Vec
    # (Par de un número y un vector)
    dep_pair_type = SigmaType("n", nat, vec)
    print(f"\nTipo Sigma dependiente: {dep_pair_type}")
    
    try:
        checker.check_type_formation(ctx, dep_pair_type)
        print("✓ El tipo está bien formado")
    except TypeCheckError as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "-"*70)
    print("OBSERVACIÓN: Los tipos dependientes permiten que el tipo")
    print("del resultado dependa del valor del argumento.")
    print("-"*70)


def example_7_polymorphism():
    """Ejemplo 7: Polimorfismo."""
    print_separator("EJEMPLO 7: POLIMORFISMO")
    
    # Contexto vacío
    ctx = Context()
    
    # Función polimórfica identidad: Π(A : Type). A → A
    type0 = Universe(0)
    a = TypeVar("A")
    poly_id_type = PiType("A", type0, arrow_type(a, a))
    
    print(f"\nTipo de la función identidad polimórfica:")
    print(f"{poly_id_type}")
    
    checker = TypeChecker()
    try:
        checker.check_type_formation(ctx, poly_id_type)
        print("✓ El tipo está bien formado")
    except TypeCheckError as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "-"*70)
    print("OBSERVACIÓN: En teoría de tipos dependientes, el polimorfismo")
    print("se expresa como una función que toma un tipo como argumento.")
    print("-"*70)


def main():
    """Función principal."""
    print("\n" + "="*70)
    print("VERIFICADOR DE TIPOS PARA HOTT EN LATTICEWEAVER")
    print("="*70)
    
    example_1_basic_type_checking()
    example_2_function_application()
    example_3_pairs_and_projections()
    example_4_equality_and_paths()
    example_5_type_errors()
    example_6_dependent_types()
    example_7_polymorphism()
    
    print_separator("CONCLUSIÓN")
    
    print("\nEl verificador de tipos proporciona:")
    print("  1. Inferencia de tipos para términos")
    print("  2. Verificación de tipos contra especificaciones")
    print("  3. Detección de errores de tipos")
    print("  4. Soporte para tipos dependientes")
    print("  5. Validación de formación de tipos")
    
    print("\nEsto garantiza que las construcciones y pruebas en")
    print("LatticeWeaver sean correctas según las reglas de HoTT.")
    print()


if __name__ == '__main__':
    main()

