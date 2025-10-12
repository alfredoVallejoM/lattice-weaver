"""
Ejemplo de Uso: Sintaxis Cúbica (HoTT) en LatticeWeaver

Este ejemplo demuestra cómo usar la sintaxis cúbica para construir
y manipular términos y tipos de la Teoría de Tipos Homotópica.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
import os

# Añadir el directorio padre al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lattice_weaver.formal import *


def example_1_basic_types():
    """Ejemplo 1: Tipos básicos."""
    print("="*70)
    print("EJEMPLO 1: TIPOS BÁSICOS EN HOTT")
    print("="*70)
    
    # Universo de tipos
    type0 = Universe(0)
    print(f"\nUniverso: {type0}")
    
    # Variables de tipo
    a = TypeVar("A")
    b = TypeVar("B")
    print(f"Variables de tipo: {a}, {b}")
    
    # Tipo función A → B
    func_type = arrow_type(a, b)
    print(f"\nTipo función: {func_type}")
    
    # Tipo producto A × B
    prod_type = product_type(a, b)
    print(f"Tipo producto: {prod_type}")
    
    # Tipo dependiente Π(x : A). B
    dep_func = PiType("x", a, b)
    print(f"Tipo Pi dependiente: {dep_func}")
    
    # Tipo de caminos (igualdad)
    x = Var("x")
    y = Var("y")
    path_type = identity_type(a, x, y)
    print(f"Tipo de caminos: {path_type}")


def example_2_lambda_calculus():
    """Ejemplo 2: Cálculo lambda."""
    print("\n" + "="*70)
    print("EJEMPLO 2: CÁLCULO LAMBDA")
    print("="*70)
    
    # Función identidad: λ(x : A). x
    a = TypeVar("A")
    id_func = identity_function(a)
    print(f"\nFunción identidad: {id_func}")
    
    # Aplicar a un argumento
    arg = Var("y")
    app = App(id_func, arg)
    print(f"Aplicación: {app}")
    
    # Beta-reducir
    result = beta_reduce(app)
    print(f"Después de beta-reducción: {result}")
    
    # Función constante: λ(x : A). c
    b = TypeVar("B")
    c = Var("c")
    const = constant_function(a, b, c)
    print(f"\nFunción constante: {const}")
    
    # Aplicar y reducir
    app2 = App(const, arg)
    result2 = normalize(app2)
    print(f"Aplicación y normalización: {app2} → {result2}")


def example_3_pairs_and_projections():
    """Ejemplo 3: Pares y proyecciones."""
    print("\n" + "="*70)
    print("EJEMPLO 3: PARES Y PROYECCIONES")
    print("="*70)
    
    # Crear un par
    x = Var("x")
    y = Var("y")
    pair = Pair(x, y)
    print(f"\nPar: {pair}")
    
    # Proyecciones
    fst = Fst(pair)
    snd = Snd(pair)
    print(f"Primera proyección: {fst}")
    print(f"Segunda proyección: {snd}")
    
    # Reducir
    fst_reduced = beta_reduce(fst)
    snd_reduced = beta_reduce(snd)
    print(f"\nDespués de reducción:")
    print(f"  fst {pair} → {fst_reduced}")
    print(f"  snd {pair} → {snd_reduced}")
    
    # Función swap
    a = TypeVar("A")
    b = TypeVar("B")
    swap = swap_pair(a, b)
    print(f"\nFunción swap: {swap}")
    
    # Aplicar swap
    app = App(swap, pair)
    result = normalize(app)
    print(f"swap {pair} → {result}")


def example_4_paths_and_equality():
    """Ejemplo 4: Caminos e igualdad."""
    print("\n" + "="*70)
    print("EJEMPLO 4: CAMINOS E IGUALDAD (HOTT)")
    print("="*70)
    
    # Reflexividad: el camino trivial de x a x
    a = TypeVar("A")
    x = Var("x")
    refl_x = Refl(x)
    print(f"\nReflexividad: {refl_x}")
    print(f"Tipo: Path {a} {x} {x}")
    
    # Abstracción de camino: <i> t
    path = PathAbs("i", x)
    print(f"\nAbstracción de camino: {path}")
    
    # Aplicación de camino: p @ 0
    app0 = PathApp(refl_x, "0")
    app1 = PathApp(refl_x, "1")
    print(f"\nAplicación de camino:")
    print(f"  {refl_x} @ 0 = {app0}")
    print(f"  {refl_x} @ 1 = {app1}")
    
    # Reducir
    result0 = beta_reduce(app0)
    result1 = beta_reduce(app1)
    print(f"\nDespués de reducción:")
    print(f"  {app0} → {result0}")
    print(f"  {app1} → {result1}")
    
    print("\n" + "-"*70)
    print("OBSERVACIÓN: En HoTT, la igualdad es un tipo de caminos.")
    print("Esto permite razonar sobre igualdades de manera constructiva.")
    print("-"*70)


def example_5_dependent_types():
    """Ejemplo 5: Tipos dependientes."""
    print("\n" + "="*70)
    print("EJEMPLO 5: TIPOS DEPENDIENTES")
    print("="*70)
    
    # Tipo Pi dependiente: Π(n : Nat). Vec A n
    # (Vectores de longitud n)
    nat = TypeVar("Nat")
    a = TypeVar("A")
    vec_n = TypeVar("Vec_n")  # Simplificación: Vec A n
    
    vec_type = PiType("n", nat, vec_n)
    print(f"\nTipo de vectores: {vec_type}")
    
    # Tipo Sigma dependiente: Σ(n : Nat). Vec A n
    # (Par de un número y un vector de esa longitud)
    dep_pair_type = SigmaType("n", nat, vec_n)
    print(f"Tipo de pares dependientes: {dep_pair_type}")
    
    # Función polimórfica: Π(A : Type). A → A
    type0 = Universe(0)
    poly_id = PiType("A", type0, arrow_type(TypeVar("A"), TypeVar("A")))
    print(f"\nFunción polimórfica identidad: {poly_id}")
    
    print("\n" + "-"*70)
    print("OBSERVACIÓN: Los tipos dependientes permiten que tipos")
    print("dependan de valores, habilitando especificaciones precisas.")
    print("-"*70)


def example_6_context_and_typing():
    """Ejemplo 6: Contextos y juicios de tipado."""
    print("\n" + "="*70)
    print("EJEMPLO 6: CONTEXTOS Y JUICIOS DE TIPADO")
    print("="*70)
    
    # Crear contexto vacío
    ctx = Context()
    print(f"\nContexto vacío: {ctx}")
    
    # Extender con variables
    a = TypeVar("A")
    b = TypeVar("B")
    
    ctx = ctx.extend("x", a)
    print(f"Después de añadir x : A: {ctx}")
    
    ctx = ctx.extend("y", b)
    print(f"Después de añadir y : B: {ctx}")
    
    # Buscar tipos
    type_x = ctx.lookup("x")
    type_y = ctx.lookup("y")
    type_z = ctx.lookup("z")
    
    print(f"\nBúsqueda de tipos:")
    print(f"  x : {type_x}")
    print(f"  y : {type_y}")
    print(f"  z : {type_z}")
    
    # Sombreado de variables
    c = TypeVar("C")
    ctx = ctx.extend("x", c)
    print(f"\nDespués de sombrear x con tipo C: {ctx}")
    print(f"  x : {ctx.lookup('x')}")
    
    print("\n" + "-"*70)
    print("OBSERVACIÓN: Los contextos mantienen las asunciones sobre")
    print("los tipos de las variables, permitiendo verificación de tipos.")
    print("-"*70)


def main():
    """Función principal."""
    print("\n" + "="*70)
    print("SINTAXIS CÚBICA EN LATTICEWEAVER")
    print("Abstract Syntax Tree para Teoría de Tipos Homotópica")
    print("="*70)
    
    example_1_basic_types()
    example_2_lambda_calculus()
    example_3_pairs_and_projections()
    example_4_paths_and_equality()
    example_5_dependent_types()
    example_6_context_and_typing()
    
    print("\n" + "="*70)
    print("CONCLUSIÓN")
    print("="*70)
    
    print("\nLa sintaxis cúbica proporciona a LatticeWeaver:")
    print("  1. Un lenguaje formal para expresar tipos y términos de HoTT")
    print("  2. Operaciones de reducción y normalización")
    print("  3. Soporte para tipos dependientes y caminos")
    print("  4. Base para verificación de tipos y búsqueda de pruebas")
    
    print("\nEsto permite formalizar el razonamiento sobre CSP en un")
    print("sistema de tipos con fundamentos homotópicos.")
    print()


if __name__ == '__main__':
    main()

