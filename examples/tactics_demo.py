#!/usr/bin/env python3
# tactics_demo.py

"""
Demostración de Tácticas Avanzadas de Búsqueda de Pruebas

Ejemplos de uso de las tácticas implementadas para búsqueda automática
de pruebas en el sistema formal HoTT.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_weaver.formal import (
    CubicalEngine, ProofGoal,
    Context, TypeVar, PiType, SigmaType, PathType,
    Var, Universe
)


def demo_identity_function():
    """Demostración: Probar función identidad usando tácticas."""
    print("=" * 70)
    print("DEMO 1: Función Identidad (A → A)")
    print("=" * 70)
    
    engine = CubicalEngine()
    
    # Contexto
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    
    # Meta: A → A
    goal = ProofGoal(
        PiType("x", TypeVar("A"), TypeVar("A")),
        ctx,
        "identity"
    )
    
    print(f"\nMeta: {goal.type_}")
    print("Estrategia: intro + assumption")
    
    # Buscar con tácticas
    proof = engine.search_proof_with_tactics(goal, max_depth=3)
    
    if proof:
        print(f"\n✅ Prueba encontrada: {proof.term}")
        print(f"Tipo: {proof.type_}")
    else:
        print("\n❌ No se encontró prueba")
    
    print("\n" + "=" * 70)


def demo_constant_function():
    """Demostración: Probar función constante."""
    print("\n" + "=" * 70)
    print("DEMO 2: Función Constante (A → B → A)")
    print("=" * 70)
    
    engine = CubicalEngine()
    
    # Contexto
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("B", Universe(0))
    
    # Meta: A → B → A
    goal = ProofGoal(
        PiType("x", TypeVar("A"), 
               PiType("y", TypeVar("B"), TypeVar("A"))),
        ctx,
        "const"
    )
    
    print(f"\nMeta: {goal.type_}")
    print("Estrategia: intro + intro + assumption")
    
    # Buscar con tácticas
    proof = engine.search_proof_with_tactics(goal, max_depth=4)
    
    if proof:
        print(f"\n✅ Prueba encontrada: {proof.term}")
    else:
        print("\n❌ No se encontró prueba")
    
    print("\n" + "=" * 70)


def demo_pair_construction():
    """Demostración: Construir par usando split."""
    print("\n" + "=" * 70)
    print("DEMO 3: Construcción de Par (A × B)")
    print("=" * 70)
    
    engine = CubicalEngine()
    
    # Contexto con elementos
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("B", Universe(0))
    ctx = ctx.extend("a", TypeVar("A"))
    ctx = ctx.extend("b", TypeVar("B"))
    
    # Meta: A × B
    goal = ProofGoal(
        SigmaType("_", TypeVar("A"), TypeVar("B")),
        ctx,
        "pair"
    )
    
    print(f"\nMeta: {goal.type_}")
    print(f"Contexto: a : A, b : B")
    print("Estrategia: split + assumption × 2")
    
    # Buscar con tácticas
    proof = engine.search_proof_with_tactics(goal, max_depth=3)
    
    if proof:
        print(f"\n✅ Prueba encontrada: {proof.term}")
    else:
        print("\n❌ No se encontró prueba")
    
    print("\n" + "=" * 70)


def demo_reflexivity():
    """Demostración: Reflexividad de la igualdad."""
    print("\n" + "=" * 70)
    print("DEMO 4: Reflexividad (x = x)")
    print("=" * 70)
    
    engine = CubicalEngine()
    
    # Contexto
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("x", TypeVar("A"))
    
    # Meta: x = x
    goal = ProofGoal(
        PathType(TypeVar("A"), Var("x"), Var("x")),
        ctx,
        "refl_x"
    )
    
    print(f"\nMeta: {goal.type_}")
    print("Estrategia: reflexivity")
    
    # Buscar con tácticas
    proof = engine.search_proof_with_tactics(goal, max_depth=1)
    
    if proof:
        print(f"\n✅ Prueba encontrada: {proof.term}")
    else:
        print("\n❌ No se encontró prueba")
    
    print("\n" + "=" * 70)


def demo_ex_falso():
    """Demostración: Ex falso quodlibet (de la falsedad se sigue cualquier cosa)."""
    print("\n" + "=" * 70)
    print("DEMO 5: Ex Falso Quodlibet (⊥ → A)")
    print("=" * 70)
    
    engine = CubicalEngine()
    
    # Contexto con falsedad
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("absurd_proof", TypeVar("Empty"))
    
    # Meta: A (cualquier cosa)
    goal = ProofGoal(TypeVar("A"), ctx, "from_false")
    
    print(f"\nMeta: {goal.type_}")
    print(f"Contexto: absurd_proof : ⊥")
    print("Estrategia: contradiction")
    
    # Buscar con tácticas
    proof = engine.search_proof_with_tactics(goal, max_depth=1)
    
    if proof:
        print(f"\n✅ Prueba encontrada: {proof.term}")
    else:
        print("\n❌ No se encontró prueba")
    
    print("\n" + "=" * 70)


def demo_composition():
    """Demostración: Composición de funciones."""
    print("\n" + "=" * 70)
    print("DEMO 6: Composición de Funciones")
    print("=" * 70)
    
    engine = CubicalEngine()
    
    # Contexto
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("B", Universe(0))
    ctx = ctx.extend("C", Universe(0))
    ctx = ctx.extend("f", PiType("x", TypeVar("A"), TypeVar("B")))
    ctx = ctx.extend("g", PiType("y", TypeVar("B"), TypeVar("C")))
    
    # Meta: A → C
    goal = ProofGoal(
        PiType("x", TypeVar("A"), TypeVar("C")),
        ctx,
        "compose"
    )
    
    print(f"\nMeta: {goal.type_}")
    print(f"Contexto: f : A → B, g : B → C")
    print("Estrategia: intro + aplicaciones")
    
    # Buscar con tácticas
    proof = engine.search_proof_with_tactics(goal, max_depth=5)
    
    if proof:
        print(f"\n✅ Prueba encontrada: {proof.term}")
    else:
        print("\n❌ No se encontró prueba (esperado - requiere aplicación)")
    
    print("\n" + "=" * 70)


def demo_tactic_statistics():
    """Demostración: Estadísticas de uso de tácticas."""
    print("\n" + "=" * 70)
    print("DEMO 7: Estadísticas de Tácticas")
    print("=" * 70)
    
    engine = CubicalEngine()
    tactics = engine.tactics
    
    # Resetear estadísticas
    tactics.reset_statistics()
    
    print("\nEjecutando varias pruebas...")
    
    # Prueba 1: Identidad
    ctx1 = Context().extend("A", Universe(0))
    goal1 = ProofGoal(PiType("x", TypeVar("A"), TypeVar("A")), ctx1, "id")
    engine.search_proof_with_tactics(goal1)
    
    # Prueba 2: Par
    ctx2 = Context().extend("A", Universe(0)).extend("B", Universe(0))
    ctx2 = ctx2.extend("a", TypeVar("A")).extend("b", TypeVar("B"))
    goal2 = ProofGoal(SigmaType("_", TypeVar("A"), TypeVar("B")), ctx2, "pair")
    engine.search_proof_with_tactics(goal2)
    
    # Prueba 3: Reflexividad
    ctx3 = Context().extend("A", Universe(0)).extend("x", TypeVar("A"))
    goal3 = ProofGoal(PathType(TypeVar("A"), Var("x"), Var("x")), ctx3, "refl")
    engine.search_proof_with_tactics(goal3)
    
    # Obtener estadísticas
    stats = tactics.get_statistics()
    
    print(f"\nEstadísticas de tácticas utilizadas:")
    for tactic, count in sorted(stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {tactic:15s}: {count:3d} veces")
    
    total = sum(stats.values())
    print(f"\nTotal de aplicaciones: {total}")
    
    print("\n" + "=" * 70)


def demo_auto_tactic_power():
    """Demostración: Poder de la táctica auto."""
    print("\n" + "=" * 70)
    print("DEMO 8: Poder de la Táctica Auto")
    print("=" * 70)
    
    engine = CubicalEngine()
    
    test_cases = [
        ("Identidad", PiType("x", TypeVar("A"), TypeVar("A"))),
        ("Par simple", SigmaType("_", TypeVar("A"), TypeVar("B"))),
        ("Reflexividad", PathType(TypeVar("A"), Var("x"), Var("x"))),
    ]
    
    print("\nProbando táctica auto en diferentes metas:\n")
    
    for name, goal_type in test_cases:
        # Preparar contexto apropiado
        ctx = Context().extend("A", Universe(0))
        
        if "Par" in name:
            ctx = ctx.extend("B", Universe(0))
            ctx = ctx.extend("a", TypeVar("A"))
            ctx = ctx.extend("b", TypeVar("B"))
        elif "Reflexividad" in name:
            ctx = ctx.extend("x", TypeVar("A"))
        
        goal = ProofGoal(goal_type, ctx, name.lower().replace(" ", "_"))
        
        # Intentar con auto
        result = engine.tactics.apply_auto(goal, max_depth=3)
        
        status = "✅" if result.success else "❌"
        print(f"{status} {name:20s}: {result.message}")
        if result.proof:
            print(f"   Prueba: {result.proof.term}")
    
    print("\n" + "=" * 70)


def main():
    """Ejecuta las demostraciones."""
    print("\nDemostraciones de Tácticas Avanzadas")
    print("LatticeWeaver v4\n")
    
    try:
        demo_identity_function()
        demo_constant_function()
        demo_pair_construction()
        demo_reflexivity()
        demo_ex_falso()
        demo_composition()
        demo_tactic_statistics()
        demo_auto_tactic_power()
        
        print("\n✅ Todas las demostraciones completadas\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

