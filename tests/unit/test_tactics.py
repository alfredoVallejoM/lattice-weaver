#!/usr/bin/env python3
# test_tactics.py

"""
Tests para Tácticas Avanzadas de Búsqueda de Pruebas

Valida las tácticas implementadas: reflexivity, assumption, intro, split,
contradiction, rewrite, y auto.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_weaver.formal import (
    CubicalEngine, ProofGoal, ProofTerm,
    TacticEngine, create_tactic_engine,
    Context, TypeVar, PiType, SigmaType, PathType,
    Var, Lambda, Pair, Refl, Universe
)


def test_reflexivity_tactic():
    """Test: Táctica de reflexividad."""
    print("=" * 60)
    print("TEST 1: Táctica de Reflexividad")
    print("=" * 60)
    
    engine = CubicalEngine()
    tactics = engine.tactics
    
    # Crear contexto
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("x", TypeVar("A"))
    
    # Meta: x = x
    goal = ProofGoal(
        PathType(TypeVar("A"), Var("x"), Var("x")),
        ctx,
        "refl_goal"
    )
    
    print(f"\nMeta: {goal.type_}")
    
    # Aplicar táctica
    result = tactics.apply_reflexivity(goal)
    
    print(f"Resultado: {result.message}")
    print(f"Éxito: {result.success}")
    
    if result.proof:
        print(f"Prueba: {result.proof.term}")
    
    assert result.success
    assert result.proof is not None
    
    print("\n✅ Test pasado")
    return True


def test_assumption_tactic():
    """Test: Táctica de asunción."""
    print("\n" + "=" * 60)
    print("TEST 2: Táctica de Asunción")
    print("=" * 60)
    
    engine = CubicalEngine()
    tactics = engine.tactics
    
    # Crear contexto con una asunción
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("h", TypeVar("A"))  # Asunción: h : A
    
    # Meta: A (que ya está en el contexto como h)
    goal = ProofGoal(TypeVar("A"), ctx, "assumption_goal")
    
    print(f"\nContexto: {[b.var for b in ctx.bindings]}")
    print(f"Meta: {goal.type_}")
    
    # Aplicar táctica
    result = tactics.apply_assumption(goal)
    
    print(f"Resultado: {result.message}")
    print(f"Éxito: {result.success}")
    
    if result.proof:
        print(f"Prueba: {result.proof.term}")
    
    assert result.success
    assert result.proof is not None
    
    print("\n✅ Test pasado")
    return True


def test_intro_tactic():
    """Test: Táctica de introducción."""
    print("\n" + "=" * 60)
    print("TEST 3: Táctica de Introducción")
    print("=" * 60)
    
    engine = CubicalEngine()
    tactics = engine.tactics
    
    # Crear contexto
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("B", Universe(0))
    
    # Meta: A → B
    goal = ProofGoal(
        PiType("x", TypeVar("A"), TypeVar("B")),
        ctx,
        "intro_goal"
    )
    
    print(f"\nMeta: {goal.type_}")
    
    # Aplicar táctica
    result = tactics.apply_intro(goal)
    
    print(f"Resultado: {result.message}")
    print(f"Éxito: {result.success}")
    print(f"Submetas generadas: {len(result.subgoals)}")
    
    if result.subgoals:
        print(f"Submeta: {result.subgoals[0].type_}")
        print(f"Nuevo contexto: {[b.var for b in result.subgoals[0].context.bindings]}")
    
    assert result.success
    assert len(result.subgoals) == 1
    
    print("\n✅ Test pasado")
    return True


def test_split_tactic():
    """Test: Táctica de división (split)."""
    print("\n" + "=" * 60)
    print("TEST 4: Táctica de División (Split)")
    print("=" * 60)
    
    engine = CubicalEngine()
    tactics = engine.tactics
    
    # Crear contexto
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("B", Universe(0))
    
    # Meta: A × B (producto no dependiente)
    goal = ProofGoal(
        SigmaType("_", TypeVar("A"), TypeVar("B")),
        ctx,
        "split_goal"
    )
    
    print(f"\nMeta: {goal.type_}")
    
    # Aplicar táctica
    result = tactics.apply_split(goal)
    
    print(f"Resultado: {result.message}")
    print(f"Éxito: {result.success}")
    print(f"Submetas generadas: {len(result.subgoals)}")
    
    if result.subgoals:
        print(f"Submeta 1: {result.subgoals[0].type_}")
        print(f"Submeta 2: {result.subgoals[1].type_}")
    
    assert result.success
    assert len(result.subgoals) == 2
    
    print("\n✅ Test pasado")
    return True


def test_contradiction_tactic():
    """Test: Táctica de contradicción."""
    print("\n" + "=" * 60)
    print("TEST 5: Táctica de Contradicción")
    print("=" * 60)
    
    engine = CubicalEngine()
    tactics = engine.tactics
    
    # Crear contexto con falsedad
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("absurd_hyp", TypeVar("Empty"))  # ⊥
    
    # Meta: A (cualquier cosa)
    goal = ProofGoal(TypeVar("A"), ctx, "contradiction_goal")
    
    print(f"\nContexto: {[(b.var, b.type_) for b in ctx.bindings]}")
    print(f"Meta: {goal.type_}")
    
    # Aplicar táctica
    result = tactics.apply_contradiction(goal)
    
    print(f"Resultado: {result.message}")
    print(f"Éxito: {result.success}")
    
    if result.proof:
        print(f"Prueba: {result.proof.term}")
    
    assert result.success
    assert result.proof is not None
    
    print("\n✅ Test pasado")
    return True


def test_auto_tactic_simple():
    """Test: Táctica automática (caso simple)."""
    print("\n" + "=" * 60)
    print("TEST 6: Táctica Automática (Simple)")
    print("=" * 60)
    
    engine = CubicalEngine()
    tactics = engine.tactics
    
    # Crear contexto
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("x", TypeVar("A"))
    
    # Meta: x = x (debería resolverse con reflexivity)
    goal = ProofGoal(
        PathType(TypeVar("A"), Var("x"), Var("x")),
        ctx,
        "auto_simple"
    )
    
    print(f"\nMeta: {goal.type_}")
    
    # Aplicar táctica auto
    result = tactics.apply_auto(goal, max_depth=3)
    
    print(f"Resultado: {result.message}")
    print(f"Éxito: {result.success}")
    
    if result.proof:
        print(f"Prueba: {result.proof.term}")
    
    assert result.success
    assert result.proof is not None
    
    print("\n✅ Test pasado")
    return True


def test_auto_tactic_intro():
    """Test: Táctica automática con introducción."""
    print("\n" + "=" * 60)
    print("TEST 7: Táctica Automática (con Intro)")
    print("=" * 60)
    
    engine = CubicalEngine()
    tactics = engine.tactics
    
    # Crear contexto
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    
    # Meta: A → A (identidad)
    goal = ProofGoal(
        PiType("x", TypeVar("A"), TypeVar("A")),
        ctx,
        "auto_intro"
    )
    
    print(f"\nMeta: {goal.type_}")
    
    # Aplicar táctica auto
    result = tactics.apply_auto(goal, max_depth=3)
    
    print(f"Resultado: {result.message}")
    print(f"Éxito: {result.success}")
    
    if result.proof:
        print(f"Prueba: {result.proof.term}")
    
    assert result.success
    assert result.proof is not None
    
    print("\n✅ Test pasado")
    return True


def test_auto_tactic_product():
    """Test: Táctica automática con producto."""
    print("\n" + "=" * 60)
    print("TEST 8: Táctica Automática (con Split)")
    print("=" * 60)
    
    engine = CubicalEngine()
    tactics = engine.tactics
    
    # Crear contexto con asunciones
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("B", Universe(0))
    ctx = ctx.extend("a", TypeVar("A"))
    ctx = ctx.extend("b", TypeVar("B"))
    
    # Meta: A × B
    goal = ProofGoal(
        SigmaType("_", TypeVar("A"), TypeVar("B")),
        ctx,
        "auto_product"
    )
    
    print(f"\nMeta: {goal.type_}")
    print(f"Contexto: {[b.var for b in ctx.bindings]}")
    
    # Aplicar táctica auto
    result = tactics.apply_auto(goal, max_depth=3)
    
    print(f"Resultado: {result.message}")
    print(f"Éxito: {result.success}")
    
    if result.proof:
        print(f"Prueba: {result.proof.term}")
    
    assert result.success
    assert result.proof is not None
    
    print("\n✅ Test pasado")
    return True


def test_tactic_statistics():
    """Test: Estadísticas de tácticas."""
    print("\n" + "=" * 60)
    print("TEST 9: Estadísticas de Tácticas")
    print("=" * 60)
    
    engine = CubicalEngine()
    tactics = engine.tactics
    
    # Resetear estadísticas
    tactics.reset_statistics()
    
    # Aplicar varias tácticas
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    ctx = ctx.extend("x", TypeVar("A"))
    
    # Reflexivity
    goal1 = ProofGoal(PathType(TypeVar("A"), Var("x"), Var("x")), ctx, "g1")
    tactics.apply_reflexivity(goal1)
    
    # Assumption
    goal2 = ProofGoal(TypeVar("A"), ctx, "g2")
    tactics.apply_assumption(goal2)
    
    # Auto
    goal3 = ProofGoal(PathType(TypeVar("A"), Var("x"), Var("x")), ctx, "g3")
    tactics.apply_auto(goal3)
    
    # Obtener estadísticas
    stats = tactics.get_statistics()
    
    print(f"\nEstadísticas de tácticas:")
    for tactic, count in stats.items():
        if count > 0:
            print(f"  {tactic}: {count}")
    
    assert stats['reflexivity'] >= 1
    assert stats['assumption'] >= 1
    assert stats['auto'] >= 1
    
    print("\n✅ Test pasado")
    return True


def test_search_with_tactics():
    """Test: Búsqueda con tácticas integrada."""
    print("\n" + "=" * 60)
    print("TEST 10: Búsqueda con Tácticas Integrada")
    print("=" * 60)
    
    engine = CubicalEngine()
    
    # Crear contexto
    ctx = Context()
    ctx = ctx.extend("A", Universe(0))
    
    # Meta: A → A
    goal = ProofGoal(
        PiType("x", TypeVar("A"), TypeVar("A")),
        ctx,
        "identity"
    )
    
    print(f"\nMeta: {goal.type_}")
    
    # Buscar con tácticas
    proof = engine.search_proof_with_tactics(goal, max_depth=3)
    
    if proof:
        print(f"Prueba encontrada: {proof.term}")
        assert proof is not None
    else:
        print("No se encontró prueba")
        assert False, "Debería encontrar prueba para A → A"
    
    print("\n✅ Test pasado")
    return True


def main():
    """Ejecuta todos los tests."""
    print("Tests de Tácticas Avanzadas de Búsqueda de Pruebas")
    print("LatticeWeaver v4\n")
    
    try:
        test_reflexivity_tactic()
        test_assumption_tactic()
        test_intro_tactic()
        test_split_tactic()
        test_contradiction_tactic()
        test_auto_tactic_simple()
        test_auto_tactic_intro()
        test_auto_tactic_product()
        test_tactic_statistics()
        test_search_with_tactics()
        
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

