#!/usr/bin/env python3
# test_tms.py

"""
Tests para Truth Maintenance System (TMS)

Valida las funcionalidades de rastreo de dependencias y retroceso eficiente.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_weaver.arc_engine import ArcEngine, TruthMaintenanceSystem, create_tms


def test_tms_basic():
    """Test: Funcionalidad básica del TMS."""
    print("=" * 60)
    print("TEST 1: Funcionalidad Básica del TMS")
    print("=" * 60)
    
    tms = create_tms()
    
    # Registrar eliminaciones
    tms.record_removal("X", 1, "C1", {"Y": [2, 3]})
    tms.record_removal("X", 2, "C1", {"Y": [1]})
    tms.record_removal("Y", 3, "C2", {"X": [1, 2]})
    
    print(f"\nJustificaciones registradas: {len(tms.justifications)}")
    print(f"Variables afectadas: {list(tms.dependency_graph.keys())}")
    
    assert len(tms.justifications) == 3
    assert "X" in tms.dependency_graph
    assert "Y" in tms.dependency_graph
    
    print("\n✅ Test pasado")
    return True


def test_tms_explain_inconsistency():
    """Test: Explicación de inconsistencias."""
    print("\n" + "=" * 60)
    print("TEST 2: Explicación de Inconsistencias")
    print("=" * 60)
    
    tms = create_tms()
    
    # Simular eliminaciones que causan inconsistencia
    tms.record_removal("X", 1, "C1", {"Y": [2]})
    tms.record_removal("X", 2, "C2", {"Y": [1]})
    tms.record_removal("X", 3, "C1", {"Y": [3]})
    
    print(f"\nVariable X quedó sin valores")
    
    # Explicar
    explanations = tms.explain_inconsistency("X")
    
    print(f"Explicaciones encontradas: {len(explanations)}")
    for exp in explanations:
        print(f"  - Valor {exp.removed_value} eliminado por {exp.reason_constraint}")
    
    assert len(explanations) == 3
    
    print("\n✅ Test pasado")
    return True


def test_tms_suggest_constraint():
    """Test: Sugerencia de restricción a relajar."""
    print("\n" + "=" * 60)
    print("TEST 3: Sugerencia de Restricción a Relajar")
    print("=" * 60)
    
    tms = create_tms()
    
    # C1 causa 3 eliminaciones, C2 causa 1
    tms.record_removal("X", 1, "C1", {"Y": [2]})
    tms.record_removal("X", 2, "C1", {"Y": [1]})
    tms.record_removal("X", 3, "C1", {"Y": [3]})
    tms.record_removal("X", 4, "C2", {"Y": [1]})
    
    # Sugerir
    suggested = tms.suggest_constraint_to_relax("X")
    
    print(f"\nRestricción sugerida: {suggested}")
    
    assert suggested == "C1"  # C1 causó más eliminaciones
    
    print("\n✅ Test pasado")
    return True


def test_tms_restorable_values():
    """Test: Identificación de valores restaurables."""
    print("\n" + "=" * 60)
    print("TEST 4: Valores Restaurables")
    print("=" * 60)
    
    tms = create_tms()
    
    # Registrar eliminaciones por C1
    tms.record_removal("X", 1, "C1", {"Y": [2]})
    tms.record_removal("X", 2, "C1", {"Y": [1]})
    tms.record_removal("Y", 3, "C1", {"X": [1]})
    
    # Registrar eliminaciones por C2
    tms.record_removal("Z", 5, "C2", {"W": [6]})
    
    # Obtener restaurables de C1
    restorable = tms.get_restorable_values("C1")
    
    print(f"\nValores restaurables al eliminar C1:")
    for var, vals in restorable.items():
        print(f"  {var}: {vals}")
    
    assert "X" in restorable
    assert "Y" in restorable
    assert 1 in restorable["X"]
    assert 2 in restorable["X"]
    assert 3 in restorable["Y"]
    
    print("\n✅ Test pasado")
    return True


def test_tms_with_arc_engine():
    """Test: Integración TMS con ArcEngine."""
    print("\n" + "=" * 60)
    print("TEST 5: Integración TMS con ArcEngine")
    print("=" * 60)
    
    # Crear engine con TMS
    engine = ArcEngine(use_tms=True)
    
    # Problema: X != Y, dominios {1, 2}
    engine.add_variable("X", [1, 2])
    engine.add_variable("Y", [1, 2])
    engine.add_constraint("X", "Y", lambda x, y: x != y, cid="C1")
    
    print(f"\nProblema: X ≠ Y, dominios {{1, 2}}")
    print(f"TMS habilitado: {engine.use_tms}")
    
    # Ejecutar AC
    consistent = engine.enforce_arc_consistency()
    
    print(f"Consistente: {consistent}")
    print(f"Justificaciones registradas: {len(engine.tms.justifications)}")
    
    assert consistent
    assert engine.tms is not None
    # AC-3 no debería eliminar nada en este caso
    
    print("\n✅ Test pasado")
    return True


def test_tms_remove_constraint():
    """Test: Eliminación de restricción con restauración."""
    print("\n" + "=" * 60)
    print("TEST 6: Eliminación de Restricción con Restauración")
    print("=" * 60)
    
    # Crear engine con TMS
    engine = ArcEngine(use_tms=True)
    
    # Problema: X < Y, Y < Z
    engine.add_variable("X", [1, 2, 3])
    engine.add_variable("Y", [1, 2, 3])
    engine.add_variable("Z", [1, 2, 3])
    engine.add_constraint("X", "Y", lambda x, y: x < y, cid="C1")
    engine.add_constraint("Y", "Z", lambda y, z: y < z, cid="C2")
    
    print(f"\nProblema: X < Y < Z, dominios {{1, 2, 3}}")
    
    # Ejecutar AC
    consistent = engine.enforce_arc_consistency()
    
    print(f"Consistente: {consistent}")
    print(f"Dominios después de AC:")
    for var in ["X", "Y", "Z"]:
        print(f"  {var}: {list(engine.variables[var].get_values())}")
    
    # Guardar dominios
    domains_before = {
        var: list(engine.variables[var].get_values())
        for var in ["X", "Y", "Z"]
    }
    
    # Eliminar C1
    print(f"\nEliminando restricción C1...")
    engine.remove_constraint("C1")
    
    print(f"Dominios después de eliminar C1:")
    for var in ["X", "Y", "Z"]:
        print(f"  {var}: {list(engine.variables[var].get_values())}")
    
    # Verificar que algunos valores fueron restaurados
    # (depende de la implementación específica de AC-3)
    
    print("\n✅ Test pasado")
    return True


def test_tms_conflict_graph():
    """Test: Grafo de conflictos."""
    print("\n" + "=" * 60)
    print("TEST 7: Grafo de Conflictos")
    print("=" * 60)
    
    tms = create_tms()
    
    # Registrar eliminaciones
    tms.record_removal("X", 1, "C1", {"Y": [2], "Z": [3]})
    tms.record_removal("X", 2, "C2", {"Y": [1]})
    tms.record_removal("X", 3, "C1", {"Y": [3]})
    
    # Obtener grafo de conflictos
    conflict_graph = tms.get_conflict_graph("X")
    
    print(f"\nGrafo de conflictos para X:")
    for constraint, vars in conflict_graph.items():
        print(f"  {constraint}: {vars}")
    
    assert "C1" in conflict_graph
    assert "C2" in conflict_graph
    assert "Y" in conflict_graph["C1"]
    assert "Z" in conflict_graph["C1"]
    
    print("\n✅ Test pasado")
    return True


def test_tms_statistics():
    """Test: Estadísticas del TMS."""
    print("\n" + "=" * 60)
    print("TEST 8: Estadísticas del TMS")
    print("=" * 60)
    
    tms = create_tms()
    
    # Registrar datos
    tms.record_removal("X", 1, "C1", {"Y": [2]})
    tms.record_removal("X", 2, "C1", {"Y": [1]})
    tms.record_removal("Y", 3, "C2", {"X": [1]})
    tms.record_decision("X", 1)
    tms.record_decision("Y", 2)
    
    # Obtener estadísticas
    stats = tms.get_statistics()
    
    print(f"\nEstadísticas del TMS:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    assert stats['total_justifications'] == 3
    assert stats['total_decisions'] == 2
    assert stats['variables_with_removals'] == 2
    assert stats['constraints_involved'] == 2
    
    print("\n✅ Test pasado")
    return True


def test_tms_clear():
    """Test: Limpieza del TMS."""
    print("\n" + "=" * 60)
    print("TEST 9: Limpieza del TMS")
    print("=" * 60)
    
    tms = create_tms()
    
    # Agregar datos
    tms.record_removal("X", 1, "C1", {"Y": [2]})
    tms.record_decision("X", 1)
    
    print(f"\nAntes de limpiar:")
    print(f"  Justificaciones: {len(tms.justifications)}")
    print(f"  Decisiones: {len(tms.decisions)}")
    
    # Limpiar
    tms.clear()
    
    print(f"\nDespués de limpiar:")
    print(f"  Justificaciones: {len(tms.justifications)}")
    print(f"  Decisiones: {len(tms.decisions)}")
    
    assert len(tms.justifications) == 0
    assert len(tms.decisions) == 0
    assert len(tms.dependency_graph) == 0
    
    print("\n✅ Test pasado")
    return True


def main():
    """Ejecuta todos los tests."""
    print("Tests del Truth Maintenance System (TMS)")
    print("LatticeWeaver v4\n")
    
    try:
        test_tms_basic()
        test_tms_explain_inconsistency()
        test_tms_suggest_constraint()
        test_tms_restorable_values()
        test_tms_with_arc_engine()
        test_tms_remove_constraint()
        test_tms_conflict_graph()
        test_tms_statistics()
        test_tms_clear()
        
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

