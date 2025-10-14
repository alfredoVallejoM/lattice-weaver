"""
Tests unitarios para ConstraintHierarchy

Pruebas para el módulo de jerarquía de restricciones del Flujo de Fibración.
"""

import pytest
from lattice_weaver.fibration import (
    ConstraintHierarchy,
    Constraint,
    ConstraintLevel,
    Hardness
)


class TestConstraint:
    """Tests para la clase Constraint."""
    
    def test_constraint_creation(self):
        """Test: Crear una restricción básica."""
        constraint = Constraint(
            level=ConstraintLevel.LOCAL,
            variables=["x", "y"],
            predicate=lambda a: a["x"] != a["y"],
            weight=1.0,
            hardness=Hardness.HARD
        )
        
        assert constraint.level == ConstraintLevel.LOCAL
        assert constraint.variables == ["x", "y"]
        assert constraint.weight == 1.0
        assert constraint.hardness == Hardness.HARD
    
    def test_constraint_evaluate_satisfied(self):
        """Test: Evaluar restricción satisfecha."""
        constraint = Constraint(
            level=ConstraintLevel.LOCAL,
            variables=["x", "y"],
            predicate=lambda a: a["x"] != a["y"]
        )
        
        assignment = {"x": 1, "y": 2}
        satisfied, violation = constraint.evaluate(assignment)
        
        assert satisfied is True
        assert violation == 0.0
    
    def test_constraint_evaluate_violated(self):
        """Test: Evaluar restricción violada."""
        constraint = Constraint(
            level=ConstraintLevel.LOCAL,
            variables=["x", "y"],
            predicate=lambda a: a["x"] != a["y"]
        )
        
        assignment = {"x": 1, "y": 1}
        satisfied, violation = constraint.evaluate(assignment)
        
        assert satisfied is False
        assert violation == 1.0
    
    def test_constraint_evaluate_partial_assignment(self):
        """Test: Evaluar restricción con asignación parcial."""
        constraint = Constraint(
            level=ConstraintLevel.LOCAL,
            variables=["x", "y"],
            predicate=lambda a: a["x"] != a["y"]
        )
        
        # Solo x asignada, y no
        assignment = {"x": 1}
        satisfied, violation = constraint.evaluate(assignment)
        
        # Restricción no evaluable aún -> considerada satisfecha
        assert satisfied is True
        assert violation == 0.0
    
    def test_constraint_evaluate_with_float_result(self):
        """Test: Evaluar restricción que devuelve float (grado de violación)."""
        def distance_constraint(a):
            """Restricción soft: preferir que x e y estén cerca."""
            return abs(a["x"] - a["y"]) / 10.0  # Normalizado a [0, 1]
        
        constraint = Constraint(
            level=ConstraintLevel.PATTERN,
            variables=["x", "y"],
            predicate=distance_constraint,
            hardness=Hardness.SOFT
        )
        
        # x=0, y=5 -> distancia 5 -> violación 0.5
        assignment = {"x": 0, "y": 5}
        satisfied, violation = constraint.evaluate(assignment)
        
        assert satisfied is False  # violación > 0
        assert violation == 0.5


class TestConstraintHierarchy:
    """Tests para la clase ConstraintHierarchy."""
    
    def test_hierarchy_creation(self):
        """Test: Crear jerarquía vacía."""
        hierarchy = ConstraintHierarchy()
        
        assert len(hierarchy.get_constraints_at_level(ConstraintLevel.LOCAL)) == 0
        assert len(hierarchy.get_constraints_at_level(ConstraintLevel.PATTERN)) == 0
        assert len(hierarchy.get_constraints_at_level(ConstraintLevel.GLOBAL)) == 0
    
    def test_add_local_constraint(self):
        """Test: Añadir restricción local."""
        hierarchy = ConstraintHierarchy()
        
        hierarchy.add_local_constraint(
            "x", "y",
            lambda a: a["x"] != a["y"]
        )
        
        local_constraints = hierarchy.get_constraints_at_level(ConstraintLevel.LOCAL)
        assert len(local_constraints) == 1
        assert local_constraints[0].variables == ["x", "y"]
    
    def test_add_unary_constraint(self):
        """Test: Añadir restricción unaria."""
        hierarchy = ConstraintHierarchy()
        
        hierarchy.add_unary_constraint(
            "x",
            lambda a: a["x"] > 0
        )
        
        local_constraints = hierarchy.get_constraints_at_level(ConstraintLevel.LOCAL)
        assert len(local_constraints) == 1
        assert local_constraints[0].variables == ["x"]
    
    def test_add_pattern_constraint(self):
        """Test: Añadir restricción de patrón."""
        hierarchy = ConstraintHierarchy()
        
        hierarchy.add_pattern_constraint(
            ["x", "y", "z"],
            lambda a: len(set(a.values())) == len(a),  # All different
            pattern_type="all_different"
        )
        
        pattern_constraints = hierarchy.get_constraints_at_level(ConstraintLevel.PATTERN)
        assert len(pattern_constraints) == 1
        assert pattern_constraints[0].variables == ["x", "y", "z"]
        assert pattern_constraints[0].metadata["pattern_type"] == "all_different"
    
    def test_add_global_constraint(self):
        """Test: Añadir restricción global."""
        hierarchy = ConstraintHierarchy()
        
        hierarchy.add_global_constraint(
            ["x", "y", "z"],
            lambda a: sum(a.values()),  # Minimizar suma
            objective="minimize"
        )
        
        global_constraints = hierarchy.get_constraints_at_level(ConstraintLevel.GLOBAL)
        assert len(global_constraints) == 1
        assert global_constraints[0].metadata["objective"] == "minimize"
    
    def test_get_constraints_involving(self):
        """Test: Obtener restricciones que involucran una variable."""
        hierarchy = ConstraintHierarchy()
        
        # Añadir varias restricciones
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"])
        hierarchy.add_local_constraint("x", "z", lambda a: a["x"] != a["z"])
        hierarchy.add_local_constraint("y", "z", lambda a: a["y"] != a["z"])
        
        # x aparece en 2 restricciones
        x_constraints = hierarchy.get_constraints_involving("x")
        assert len(x_constraints) == 2
        
        # y aparece en 2 restricciones
        y_constraints = hierarchy.get_constraints_involving("y")
        assert len(y_constraints) == 2
        
        # z aparece en 2 restricciones
        z_constraints = hierarchy.get_constraints_involving("z")
        assert len(z_constraints) == 2
    
    def test_classify_by_hardness(self):
        """Test: Clasificar restricciones por dureza."""
        hierarchy = ConstraintHierarchy()
        
        # Añadir restricciones HARD y SOFT
        hierarchy.add_local_constraint(
            "x", "y",
            lambda a: a["x"] != a["y"],
            hardness=Hardness.HARD
        )
        
        hierarchy.add_global_constraint(
            ["x", "y"],
            lambda a: sum(a.values()),
            hardness=Hardness.SOFT
        )
        
        by_hardness = hierarchy.classify_by_hardness()
        
        assert len(by_hardness[Hardness.HARD]) == 1
        assert len(by_hardness[Hardness.SOFT]) == 1
    
    def test_get_statistics(self):
        """Test: Obtener estadísticas de la jerarquía."""
        hierarchy = ConstraintHierarchy()
        
        # Añadir restricciones de diferentes tipos
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"])
        hierarchy.add_local_constraint("y", "z", lambda a: a["y"] != a["z"])
        hierarchy.add_pattern_constraint(["x", "y", "z"], lambda a: True)
        hierarchy.add_global_constraint(["x", "y", "z"], lambda a: 0.0)
        
        stats = hierarchy.get_statistics()
        
        assert stats['total_constraints'] == 4
        assert stats['by_level']['LOCAL'] == 2
        assert stats['by_level']['PATTERN'] == 1
        assert stats['by_level']['GLOBAL'] == 1


class TestConstraintEvaluation:
    """Tests para la evaluación de restricciones complejas."""
    
    def test_all_different_constraint(self):
        """Test: Restricción all_different."""
        def all_different(assignment):
            values = list(assignment.values())
            return len(values) == len(set(values))
        
        constraint = Constraint(
            level=ConstraintLevel.PATTERN,
            variables=["x", "y", "z"],
            predicate=all_different
        )
        
        # Todos diferentes -> satisfecha
        assignment1 = {"x": 1, "y": 2, "z": 3}
        satisfied1, _ = constraint.evaluate(assignment1)
        assert satisfied1 is True
        
        # Dos iguales -> violada
        assignment2 = {"x": 1, "y": 1, "z": 3}
        satisfied2, _ = constraint.evaluate(assignment2)
        assert satisfied2 is False
    
    def test_sum_constraint(self):
        """Test: Restricción de suma."""
        def sum_equals_10(assignment):
            total = sum(assignment.values())
            # Devolver grado de violación: distancia a 10
            return abs(total - 10) / 10.0
        
        constraint = Constraint(
            level=ConstraintLevel.GLOBAL,
            variables=["x", "y", "z"],
            predicate=sum_equals_10,
            hardness=Hardness.SOFT
        )
        
        # Suma = 10 -> satisfecha
        assignment1 = {"x": 3, "y": 3, "z": 4}
        satisfied1, violation1 = constraint.evaluate(assignment1)
        assert satisfied1 is True
        assert violation1 == 0.0
        
        # Suma = 15 -> violación 0.5
        assignment2 = {"x": 5, "y": 5, "z": 5}
        satisfied2, violation2 = constraint.evaluate(assignment2)
        assert satisfied2 is False
        assert violation2 == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

