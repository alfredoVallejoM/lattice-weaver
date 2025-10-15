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
            variables=("x", "y"), # Usar tupla para inmutabilidad
            predicate=lambda a: a["x"] != a["y"],
            weight=1.0,
            hardness=Hardness.HARD
        )
        
        assert constraint.level == ConstraintLevel.LOCAL
        assert constraint.variables == ("x", "y")
        assert constraint.weight == 1.0
        assert constraint.hardness == Hardness.HARD
    
    def test_constraint_evaluate_satisfied(self):
        """Test: Evaluar restricción satisfecha."""
        constraint = Constraint(
            level=ConstraintLevel.LOCAL,
            variables=("x", "y"),
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
            variables=("x", "y"),
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
            variables=("x", "y"),
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
            variables=("x", "y"),
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
        
        assert len(hierarchy.get_constraints_by_level(ConstraintLevel.LOCAL)) == 0
        assert len(hierarchy.get_constraints_by_level(ConstraintLevel.PATTERN)) == 0
        assert len(hierarchy.get_constraints_by_level(ConstraintLevel.GLOBAL)) == 0
    
    def test_add_local_constraint(self):
        """Test: Añadir restricción local."""
        hierarchy = ConstraintHierarchy()
        
        hierarchy.add_local_constraint(
            "x", "y",
            lambda a: a["x"] != a["y"]
        )
        
        local_constraints = hierarchy.get_constraints_by_level(ConstraintLevel.LOCAL)
        assert len(local_constraints) == 1
        assert local_constraints[0].variables == ("x", "y")
    
    def test_add_unary_constraint(self):
        """Test: Añadir restricción unaria."""
        hierarchy = ConstraintHierarchy()
        
        hierarchy.add_unary_constraint(
            "x",
            lambda a: a["x"] > 0
        )
        
        local_constraints = hierarchy.get_constraints_by_level(ConstraintLevel.LOCAL)
        assert len(local_constraints) == 1
        assert local_constraints[0].variables == ("x",)
    
    def test_add_pattern_constraint(self):
        """Test: Añadir restricción de patrón."""
        hierarchy = ConstraintHierarchy()
        
        hierarchy.add_pattern_constraint(
            ["x", "y", "z"],
            lambda a: len(set(a.values())) == len(a),  # All different
            pattern_type="all_different"
        )
        
        pattern_constraints = hierarchy.get_constraints_by_level(ConstraintLevel.PATTERN)
        assert len(pattern_constraints) == 1
        assert pattern_constraints[0].variables == ("x", "y", "z")
        assert pattern_constraints[0].metadata["pattern_type"] == "all_different"
    
    def test_add_global_constraint(self):
        """Test: Añadir restricción global."""
        hierarchy = ConstraintHierarchy()
        
        hierarchy.add_global_constraint(
            ["x", "y", "z"],
            lambda a: 0.0, # No hay un objetivo directo en el Constraint, solo en EnergyLandscape
            metadata={"objective": "minimize"}
        )
        
        global_constraints = hierarchy.get_constraints_by_level(ConstraintLevel.GLOBAL)
        assert len(global_constraints) == 1
        assert global_constraints[0].metadata["objective"] == "minimize"

    def test_add_custom_level_constraint(self):
        """Test: Añadir restricción a un nivel personalizado."""
        hierarchy = ConstraintHierarchy()
        custom_level_name = "CUSTOM_LEVEL"
        hierarchy.add_level(custom_level_name)

        constraint = Constraint(
            level=custom_level_name,
            variables=("a", "b"),
            predicate=lambda a: a["a"] == a["b"]
        )
        hierarchy.add_constraint(constraint)

        custom_constraints = hierarchy.get_constraints_by_level(custom_level_name)
        assert len(custom_constraints) == 1
        assert custom_constraints[0].variables == ("a", "b")
        assert custom_constraints[0].level == custom_level_name

    def test_evaluate_solution_hard_satisfied(self):
        """Test: Evaluar solución con restricciones HARD satisfechas."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], hardness=Hardness.HARD)
        hierarchy.add_global_constraint(["z"], lambda a: a["z"] > 0, hardness=Hardness.HARD)

        solution = {"x": 1, "y": 2, "z": 5}
        satisfied, energy = hierarchy.evaluate_solution(solution)
        assert satisfied is True
        assert energy == 0.0

    def test_evaluate_solution_hard_violated(self):
        """Test: Evaluar solución con restricciones HARD violadas."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], hardness=Hardness.HARD)

        solution = {"x": 1, "y": 1}
        satisfied, energy = hierarchy.evaluate_solution(solution)
        assert satisfied is False

    def test_evaluate_solution_soft_constraints(self):
        """Test: Evaluar solución con restricciones SOFT."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: abs(a["x"] - a["y"]) / 10.0, weight=2.0, hardness=Hardness.SOFT)
        hierarchy.add_global_constraint(["z"], lambda a: abs(a["z"]) / 5.0, weight=3.0, hardness=Hardness.SOFT)

        solution = {"x": 1, "y": 6, "z": 10}
        satisfied, energy = hierarchy.evaluate_solution(solution)
        # Local: abs(1-6)/10 = 0.5. Energy = 0.5 * 2.0 = 1.0
        # Global: abs(10)/5 = 2.0. Energy = 2.0 * 3.0 = 6.0
        # Total energy = 1.0 + 6.0 = 7.0
        assert satisfied is True # No hay HARD constraints violadas
        assert energy == 7.0

    def test_evaluate_solution_mixed_constraints(self):
        """Test: Evaluar solución con restricciones HARD y SOFT mezcladas."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], hardness=Hardness.HARD)
        hierarchy.add_global_constraint(["z"], lambda a: abs(a["z"]) / 5.0, weight=3.0, hardness=Hardness.SOFT)

        solution = {"x": 1, "y": 2, "z": 10}
        satisfied, energy = hierarchy.evaluate_solution(solution)
        assert satisfied is True
        assert energy == 6.0

        solution_violated = {"x": 1, "y": 1, "z": 10}
        satisfied_v, energy_v = hierarchy.evaluate_solution(solution_violated)
        assert satisfied_v is False
        # La energía de las soft constraints no se suma si una hard constraint falla
        # (aunque la implementación actual la calcula, el 'satisfied' es lo importante)
        # En un sistema real, la búsqueda se detendría antes.
        assert energy_v == 6.0 # La energía de las soft constraints se sigue calculando

    def test_get_all_constraints(self):
        """Test: Obtener todas las restricciones organizadas por nivel."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], hardness=Hardness.HARD)
        hierarchy.add_pattern_constraint(["a", "b"], lambda a: a["a"] > a["b"], hardness=Hardness.SOFT)

        all_constraints = hierarchy.get_all_constraints()
        assert "LOCAL" in all_constraints
        assert "PATTERN" in all_constraints
        assert "GLOBAL" in all_constraints # GLOBAL existe aunque esté vacío
        assert len(all_constraints["LOCAL"]) == 1
        assert len(all_constraints["PATTERN"]) == 1
        assert len(all_constraints["GLOBAL"]) == 0

    def test_to_json_from_json_placeholder(self):
        """Test: Serialización y deserialización (placeholder)."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], hardness=Hardness.HARD)
        hierarchy.add_global_constraint(["z"], lambda a: abs(a["z"]) / 5.0, weight=3.0, hardness=Hardness.SOFT)

        json_data = hierarchy.to_json()
        new_hierarchy = ConstraintHierarchy()
        new_hierarchy.from_json(json_data)

        # Verificar que los niveles y el número de restricciones se mantienen
        assert len(new_hierarchy.get_constraints_by_level(ConstraintLevel.LOCAL)) == 1
        assert len(new_hierarchy.get_constraints_by_level(ConstraintLevel.GLOBAL)) == 1
        # Los predicados son placeholders en la deserialización, así que no se pueden comparar directamente
        assert new_hierarchy.get_constraints_by_level(ConstraintLevel.LOCAL)[0].hardness == Hardness.HARD
        assert new_hierarchy.get_constraints_by_level(ConstraintLevel.GLOBAL)[0].hardness == Hardness.SOFT
        assert new_hierarchy.get_constraints_by_level(ConstraintLevel.GLOBAL)[0].weight == 3.0


# Los métodos `get_constraints_involving`, `classify_by_hardness` y `get_statistics`
# no forman parte de la API abstracta y se eliminarán o refactorizarán si es necesario
# en fases posteriores, o se moverán a una clase de utilidad si su funcionalidad
# es genérica y no específica de la jerarquía de restricciones en sí.
# Por ahora, se eliminan de los tests para adherirse a la API refactorizada.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

