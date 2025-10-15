import pytest
from lattice_weaver.fibration import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness,
    EnergyLandscapeOptimized
)


class TestEnergyLandscapeOptimized:
    """Tests para la clase EnergyLandscapeOptimized."""
    
    def test_landscape_creation(self):
        """Test: Crear paisaje de energía."""
        hierarchy = ConstraintHierarchy()
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        assert landscape.hierarchy == hierarchy
        assert landscape.level_weights[ConstraintLevel.LOCAL] == 1.0
        assert landscape.level_weights[ConstraintLevel.PATTERN] == 1.0
        assert landscape.level_weights[ConstraintLevel.GLOBAL] == 1.0
    
    def test_compute_energy_empty_assignment(self):
        """Test: Calcular energía de asignación vacía."""
        hierarchy = ConstraintHierarchy()
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        satisfied, total_energy, local_energy, pattern_energy, global_energy = landscape.compute_energy({})
        
        assert satisfied is True
        assert total_energy == 0.0
        assert local_energy == 0.0
        assert pattern_energy == 0.0
        assert global_energy == 0.0
    
    def test_compute_energy_satisfied_constraints(self):
        """Test: Calcular energía con restricciones satisfechas."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], hardness=Hardness.HARD)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Asignación que satisface la restricción
        assignment = {"x": 1, "y": 2}
        satisfied, total_energy, local_energy, pattern_energy, global_energy = landscape.compute_energy(assignment)
        
        assert satisfied is True
        assert total_energy == 0.0
        assert local_energy == 0.0
        assert pattern_energy == 0.0
        assert global_energy == 0.0
    
    def test_compute_energy_violated_hard_constraints(self):
        """Test: Calcular energía con restricciones HARD violadas."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], hardness=Hardness.HARD)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Asignación que viola la restricción
        assignment = {"x": 1, "y": 1}
        satisfied, total_energy, local_energy, pattern_energy, global_energy = landscape.compute_energy(assignment)
        
        assert satisfied is False
        assert total_energy == 0.0 # La energía de soft constraints no se suma si una hard falla
        assert local_energy == 0.0
        assert pattern_energy == 0.0
        assert global_energy == 0.0

    def test_compute_energy_violated_soft_constraints(self):
        """Test: Calcular energía con restricciones SOFT violadas."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: abs(a["x"] - a["y"]), weight=1.0, hardness=Hardness.SOFT)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Asignación que viola la restricción
        assignment = {"x": 1, "y": 3}
        satisfied, total_energy, local_energy, pattern_energy, global_energy = landscape.compute_energy(assignment)
        
        # Energía = peso_nivel (1.0) * peso_restricción (1.0) * violación (2.0) = 2.0
        assert satisfied is True # No hay hard constraints violadas
        assert total_energy == 2.0
        assert local_energy == 2.0
        assert pattern_energy == 0.0
        assert global_energy == 0.0
    
    def test_compute_energy_multiple_levels_soft_violated(self):
        """Test: Calcular energía con restricciones SOFT en múltiples niveles."""
        hierarchy = ConstraintHierarchy()
        
        # Restricción local violada
        hierarchy.add_local_constraint("x", "y", lambda a: abs(a["x"] - a["y"]), weight=1.0, hardness=Hardness.SOFT)
        
        # Restricción de patrón violada
        hierarchy.add_pattern_constraint(
            ["x", "y", "z"],
            lambda a: 1.0 if len(set(a.values())) != len(a) else 0.0,  # All different
            weight=2.0, hardness=Hardness.SOFT
        )
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Asignación que viola ambas restricciones
        assignment = {"x": 1, "y": 1, "z": 1}
        satisfied, total_energy, local_energy, pattern_energy, global_energy = landscape.compute_energy(assignment)
        
        # Local: 1.0 * 1.0 * 0.0 = 0.0 (x-y=0)
        # Pattern: 1.0 * 2.0 * 1.0 = 2.0 (no all different)
        # Total: 2.0
        assert satisfied is True
        assert total_energy == 2.0
        assert local_energy == 0.0
        assert pattern_energy == 2.0
        assert global_energy == 0.0
    
    def test_compute_energy_with_level_weights(self):
        """Test: Calcular energía con pesos de nivel modificados."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: abs(a["x"] - a["y"]), weight=1.0, hardness=Hardness.SOFT)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Modificar peso del nivel LOCAL
        landscape.level_weights[ConstraintLevel.LOCAL] = 2.0
        
        # Asignación que viola la restricción
        assignment = {"x": 1, "y": 2}
        satisfied, total_energy, local_energy, pattern_energy, global_energy = landscape.compute_energy(assignment)
        
        # Energía = peso_nivel (2.0) * peso_restricción (1.0) * violación (1.0) = 2.0
        assert satisfied is True
        assert total_energy == 2.0
        assert local_energy == 2.0
        assert pattern_energy == 0.0
        assert global_energy == 0.0
    
    def test_energy_cache(self):
        """Test: Cache de energías."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], hardness=Hardness.HARD)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        assignment = {"x": 1, "y": 2}
        
        # Primera llamada -> cache miss
        satisfied1, energy1, _, _, _ = landscape.compute_energy(assignment, use_cache=True)
        assert landscape.cache_misses == 1
        assert landscape.cache_hits == 0
        
        # Segunda llamada -> cache hit
        satisfied2, energy2, _, _, _ = landscape.compute_energy(assignment, use_cache=True)
        assert landscape.cache_misses == 1
        assert landscape.cache_hits == 1
        
        # Energías deben ser iguales
        assert satisfied1 == satisfied2
        assert energy1 == energy2
    
    def test_clear_cache(self):
        """Test: Limpiar cache."""
        hierarchy = ConstraintHierarchy()
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Calcular algunas energías
        landscape.compute_energy({"x": 1})
        landscape.compute_energy({"x": 2})
        
        assert len(landscape._energy_cache) == 2
        
        # Limpiar cache
        landscape.clear_cache()
        
        assert len(landscape._energy_cache) == 0
        assert landscape.cache_hits == 0
        assert landscape.cache_misses == 0

    def test_compute_energy_incremental_soft_constraint(self):
        """Test: Calcular energía incremental para soft constraints."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: abs(a["x"] - a["y"]), weight=1.0, hardness=Hardness.SOFT)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        base_assignment = {"x": 1}
        base_satisfied, base_total_energy, base_local_energy, base_pattern_energy, base_global_energy = landscape.compute_energy(base_assignment)
        
        # Cambiar y de no asignado a 2
        new_satisfied, new_total_energy, new_local_energy, new_pattern_energy, new_global_energy = landscape.compute_energy_incremental(
            base_assignment, (base_satisfied, base_total_energy, base_local_energy, base_pattern_energy, base_global_energy), "y", 2
        )
        
        assert new_satisfied is True
        assert new_total_energy == 1.0 # abs(1-2) = 1.0
        assert new_local_energy == 1.0
        assert new_pattern_energy == 0.0
        assert new_global_energy == 0.0

        # Cambiar y de 2 a 3
        new_satisfied_2, new_total_energy_2, new_local_energy_2, new_pattern_energy_2, new_global_energy_2 = landscape.compute_energy_incremental(
            {"x": 1, "y": 2}, (new_satisfied, new_total_energy, new_local_energy, new_pattern_energy, new_global_energy), "y", 3
        )
        assert new_satisfied_2 is True
        assert new_total_energy_2 == 2.0 # abs(1-3) = 2.0
        assert new_local_energy_2 == 2.0
        assert new_pattern_energy_2 == 0.0
        assert new_global_energy_2 == 0.0

    def test_compute_energy_incremental_hard_constraint_satisfied(self):
        """Test: Calcular energía incremental para hard constraints (satisfecha)."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], hardness=Hardness.HARD)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        base_assignment = {"x": 1}
        base_satisfied, base_total_energy, base_local_energy, base_pattern_energy, base_global_energy = landscape.compute_energy(base_assignment)
        
        # Añadir y=2, satisface la hard constraint
        new_satisfied, new_total_energy, new_local_energy, new_pattern_energy, new_global_energy = landscape.compute_energy_incremental(
            base_assignment, (base_satisfied, base_total_energy, base_local_energy, base_pattern_energy, base_global_energy), "y", 2
        )
        
        assert new_satisfied is True
        assert new_total_energy == 0.0
        assert new_local_energy == 0.0
        assert new_pattern_energy == 0.0
        assert new_global_energy == 0.0

    def test_compute_energy_incremental_hard_constraint_violated(self):
        """Test: Calcular energía incremental para hard constraints (violada)."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], hardness=Hardness.HARD)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        base_assignment = {"x": 1}
        base_satisfied, base_total_energy, base_local_energy, base_pattern_energy, base_global_energy = landscape.compute_energy(base_assignment)
        
        # Añadir y=1, viola la hard constraint
        new_satisfied, new_total_energy, new_local_energy, new_pattern_energy, new_global_energy = landscape.compute_energy_incremental(
            base_assignment, (base_satisfied, base_total_energy, base_local_energy, base_pattern_energy, base_global_energy), "y", 1
        )
        
        assert new_satisfied is False
        assert new_total_energy == 0.0 # La energía de soft constraints no se suma si una hard falla
        assert new_local_energy == 0.0
        assert new_pattern_energy == 0.0
        assert new_global_energy == 0.0

    def test_compute_energy_gradient_optimized(self):
        """Test: Calcular gradiente de energía optimizado."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: abs(a["x"] - a["y"]), weight=1.0, hardness=Hardness.SOFT)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Asignación parcial
        assignment = {"y": 1}
        base_satisfied, base_total_energy, base_local_energy, base_pattern_energy, base_global_energy = landscape.compute_energy(assignment)
        
        # Calcular gradiente para x con dominio [0, 1, 2]
        gradient = landscape.compute_energy_gradient_optimized(assignment, (base_satisfied, base_total_energy, base_local_energy, base_pattern_energy, base_global_energy), "x", [0, 1, 2])
        
        # x=0 -> abs(0-1) = 1.0
        assert gradient[0] == 1.0
        
        # x=1 -> abs(1-1) = 0.0
        assert gradient[1] == 0.0
        
        # x=2 -> abs(2-1) = 1.0
        assert gradient[2] == 1.0

    def test_to_json_from_json(self):
        """Test: Serialización y deserialización del paisaje de energía."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], hardness=Hardness.HARD)
        hierarchy.add_global_constraint(["z"], lambda a: abs(a["z"]) / 5.0, weight=3.0, hardness=Hardness.SOFT)

        landscape = EnergyLandscapeOptimized(hierarchy)
        landscape.level_weights[ConstraintLevel.LOCAL] = 0.5

        json_data = landscape.to_json()
        new_landscape = EnergyLandscapeOptimized(ConstraintHierarchy()) # Pasa una jerarquía vacía inicialmente
        new_landscape.from_json(json_data)

        # Verificar que los pesos de nivel se mantienen
        assert new_landscape.level_weights[ConstraintLevel.LOCAL] == 0.5
        assert new_landscape.level_weights[ConstraintLevel.GLOBAL] == 1.0

        # Verificar que las restricciones se han deserializado (predicados son placeholders)
        assert len(new_landscape.hierarchy.get_constraints_by_level(ConstraintLevel.LOCAL)) == 1
        assert len(new_landscape.hierarchy.get_constraints_by_level(ConstraintLevel.GLOBAL)) == 1
        assert new_landscape.hierarchy.get_constraints_by_level(ConstraintLevel.LOCAL)[0].hardness == Hardness.HARD
        assert new_landscape.hierarchy.get_constraints_by_level(ConstraintLevel.GLOBAL)[0].hardness == Hardness.SOFT
        assert new_landscape.hierarchy.get_constraints_by_level(ConstraintLevel.GLOBAL)[0].weight == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

