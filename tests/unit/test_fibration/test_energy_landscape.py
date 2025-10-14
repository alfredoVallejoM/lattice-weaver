"""
Tests unitarios para EnergyLandscape

Pruebas para el módulo de paisaje de energía del Flujo de Fibración.
"""

import pytest
from lattice_weaver.fibration import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness,
    EnergyLandscape,
    EnergyComponents
)


class TestEnergyLandscape:
    """Tests para la clase EnergyLandscape."""
    
    def test_landscape_creation(self):
        """Test: Crear paisaje de energía."""
        hierarchy = ConstraintHierarchy()
        landscape = EnergyLandscape(hierarchy)
        
        assert landscape.hierarchy == hierarchy
        assert landscape.level_weights[ConstraintLevel.LOCAL] == 1.0
        assert landscape.level_weights[ConstraintLevel.PATTERN] == 1.0
        assert landscape.level_weights[ConstraintLevel.GLOBAL] == 1.0
    
    def test_compute_energy_empty_assignment(self):
        """Test: Calcular energía de asignación vacía."""
        hierarchy = ConstraintHierarchy()
        landscape = EnergyLandscape(hierarchy)
        
        energy = landscape.compute_energy({})
        
        assert energy.local_energy == 0.0
        assert energy.pattern_energy == 0.0
        assert energy.global_energy == 0.0
        assert energy.total_energy == 0.0
    
    def test_compute_energy_satisfied_constraints(self):
        """Test: Calcular energía con restricciones satisfechas."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"])
        
        landscape = EnergyLandscape(hierarchy)
        
        # Asignación que satisface la restricción
        assignment = {"x": 1, "y": 2}
        energy = landscape.compute_energy(assignment)
        
        assert energy.local_energy == 0.0
        assert energy.total_energy == 0.0
    
    def test_compute_energy_violated_constraints(self):
        """Test: Calcular energía con restricciones violadas."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], weight=1.0)
        
        landscape = EnergyLandscape(hierarchy)
        
        # Asignación que viola la restricción
        assignment = {"x": 1, "y": 1}
        energy = landscape.compute_energy(assignment)
        
        # Energía = peso_nivel (1.0) * peso_restricción (1.0) * violación (1.0) = 1.0
        assert energy.local_energy == 1.0
        assert energy.total_energy == 1.0
    
    def test_compute_energy_multiple_levels(self):
        """Test: Calcular energía con restricciones en múltiples niveles."""
        hierarchy = ConstraintHierarchy()
        
        # Restricción local violada
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], weight=1.0)
        
        # Restricción de patrón violada
        hierarchy.add_pattern_constraint(
            ["x", "y", "z"],
            lambda a: len(set(a.values())) == len(a),  # All different
            weight=2.0
        )
        
        landscape = EnergyLandscape(hierarchy)
        
        # Asignación que viola ambas restricciones
        assignment = {"x": 1, "y": 1, "z": 1}
        energy = landscape.compute_energy(assignment)
        
        # Local: 1.0 * 1.0 * 1.0 = 1.0
        # Pattern: 1.0 * 2.0 * 1.0 = 2.0
        # Total: 3.0
        assert energy.local_energy == 1.0
        assert energy.pattern_energy == 2.0
        assert energy.total_energy == 3.0
    
    def test_compute_energy_with_level_weights(self):
        """Test: Calcular energía con pesos de nivel modificados."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"], weight=1.0)
        
        landscape = EnergyLandscape(hierarchy)
        
        # Modificar peso del nivel LOCAL
        landscape.level_weights[ConstraintLevel.LOCAL] = 2.0
        
        # Asignación que viola la restricción
        assignment = {"x": 1, "y": 1}
        energy = landscape.compute_energy(assignment)
        
        # Energía = peso_nivel (2.0) * peso_restricción (1.0) * violación (1.0) = 2.0
        assert energy.local_energy == 2.0
        assert energy.total_energy == 2.0
    
    def test_energy_cache(self):
        """Test: Cache de energías."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"])
        
        landscape = EnergyLandscape(hierarchy)
        
        assignment = {"x": 1, "y": 2}
        
        # Primera llamada -> cache miss
        energy1 = landscape.compute_energy(assignment, use_cache=True)
        assert landscape.cache_misses == 1
        assert landscape.cache_hits == 0
        
        # Segunda llamada -> cache hit
        energy2 = landscape.compute_energy(assignment, use_cache=True)
        assert landscape.cache_misses == 1
        assert landscape.cache_hits == 1
        
        # Energías deben ser iguales
        assert energy1.total_energy == energy2.total_energy
    
    def test_clear_cache(self):
        """Test: Limpiar cache."""
        hierarchy = ConstraintHierarchy()
        landscape = EnergyLandscape(hierarchy)
        
        # Calcular algunas energías
        landscape.compute_energy({"x": 1})
        landscape.compute_energy({"x": 2})
        
        assert len(landscape._energy_cache) == 2
        
        # Limpiar cache
        landscape.clear_cache()
        
        assert len(landscape._energy_cache) == 0
        assert landscape.cache_hits == 0
        assert landscape.cache_misses == 0


class TestEnergyGradient:
    """Tests para el cálculo de gradientes de energía."""
    
    def test_compute_energy_gradient(self):
        """Test: Calcular gradiente de energía."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"])
        
        landscape = EnergyLandscape(hierarchy)
        
        # Asignación parcial
        assignment = {"y": 1}
        
        # Calcular gradiente para x con dominio [0, 1, 2]
        gradient = landscape.compute_energy_gradient(assignment, "x", [0, 1, 2])
        
        # x=0 -> satisface restricción -> energía 0
        assert gradient[0] == 0.0
        
        # x=1 -> viola restricción (x=y=1) -> energía 1.0
        assert gradient[1] == 1.0
        
        # x=2 -> satisface restricción -> energía 0
        assert gradient[2] == 0.0
    
    def test_find_energy_minimum(self):
        """Test: Encontrar mínimo de energía."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"])
        
        landscape = EnergyLandscape(hierarchy)
        
        # Asignación parcial
        assignment = {"y": 1}
        
        # Encontrar valor de x que minimiza energía
        min_value, min_energy = landscape.find_energy_minimum(
            assignment, "x", [0, 1, 2]
        )
        
        # Mínimo debe ser x=0 o x=2 (ambos con energía 0)
        assert min_value in [0, 2]
        assert min_energy == 0.0
    
    def test_find_local_minima(self):
        """Test: Encontrar mínimos locales."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"])
        hierarchy.add_local_constraint("y", "z", lambda a: a["y"] != a["z"])
        
        landscape = EnergyLandscape(hierarchy)
        
        # Asignación parcial que viola restricciones (energía > 0)
        assignment = {"x": 1, "y": 1}  # Viola x != y
        domains = {"y": [0, 1, 2], "z": [0, 1, 2]}
        
        # Encontrar mínimos locales
        minima = landscape.find_local_minima(
            assignment, ["y", "z"], domains, max_minima=5
        )
        
        # Debe haber al menos un mínimo (cambiar y a 0 o 2 reduce energía)
        assert len(minima) > 0
        
        # Cada mínimo es una tupla (asignación, energía)
        for assignment_min, energy_min in minima:
            assert isinstance(assignment_min, dict)
            assert isinstance(energy_min, float)
            assert energy_min >= 0.0


class TestEnergyDelta:
    """Tests para el cálculo de deltas de energía."""
    
    def test_compute_energy_delta(self):
        """Test: Calcular delta de energía."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"])
        
        landscape = EnergyLandscape(hierarchy)
        
        assignment = {"x": 1, "y": 2}
        
        # Cambiar y de 2 a 1 (viola restricción)
        delta = landscape.compute_energy_delta(assignment, "y", 2, 1)
        
        # Delta debe ser positivo (aumenta energía)
        assert delta > 0.0
        assert delta == 1.0  # De 0.0 a 1.0
    
    def test_compute_energy_delta_improvement(self):
        """Test: Delta de energía con mejora."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("x", "y", lambda a: a["x"] != a["y"])
        
        landscape = EnergyLandscape(hierarchy)
        
        assignment = {"x": 1, "y": 1}  # Viola restricción
        
        # Cambiar y de 1 a 2 (satisface restricción)
        delta = landscape.compute_energy_delta(assignment, "y", 1, 2)
        
        # Delta debe ser negativo (disminuye energía)
        assert delta < 0.0
        assert delta == -1.0  # De 1.0 a 0.0


class TestCacheStatistics:
    """Tests para estadísticas del cache."""
    
    def test_get_cache_statistics(self):
        """Test: Obtener estadísticas del cache."""
        hierarchy = ConstraintHierarchy()
        landscape = EnergyLandscape(hierarchy)
        
        # Calcular algunas energías
        landscape.compute_energy({"x": 1}, use_cache=True)
        landscape.compute_energy({"x": 1}, use_cache=True)  # Cache hit
        landscape.compute_energy({"x": 2}, use_cache=True)
        
        stats = landscape.get_cache_statistics()
        
        assert stats['cache_size'] == 2
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 2
        assert stats['hit_rate'] == 1/3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

