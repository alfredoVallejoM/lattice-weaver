"""
Tests Completos para Todas las Mejoras de Fibration Flow

Tests exhaustivos para:
1. Solver Adaptativo V2
2. TMS Enhanced (CBJ + No-Good Learning)
3. HomotopyRules Optimizado (Sparse Graph)
4. Restricciones Globales (AllDifferent, Cumulative, Table)
5. Búsqueda Híbrida

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import pytest
import time
from typing import Dict, List, Any

from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness, ConstraintLevel
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
from lattice_weaver.arc_engine.core import ArcEngine

# Imports de mejoras
from lattice_weaver.fibration.fibration_search_solver_adaptive_v2 import (
    FibrationSearchSolverAdaptiveV2,
    SolverMode,
    ProblemCharacteristics
)
from lattice_weaver.arc_engine.tms_enhanced import TMSEnhanced, Conflict, NoGood
from lattice_weaver.homotopy.rules_optimized import HomotopyRulesOptimized
from lattice_weaver.fibration.global_constraints import AllDifferent, Cumulative, Table, Task
from lattice_weaver.fibration.hybrid_search import (
    HybridSearch,
    HybridSearchConfig,
    SearchStrategy
)


class TestSolverAdaptiveV2:
    """Tests para Solver Adaptativo V2."""
    
    def test_problem_characteristics_analysis(self):
        """Test análisis de características del problema."""
        hierarchy = ConstraintHierarchy()
        
        # Problema pequeño sin SOFT
        hierarchy.add_local_constraint("A", "B", lambda a: True, Hardness.HARD)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        arc_engine = ArcEngine(use_tms=False)
        
        solver = FibrationSearchSolverAdaptiveV2(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine,
            variables=["A", "B"],
            domains={"A": [1, 2], "B": [1, 2]}
        )
        
        # Verificar análisis
        assert solver.characteristics.n_variables == 2
        assert solver.characteristics.n_constraints == 1
        assert not solver.characteristics.has_soft_constraints
        assert solver.mode == SolverMode.LITE
    
    def test_mode_selection_lite(self):
        """Test selección de modo LITE."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("A", "B", lambda a: a["A"] != a["B"], Hardness.HARD)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        arc_engine = ArcEngine(use_tms=False)
        
        solver = FibrationSearchSolverAdaptiveV2(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine,
            variables=["A", "B"],
            domains={"A": [1, 2, 3], "B": [1, 2, 3]}
        )
        
        assert solver.mode == SolverMode.LITE
        assert not solver.use_homotopy
        assert not solver.use_tms
    
    def test_mode_selection_full(self):
        """Test selección de modo FULL (con SOFT)."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("A", "B", lambda a: True, Hardness.HARD)
        hierarchy.add_local_constraint("A", "A", lambda a: True, weight=1.0, hardness=Hardness.SOFT)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        arc_engine = ArcEngine(use_tms=True)
        
        solver = FibrationSearchSolverAdaptiveV2(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine,
            variables=["A", "B"],
            domains={"A": [1, 2], "B": [1, 2]}
        )
        
        assert solver.mode == SolverMode.FULL
        assert solver.use_homotopy
        assert solver.use_tms
    
    def test_solve_simple_problem(self):
        """Test resolver problema simple."""
        hierarchy = ConstraintHierarchy()
        
        def different(a):
            if "A" in a and "B" in a:
                return a["A"] != a["B"]
            return True
        
        hierarchy.add_local_constraint("A", "B", different, Hardness.HARD)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        arc_engine = ArcEngine(use_tms=False)
        
        solver = FibrationSearchSolverAdaptiveV2(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine,
            variables=["A", "B"],
            domains={"A": [1, 2], "B": [1, 2]}
        )
        
        solution = solver.solve()
        
        assert solution is not None
        assert solution["A"] != solution["B"]


class TestTMSEnhanced:
    """Tests para TMS Enhanced."""
    
    def test_initialization(self):
        """Test inicialización de TMS Enhanced."""
        tms = TMSEnhanced(
            max_nogoods=100,
            nogood_max_size=5,
            enable_learning=True,
            enable_backjumping=True
        )
        
        assert tms.max_nogoods == 100
        assert tms.nogood_max_size == 5
        assert tms.enable_learning
        assert tms.enable_backjumping
    
    def test_record_decision(self):
        """Test registro de decisiones."""
        tms = TMSEnhanced()
        
        tms.record_decision("A", 1, level=0)
        tms.record_decision("B", 2, level=1)
        
        assert len(tms.decisions) == 2
        assert tms.variable_to_level["A"] == 0
        assert tms.variable_to_level["B"] == 1
    
    def test_nogood_learning(self):
        """Test aprendizaje de no-goods."""
        tms = TMSEnhanced(enable_learning=True)
        
        tms.record_decision("A", 1, level=0)
        tms.record_decision("B", 2, level=1)
        
        # Registrar conflicto
        assignment = {"A": 1, "B": 2}
        tms.record_conflict(
            variable="B",
            value=2,
            conflicting_vars={"A"},
            constraint_ids=["c1"],
            assignment=assignment
        )
        
        assert len(tms.nogoods) == 1
        assert tms.total_conflicts == 1
    
    def test_nogood_checking(self):
        """Test verificación de no-goods."""
        tms = TMSEnhanced(enable_learning=True)
        
        # Aprender no-good manualmente
        nogood = NoGood(
            assignments={"A": 1, "B": 2},
            reason="Test conflict",
            learned_at_level=1
        )
        tms.nogoods.append(nogood)
        
        # Verificar que detecta el no-good
        assignment = {"A": 1}
        result = tms.check_nogood("B", 2, assignment)
        
        assert not result  # Debe rechazar
        assert tms.nogood_hits == 1
    
    def test_backjump_level_calculation(self):
        """Test cálculo de nivel de backjump."""
        tms = TMSEnhanced(enable_backjumping=True)
        
        tms.record_decision("A", 1, level=0)
        tms.record_decision("B", 2, level=1)
        tms.record_decision("C", 3, level=2)
        
        # Conflicto en nivel 2 con dependencia en nivel 0
        assignment = {"A": 1, "B": 2, "C": 3}
        backjump_level = tms.record_conflict(
            variable="C",
            value=3,
            conflicting_vars={"A"},  # Depende de A (nivel 0)
            constraint_ids=["c1"],
            assignment=assignment
        )
        
        # Debe backjumpear a nivel 0 (no a nivel 1)
        assert backjump_level == 0


class TestHomotopyRulesOptimized:
    """Tests para HomotopyRules Optimizado."""
    
    def test_initialization(self):
        """Test inicialización."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("A", "B", lambda a: True, Hardness.HARD)
        
        rules = HomotopyRulesOptimized(
            hierarchy=hierarchy,
            dependency_threshold=0.1,
            enable_caching=True,
            lazy_mode=True
        )
        
        assert rules.dependency_threshold == 0.1
        assert rules.enable_caching
        assert rules.lazy_mode
        assert not rules.precomputed
    
    def test_precompute_dependencies(self):
        """Test precomputation de dependencias."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("A", "B", lambda a: True, Hardness.HARD)
        hierarchy.add_local_constraint("B", "C", lambda a: True, Hardness.HARD)
        hierarchy.add_pattern_constraint(["A", "C"], lambda a: True, Hardness.HARD)
        
        rules = HomotopyRulesOptimized(hierarchy=hierarchy)
        rules.precompute_dependencies()
        
        assert rules.precomputed
        assert rules.stats['n_variables'] == 3
        assert rules.stats['n_constraints'] == 3
        assert rules.stats['n_dependencies'] > 0
        assert rules.stats['precomputation_time_ms'] > 0
    
    def test_sparse_graph_density(self):
        """Test densidad del grafo disperso."""
        hierarchy = ConstraintHierarchy()
        
        # Crear 10 variables con pocas restricciones
        for i in range(10):
            for j in range(i+1, min(i+3, 10)):  # Solo vecinos cercanos
                hierarchy.add_local_constraint(
                    f"V{i}", f"V{j}",
                    lambda a: True,
                    Hardness.HARD
                )
        
        rules = HomotopyRulesOptimized(hierarchy=hierarchy, dependency_threshold=0.1)
        rules.precompute_dependencies()
        
        # Densidad debe ser baja (sparse)
        assert rules.stats['graph_density'] < 0.5
        
        # Reducción debe ser significativa
        reduction = 1.0 - rules.stats['graph_density']
        assert reduction > 0.5  # Al menos 50% de reducción
    
    def test_get_dependencies(self):
        """Test obtener dependencias."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("A", "B", lambda a: True, Hardness.HARD)
        hierarchy.add_local_constraint("A", "C", lambda a: True, Hardness.HARD)
        
        rules = HomotopyRulesOptimized(hierarchy=hierarchy)
        rules.precompute_dependencies()
        
        deps_a = rules.get_dependencies("A")
        
        assert "B" in deps_a or "C" in deps_a
        assert len(deps_a) >= 1
    
    def test_caching(self):
        """Test caching de consultas."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("A", "B", lambda a: True, Hardness.HARD)
        
        rules = HomotopyRulesOptimized(hierarchy=hierarchy, enable_caching=True)
        rules.precompute_dependencies()
        
        # Primera consulta (cache miss)
        deps1 = rules.get_dependencies("A")
        misses1 = rules.stats['cache_misses']
        
        # Segunda consulta (cache hit)
        deps2 = rules.get_dependencies("A")
        hits2 = rules.stats['cache_hits']
        
        assert deps1 == deps2
        assert hits2 > 0


class TestGlobalConstraints:
    """Tests para restricciones globales."""
    
    def test_alldifferent_satisfied(self):
        """Test AllDifferent satisfecho."""
        alldiff = AllDifferent(["A", "B", "C"])
        
        assignment = {"A": 1, "B": 2, "C": 3}
        assert alldiff.is_satisfied(assignment)
    
    def test_alldifferent_violated(self):
        """Test AllDifferent violado."""
        alldiff = AllDifferent(["A", "B", "C"])
        
        assignment = {"A": 1, "B": 1, "C": 3}
        assert not alldiff.is_satisfied(assignment)
    
    def test_alldifferent_propagation(self):
        """Test propagación de AllDifferent."""
        alldiff = AllDifferent(["A", "B", "C"])
        
        assignment = {"A": 1}
        domains = {"A": [1], "B": [1, 2, 3], "C": [1, 2, 3]}
        
        consistent, new_domains = alldiff.propagate(assignment, domains)
        
        assert consistent
        assert 1 not in new_domains["B"]
        assert 1 not in new_domains["C"]
    
    def test_cumulative_satisfied(self):
        """Test Cumulative satisfecho."""
        tasks = [
            Task(start_var="T1", duration=2, resource=2),
            Task(start_var="T2", duration=2, resource=2)
        ]
        cumulative = Cumulative(tasks, capacity=4, time_horizon=10)
        
        # No overlap
        assignment = {"T1": 0, "T2": 3}
        assert cumulative.is_satisfied(assignment)
    
    def test_cumulative_violated(self):
        """Test Cumulative violado."""
        tasks = [
            Task(start_var="T1", duration=3, resource=3),
            Task(start_var="T2", duration=2, resource=3)
        ]
        cumulative = Cumulative(tasks, capacity=4, time_horizon=10)
        
        # Overlap que supera capacidad
        assignment = {"T1": 0, "T2": 1}
        assert not cumulative.is_satisfied(assignment)
    
    def test_table_support_mode(self):
        """Test Table en modo support."""
        table = Table(
            variables=["A", "B"],
            tuples=[(1, 2), (2, 3), (3, 4)],
            mode="support"
        )
        
        # Tupla permitida
        assert table.is_satisfied({"A": 1, "B": 2})
        
        # Tupla no permitida
        assert not table.is_satisfied({"A": 1, "B": 3})
    
    def test_table_conflict_mode(self):
        """Test Table en modo conflict."""
        table = Table(
            variables=["A", "B"],
            tuples=[(1, 1), (2, 2)],
            mode="conflict"
        )
        
        # Tupla no prohibida
        assert table.is_satisfied({"A": 1, "B": 2})
        
        # Tupla prohibida
        assert not table.is_satisfied({"A": 1, "B": 1})


class TestHybridSearch:
    """Tests para búsqueda híbrida."""
    
    def test_initialization(self):
        """Test inicialización."""
        hierarchy = ConstraintHierarchy()
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        config = HybridSearchConfig(
            strategy=SearchStrategy.HILL_CLIMBING,
            systematic_depth=3,
            local_iterations=100
        )
        
        search = HybridSearch(
            hierarchy=hierarchy,
            landscape=landscape,
            variables=["A", "B"],
            domains={"A": [1, 2], "B": [1, 2]},
            config=config
        )
        
        assert search.config.strategy == SearchStrategy.HILL_CLIMBING
        assert search.config.systematic_depth == 3
        assert search.config.local_iterations == 100
    
    def test_hill_climbing_search(self):
        """Test búsqueda con hill climbing."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint("A", "A", lambda a: True, weight=1.0, hardness=Hardness.SOFT)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        config = HybridSearchConfig(
            strategy=SearchStrategy.HILL_CLIMBING,
            systematic_depth=2,
            local_iterations=10,
            time_limit_seconds=5.0
        )
        
        search = HybridSearch(
            hierarchy=hierarchy,
            landscape=landscape,
            variables=["A", "B"],
            domains={"A": [1, 2, 3], "B": [1, 2, 3]},
            config=config
        )
        
        solution = search.search()
        
        # Debe encontrar alguna solución
        assert solution is not None
        assert len(solution) == 2


def test_integration_all_improvements():
    """Test de integración de todas las mejoras juntas."""
    # Crear problema con restricciones HARD y SOFT
    hierarchy = ConstraintHierarchy()
    
    # HARD: A != B
    hierarchy.add_local_constraint(
        "A", "B",
        lambda a: a.get("A") != a.get("B") if "A" in a and "B" in a else True,
        Hardness.HARD
    )
    
    # SOFT: Preferencia por valores bajos
    hierarchy.add_local_constraint(
        "A", "A",
        lambda a: a.get("A", 0) <= 2 if "A" in a else True,
        weight=1.0,
        hardness=Hardness.SOFT
    )
    
    landscape = EnergyLandscapeOptimized(hierarchy)
    arc_engine = ArcEngine(use_tms=True)
    
    # Usar solver adaptativo V2
    solver = FibrationSearchSolverAdaptiveV2(
        hierarchy=hierarchy,
        landscape=landscape,
        arc_engine=arc_engine,
        variables=["A", "B"],
        domains={"A": [1, 2, 3], "B": [1, 2, 3]},
        max_backtracks=1000,
        max_iterations=1000,
        time_limit_seconds=10.0
    )
    
    solution = solver.solve()
    
    # Verificar solución
    assert solution is not None
    assert solution["A"] != solution["B"]  # HARD satisfecho
    
    # Verificar estadísticas
    stats = solver.get_statistics()
    assert stats['solution']['found']
    assert stats['search']['backtracks'] >= 0
    assert stats['search']['nodes_explored'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

