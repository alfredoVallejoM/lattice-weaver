"""
Tests para HacificationEngineOptimized

Verifica que la versión optimizada mantiene la misma funcionalidad que la original
mientras mejora el rendimiento.

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import pytest
import time
from typing import Dict, Any

from lattice_weaver.fibration.hacification_engine_optimized import (
    HacificationEngineOptimized,
    ConstraintAdapter,
    HacificationResult
)
from lattice_weaver.fibration.hacification_engine import HacificationEngine
from lattice_weaver.fibration.constraint_hierarchy import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness
)
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
from lattice_weaver.arc_engine.core import ArcEngine


# Helper functions for constraints
def ne_constraint(assignment: Dict[str, Any]):
    """Not equal constraint."""
    var1, var2 = list(assignment.keys())
    return assignment[var1] != assignment[var2]


def eq_constraint(assignment: Dict[str, Any]):
    """Equal constraint."""
    var1, var2 = list(assignment.keys())
    return assignment[var1] == assignment[var2]


def unary_constraint_gt_0(assignment: Dict[str, Any]):
    """Unary constraint: value > 0."""
    var = list(assignment.keys())[0]
    return assignment[var] > 0


@pytest.fixture
def basic_hierarchy():
    """Crea una jerarquía básica de restricciones para testing."""
    hierarchy = ConstraintHierarchy()
    hierarchy.add_local_constraint(
        var1="Q0", var2="Q1",
        predicate=ne_constraint,
        hardness=Hardness.HARD,
        metadata={"name": "Q0_ne_Q1"}
    )
    hierarchy.add_local_constraint(
        var1="Q1", var2="Q2",
        predicate=ne_constraint,
        hardness=Hardness.HARD,
        metadata={"name": "Q1_ne_Q2"}
    )
    hierarchy.add_unary_constraint(
        variable="Q0",
        predicate=unary_constraint_gt_0,
        hardness=Hardness.HARD,
        metadata={"name": "Q0_gt_0"}
    )
    return hierarchy


@pytest.fixture
def arc_engine():
    """Crea una instancia de ArcEngine para testing."""
    return ArcEngine(use_tms=False, parallel=False)


@pytest.fixture
def optimized_engine(basic_hierarchy, arc_engine):
    """Crea una instancia de HacificationEngineOptimized."""
    landscape = EnergyLandscapeOptimized(basic_hierarchy)
    return HacificationEngineOptimized(basic_hierarchy, landscape, arc_engine)


@pytest.fixture
def original_engine(basic_hierarchy, arc_engine):
    """Crea una instancia de HacificationEngine original para comparación."""
    landscape = EnergyLandscapeOptimized(basic_hierarchy)
    return HacificationEngine(basic_hierarchy, landscape, arc_engine)


class TestConstraintAdapter:
    """Tests para ConstraintAdapter."""
    
    def test_adapter_initialization(self, basic_hierarchy):
        """Verifica que el adaptador se inicializa correctamente."""
        adapter = ConstraintAdapter(basic_hierarchy)
        assert adapter.hierarchy == basic_hierarchy
        assert len(adapter._adapted_predicates) == 0
        assert len(adapter._relation_names) == 0
    
    def test_get_relation_name_generates_unique_names(self, basic_hierarchy):
        """Verifica que se generan nombres únicos para cada predicado único."""
        adapter = ConstraintAdapter(basic_hierarchy)
        
        constraints = []
        for level in ConstraintLevel:
            constraints.extend(basic_hierarchy.get_constraints_at_level(level))
        
        # Agrupar por ID de predicado (restricciones con el mismo predicado comparten nombre)
        predicate_to_name = {}
        for constraint in constraints:
            pred_id = id(constraint.predicate)
            name = adapter.get_relation_name(constraint)
            
            if pred_id not in predicate_to_name:
                predicate_to_name[pred_id] = name
            else:
                # El mismo predicado debe generar el mismo nombre
                assert predicate_to_name[pred_id] == name
            
            # Llamar de nuevo debe devolver el mismo nombre
            name2 = adapter.get_relation_name(constraint)
            assert name == name2
    
    def test_adapt_binary_constraint(self, basic_hierarchy):
        """Verifica que se adaptan correctamente las restricciones binarias."""
        adapter = ConstraintAdapter(basic_hierarchy)
        
        # Obtener una restricción binaria
        constraint = basic_hierarchy.get_constraints_at_level(ConstraintLevel.LOCAL)[0]
        
        # Adaptar
        adapted = adapter.adapt(constraint)
        
        # Verificar que funciona
        assert adapted(1, 2, {}) == True  # 1 != 2
        assert adapted(1, 1, {}) == False  # 1 == 1


class TestHacificationEngineOptimized:
    """Tests para HacificationEngineOptimized."""
    
    def test_initialization(self, optimized_engine):
        """Verifica que el motor se inicializa correctamente."""
        assert optimized_engine.hierarchy is not None
        assert optimized_engine.landscape is not None
        assert optimized_engine.arc_engine is not None
        assert optimized_engine.constraint_adapter is not None
        assert optimized_engine.stats['hacify_calls'] == 0
    
    def test_hacify_coherent_assignment(self, optimized_engine):
        """Verifica que una asignación coherente es detectada correctamente."""
        assignment = {"Q0": 1, "Q1": 2, "Q2": 3}
        result = optimized_engine.hacify(assignment, strict=True)
        
        assert result.is_coherent == True
        assert result.has_hard_violation == False
        assert optimized_engine.stats['hacify_calls'] == 1
    
    def test_hacify_incoherent_assignment(self, optimized_engine):
        """Verifica que una asignación incoherente es detectada correctamente."""
        assignment = {"Q0": 1, "Q1": 1, "Q2": 2}  # Q0 == Q1, viola Q0_ne_Q1
        result = optimized_engine.hacify(assignment, strict=True)
        
        assert result.has_hard_violation == True
        assert optimized_engine.stats['hacify_calls'] == 1
    
    def test_filter_coherent_extensions(self, optimized_engine):
        """Verifica que el filtrado de extensiones coherentes funciona."""
        base_assignment = {"Q0": 1}
        domain = [0, 1, 2, 3]
        
        coherent_values = optimized_engine.filter_coherent_extensions(
            base_assignment, "Q1", domain, strict=True
        )
        
        # Q1 != Q0 (1), entonces 1 debe ser filtrado
        assert 1 not in coherent_values
        assert len(coherent_values) >= 2  # Al menos 0, 2, 3
        assert optimized_engine.stats['filter_calls'] == 1
    
    def test_state_save_and_restore(self, optimized_engine):
        """Verifica que el guardado y restauración de estado funciona."""
        # Estado inicial
        initial_vars = set(optimized_engine.arc_engine.variables.keys())
        
        # Ejecutar hacify (que guarda y restaura estado)
        assignment = {"Q0": 1, "Q1": 2}
        optimized_engine.hacify(assignment, strict=True)
        
        # Verificar que el estado fue restaurado
        final_vars = set(optimized_engine.arc_engine.variables.keys())
        assert initial_vars == final_vars
        
        # Verificar estadísticas
        assert optimized_engine.stats['state_saves'] >= 1
        assert optimized_engine.stats['state_restores'] >= 1
    
    def test_multiple_hacify_calls_reuse_engine(self, optimized_engine):
        """Verifica que múltiples llamadas reutilizan el mismo ArcEngine."""
        arc_engine_id = id(optimized_engine.arc_engine)
        
        # Múltiples llamadas
        for i in range(5):
            assignment = {"Q0": i+1, "Q1": i+2, "Q2": i+3}
            optimized_engine.hacify(assignment, strict=True)
        
        # Verificar que sigue siendo el mismo ArcEngine
        assert id(optimized_engine.arc_engine) == arc_engine_id
        assert optimized_engine.stats['hacify_calls'] == 5
    
    def test_get_statistics(self, optimized_engine):
        """Verifica que las estadísticas se reportan correctamente."""
        # Ejecutar algunas operaciones
        assignment = {"Q0": 1, "Q1": 2}
        optimized_engine.hacify(assignment, strict=True)
        optimized_engine.filter_coherent_extensions(assignment, "Q2", [0, 1, 2], strict=True)
        
        # Obtener estadísticas
        stats = optimized_engine.get_statistics()
        
        assert "energy_thresholds" in stats
        assert "performance" in stats
        assert stats["performance"]["hacify_calls"] >= 1
        assert stats["performance"]["filter_calls"] >= 1


class TestPerformanceComparison:
    """Tests de comparación de rendimiento entre versión original y optimizada."""
    
    def test_performance_hacify(self, original_engine, optimized_engine):
        """Compara el rendimiento de hacify() entre versiones."""
        assignment = {"Q0": 1, "Q1": 2, "Q2": 3}
        iterations = 100
        
        # Benchmark versión original
        start = time.time()
        for _ in range(iterations):
            original_engine.hacify(assignment, strict=True)
        original_time = time.time() - start
        
        # Benchmark versión optimizada
        start = time.time()
        for _ in range(iterations):
            optimized_engine.hacify(assignment, strict=True)
        optimized_time = time.time() - start
        
        # Calcular speedup
        speedup = original_time / optimized_time if optimized_time > 0 else 1.0
        
        print(f"\n[Performance] hacify() speedup: {speedup:.2f}x")
        print(f"  Original: {original_time:.4f}s ({iterations} iterations)")
        print(f"  Optimized: {optimized_time:.4f}s ({iterations} iterations)")
        
        # La versión optimizada debe ser al menos tan rápida (permitir pequeña variación)
        assert speedup >= 0.8, f"Optimized version is slower: {speedup:.2f}x"
    
    def test_performance_filter(self, original_engine, optimized_engine):
        """Compara el rendimiento de filter_coherent_extensions() entre versiones."""
        base_assignment = {"Q0": 1}
        domain = list(range(10))
        iterations = 50
        
        # Benchmark versión original
        start = time.time()
        for _ in range(iterations):
            original_engine.filter_coherent_extensions(
                base_assignment, "Q1", domain, strict=True
            )
        original_time = time.time() - start
        
        # Benchmark versión optimizada
        start = time.time()
        for _ in range(iterations):
            optimized_engine.filter_coherent_extensions(
                base_assignment, "Q1", domain, strict=True
            )
        optimized_time = time.time() - start
        
        # Calcular speedup
        speedup = original_time / optimized_time if optimized_time > 0 else 1.0
        
        print(f"\n[Performance] filter_coherent_extensions() speedup: {speedup:.2f}x")
        print(f"  Original: {original_time:.4f}s ({iterations} iterations)")
        print(f"  Optimized: {optimized_time:.4f}s ({iterations} iterations)")
        
        # La versión optimizada debe ser al menos tan rápida
        assert speedup >= 0.8, f"Optimized version is slower: {speedup:.2f}x"


class TestFunctionalEquivalence:
    """Tests para verificar equivalencia funcional entre versiones."""
    
    def test_hacify_results_match(self, original_engine, optimized_engine):
        """Verifica que ambas versiones producen los mismos resultados."""
        test_assignments = [
            {"Q0": 1, "Q1": 2, "Q2": 3},
            {"Q0": 1, "Q1": 1, "Q2": 2},  # Incoherente
            {"Q0": 0, "Q1": 1, "Q2": 2},  # Viola Q0 > 0
            {"Q0": 2, "Q1": 3, "Q2": 3},  # Viola Q1 != Q2
        ]
        
        for assignment in test_assignments:
            original_result = original_engine.hacify(assignment, strict=True)
            optimized_result = optimized_engine.hacify(assignment, strict=True)
            
            assert original_result.is_coherent == optimized_result.is_coherent, \
                f"Mismatch for {assignment}"
            assert original_result.has_hard_violation == optimized_result.has_hard_violation, \
                f"Mismatch for {assignment}"
    
    def test_filter_results_match(self, original_engine, optimized_engine):
        """Verifica que el filtrado produce los mismos resultados."""
        test_cases = [
            ({"Q0": 1}, "Q1", [0, 1, 2, 3]),
            ({"Q0": 2}, "Q1", [0, 1, 2, 3, 4]),
            ({"Q0": 1, "Q1": 2}, "Q2", [0, 1, 2, 3]),
        ]
        
        for base_assignment, variable, domain in test_cases:
            original_filtered = set(original_engine.filter_coherent_extensions(
                base_assignment, variable, domain, strict=True
            ))
            optimized_filtered = set(optimized_engine.filter_coherent_extensions(
                base_assignment, variable, domain, strict=True
            ))
            
            assert original_filtered == optimized_filtered, \
                f"Mismatch for {base_assignment}, {variable}"

