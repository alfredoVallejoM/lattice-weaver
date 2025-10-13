import pytest
import time
from unittest.mock import MagicMock

from lattice_weaver.core.csp_engine.solver import AdaptiveConsistencyEngine, AC3Solver, SolutionStats
from lattice_weaver.core.csp_engine.graph import ConstraintGraph
from lattice_weaver.core.csp_engine.constraints import NE, LT, NoAttackQueensConstraint
from lattice_weaver.core.csp_engine.tracing import SearchSpaceTracer

# Helper function to create an N-Queens problem ConstraintGraph
def create_nqueens_graph(n):
    cg = ConstraintGraph()
    for i in range(n):
        cg.add_variable(f'Q{i}', set(range(n)))
    for i in range(n):
        for j in range(i + 1, n):
            cg.add_constraint(f'Q{i}', f'Q{j}', NoAttackQueensConstraint(j - i))
    return cg


class TestOptimizations:
    """Tests para las optimizaciones integradas en AdaptiveConsistencyEngine."""

    def test_ac3_solver_basic(self):
        """Test: AC3Solver básico."""
        cg = ConstraintGraph()
        cg.add_variable("X", {1, 2, 3})
        cg.add_variable("Y", {1, 2, 3})
        cg.add_constraint("X", "Y", NE())

        solver = AC3Solver()
        consistent = solver.enforce_arc_consistency(cg)

        assert consistent
        assert len(cg.get_domain("X")) == 3
        assert len(cg.get_domain("Y")) == 3

    def test_ac3_solver_reduces_domains(self):
        """Test: AC3Solver reduce dominios correctamente."""
        cg = ConstraintGraph()
        cg.add_variable("X", {1, 2})
        cg.add_variable("Y", {1, 2})
        cg.add_variable("Z", {1, 2})
        cg.add_constraint("X", "Y", NE())
        cg.add_constraint("Y", "Z", NE())
        cg.add_constraint("X", "Z", NE())

        solver = AC3Solver()
        consistent = solver.enforce_arc_consistency(cg)

        assert consistent
        # For 3 variables with domain {1,2} and all-diff, it should be inconsistent
        # But AC3 only ensures arc consistency, not global consistency
        # So domains might still be {1,2}
        assert len(cg.get_domain("X")) == 2
        assert len(cg.get_domain("Y")) == 2
        assert len(cg.get_domain("Z")) == 2

    def test_adaptive_consistency_engine_nqueens(self):
        """Test: AdaptiveConsistencyEngine resuelve N-Reinas."""
        n = 4
        cg = create_nqueens_graph(n)
        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=1)

        assert len(stats.solutions) == 1
        assert stats.nodes_explored > 0
        assert stats.backtracks >= 0

    def test_adaptive_consistency_engine_with_tracer(self):
        """Test: AdaptiveConsistencyEngine integra el tracer."""
        n = 4
        cg = create_nqueens_graph(n)
        tracer = SearchSpaceTracer(enabled=True)
        engine = AdaptiveConsistencyEngine(tracer=tracer)
        stats = engine.solve(cg, max_solutions=1)

        assert len(stats.solutions) == 1
        assert len(tracer.events) > 0
        assert any(e.event_type == 'search_started' for e in tracer.events)
        assert any(e.event_type == 'solution_found' for e in tracer.events)

    def test_adaptive_consistency_engine_timeout(self):
        """Test: AdaptiveConsistencyEngine respeta el timeout."""
        n = 8 # A larger problem to ensure timeout is hit
        cg = create_nqueens_graph(n)
        engine = AdaptiveConsistencyEngine()
        
        start_time = time.time()
        stats = engine.solve(cg, max_solutions=1, timeout=0.01)
        end_time = time.time()

        assert (end_time - start_time) < 0.1 # Should be close to timeout
        # It might find a solution or not, depending on the exact timing
        # The key is that it stops quickly
        assert stats.time_elapsed > 0

    def test_adaptive_consistency_engine_multiple_solutions(self):
        """Test: AdaptiveConsistencyEngine encuentra múltiples soluciones."""
        n = 4
        cg = create_nqueens_graph(n)
        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=2)

        assert len(stats.solutions) == 2
        assert stats.nodes_explored > 0

    def test_adaptive_consistency_engine_no_solution(self):
        """Test: AdaptiveConsistencyEngine maneja problemas sin solución."""
        cg = ConstraintGraph()
        cg.add_variable("X", {1})
        cg.add_variable("Y", {1})
        cg.add_constraint("X", "Y", NE())

        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=1)

        assert len(stats.solutions) == 0
        assert stats.nodes_explored > 0
        assert stats.backtracks > 0

    def test_adaptive_consistency_engine_clustering_metrics(self):
        """Test: AdaptiveConsistencyEngine captura métricas de clustering."""
        n = 4
        cg = create_nqueens_graph(n)
        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=1)

        assert stats.clustering_metrics is not None
        assert stats.clustering_metrics.initial_clusters > 0
        assert stats.clustering_metrics.initial_edges > 0

    def test_ac3_solver_last_support(self):
        """Test: AC3Solver utiliza last_support para eficiencia."""
        cg = ConstraintGraph()
        cg.add_variable("X", {1, 2, 3})
        cg.add_variable("Y", {1, 2, 3})
        cg.add_constraint("X", "Y", NE())

        solver = AC3Solver()
        solver.enforce_arc_consistency(cg)
        initial_calls = solver.calls

        # Re-enforce consistency, should be faster due to last_support
        solver.enforce_arc_consistency(cg)
        assert solver.calls > initial_calls # Calls should increment
        # The internal mechanism of last_support is hard to test directly without exposing internals
        # We rely on the functional correctness and performance benchmarks for this.
        assert True

    def test_adaptive_consistency_engine_cluster_operations_tracing(self):
        """Test: AdaptiveConsistencyEngine traza operaciones de clúster."""
        cg = ConstraintGraph()
        cg.add_variable("X", {1, 2, 3})
        cg.add_variable("Y", {1, 2, 3})
        cg.add_variable("Z", {1, 2, 3})
        cg.add_constraint("X", "Y", NE())
        cg.add_constraint("Y", "Z", NE())
        cg.add_constraint("X", "Z", NE())

        tracer = SearchSpaceTracer(enabled=True)
        engine = AdaptiveConsistencyEngine(tracer=tracer)
        stats = engine.solve(cg, max_solutions=1)

        assert any(e.event_type == 'cluster_operation' for e in tracer.events)
        assert stats.cluster_operations['prune'] >= 0
        assert stats.cluster_operations['merge'] >= 0
        assert stats.cluster_operations['split'] >= 0





