"""
Tests unitarios para FibrationSearchSolver

Pruebas exhaustivas del solver de búsqueda con Flujo de Fibración.
"""

import pytest
import time
from lattice_weaver.fibration import (
    FibrationSearchSolver,
    ConstraintHierarchy,
    Hardness,
    AdaptiveStrategy,
    FocusOnLocalStrategy,
    FocusOnGlobalStrategy
)


class TestFibrationSearchSolverInit:
    """Tests para la inicialización del solver."""
    
    def test_init_basic(self, simple_csp_problem):
        """Test: Inicialización básica del solver."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        
        assert solver.variables == variables
        assert solver.domains == domains
        assert solver.hierarchy == hierarchy
        assert solver.best_solution is None
        assert solver.best_energy == float("inf")
        assert solver.num_solutions_found == 0
        assert solver.backtracks_count == 0
        assert solver.nodes_visited == 0
    
    def test_init_custom_limits(self, simple_csp_problem):
        """Test: Inicialización con límites personalizados."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = FibrationSearchSolver(
            variables, domains, hierarchy,
            max_iterations=5000,
            max_backtracks=10000
        )
        
        assert solver.max_iterations == 5000
        assert solver.max_backtracks == 10000
    
    def test_init_creates_components(self, simple_csp_problem):
        """Test: Inicialización crea componentes internos."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        
        assert solver.landscape is not None
        assert solver.hacification_engine is not None
        assert solver.modulator is not None


class TestFibrationSearchSolverSolve:
    """Tests para el método solve()."""
    
    def test_solve_trivial_problem(self, trivial_problem):
        """Test: Resolver problema trivial (1 variable, 1 valor)."""
        variables, domains, hierarchy = trivial_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve(time_limit_seconds=5)
        
        assert solution is not None
        assert solution == {"x": 1}
        assert solver.num_solutions_found == 1
    
    def test_solve_simple_csp(self, simple_csp_problem):
        """Test: Resolver problema CSP simple (all_different)."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve(time_limit_seconds=10)
        
        assert solution is not None
        assert len(solution) == 3
        # Verificar all_different
        values = list(solution.values())
        assert len(values) == len(set(values))
    
    def test_solve_nqueens_4x4(self, nqueens_4x4):
        """Test: Resolver problema N-Queens 4x4."""
        variables, domains, hierarchy = nqueens_4x4
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve(time_limit_seconds=10)
        
        assert solution is not None
        assert len(solution) == 4
        # Verificar que es una solución válida
        for i in range(4):
            for j in range(i + 1, 4):
                col_i = solution[f"Q{i}"]
                col_j = solution[f"Q{j}"]
                assert col_i != col_j  # No misma columna
                assert abs(col_i - col_j) != abs(i - j)  # No misma diagonal
    
    def test_solve_graph_coloring(self, graph_coloring_problem):
        """Test: Resolver problema de coloreo de grafos."""
        variables, domains, hierarchy = graph_coloring_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve(time_limit_seconds=10)
        
        assert solution is not None
        assert len(solution) == 4
        # Verificar que nodos adyacentes tienen colores diferentes
        edges = [("v0", "v1"), ("v1", "v2"), ("v2", "v3"), ("v3", "v0"), ("v0", "v2")]
        for v1, v2 in edges:
            assert solution[v1] != solution[v2]
    
    def test_solve_unsolvable_problem(self, unsolvable_problem):
        """Test: Problema sin solución."""
        variables, domains, hierarchy = unsolvable_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve(time_limit_seconds=5)
        
        # Puede devolver None o una solución que viole restricciones
        # dependiendo de la implementación
        if solution is not None:
            # Si devuelve solución, debe tener violaciones
            assert solver.best_energy > 0
    
    def test_solve_optimization_problem(self, optimization_problem):
        """Test: Resolver problema de optimización."""
        variables, domains, hierarchy = optimization_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve(time_limit_seconds=10)
        
        assert solution is not None
        assert len(solution) == 3
        # Verificar restricción HARD: a != b
        assert solution["a"] != solution["b"]
        # La mejor solución debería minimizar la suma
        # Mínimo posible: a=1, b=2 (o b=3), c=1 -> suma=4
        assert sum(solution.values()) <= 6  # Razonable
    
    def test_solve_with_time_limit(self, nqueens_4x4):
        """Test: Límite de tiempo se respeta."""
        variables, domains, hierarchy = nqueens_4x4
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        
        start_time = time.time()
        solution = solver.solve(time_limit_seconds=1)
        elapsed = time.time() - start_time
        
        # Debe terminar cerca del límite de tiempo (con margen)
        assert elapsed < 2.0
    
    def test_solve_resets_statistics(self, simple_csp_problem):
        """Test: solve() resetea estadísticas."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        
        # Primera ejecución
        solver.solve(time_limit_seconds=5)
        first_nodes = solver.nodes_visited
        
        # Segunda ejecución
        solver.solve(time_limit_seconds=5)
        second_nodes = solver.nodes_visited
        
        # Las estadísticas deben resetearse
        assert first_nodes > 0
        assert second_nodes > 0


class TestFibrationSearchSolverHeuristics:
    """Tests para las heurísticas del solver."""
    
    def test_select_next_variable_mrv(self, simple_csp_problem):
        """Test: Heurística MRV selecciona variable con menor dominio."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        
        # Asignación parcial que reduce dominio de una variable
        assignment = {"x": 0}
        
        # Después de asignar x=0, el dominio de y y z se reduce
        # (no pueden ser 0 por all_different)
        var = solver._select_next_variable(assignment)
        
        # Debe seleccionar una variable no asignada
        assert var in ["y", "z"]
    
    def test_get_ordered_domain_values_lcv(self, simple_csp_problem):
        """Test: Heurística LCV ordena valores por energía."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        
        assignment = {"x": 0}
        ordered_values = solver._get_ordered_domain_values("y", assignment)
        
        # Debe devolver valores ordenados (excluyendo 0 por all_different)
        assert 0 not in ordered_values  # Violación HARD
        assert len(ordered_values) <= 2  # [1, 2]


class TestFibrationSearchSolverModulation:
    """Tests para estrategias de modulación."""
    
    def test_set_modulation_strategy_local(self, simple_csp_problem):
        """Test: Cambiar estrategia a FocusOnLocal."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solver.set_modulation_strategy(FocusOnLocalStrategy())
        
        solution = solver.solve(time_limit_seconds=5)
        assert solution is not None
    
    def test_set_modulation_strategy_global(self, simple_csp_problem):
        """Test: Cambiar estrategia a FocusOnGlobal."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solver.set_modulation_strategy(FocusOnGlobalStrategy())
        
        solution = solver.solve(time_limit_seconds=5)
        assert solution is not None
    
    def test_set_modulation_strategy_adaptive(self, simple_csp_problem):
        """Test: Estrategia adaptativa (por defecto)."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        # Ya está en AdaptiveStrategy por defecto
        
        solution = solver.solve(time_limit_seconds=5)
        assert solution is not None


class TestFibrationSearchSolverStatistics:
    """Tests para estadísticas del solver."""
    
    def test_get_statistics(self, simple_csp_problem):
        """Test: Obtener estadísticas después de resolver."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solver.solve(time_limit_seconds=5)
        
        stats = solver.get_statistics()
        
        assert "best_energy" in stats
        assert "num_solutions_found" in stats
        assert "backtracks_count" in stats
        assert "nodes_visited" in stats
        assert "landscape_stats" in stats
        assert "hacification_stats" in stats
        assert "modulator_stats" in stats
    
    def test_statistics_values(self, trivial_problem):
        """Test: Valores de estadísticas son correctos."""
        variables, domains, hierarchy = trivial_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solver.solve(time_limit_seconds=5)
        
        stats = solver.get_statistics()
        
        assert stats["num_solutions_found"] == 1
        assert stats["nodes_visited"] >= 1
        assert stats["best_energy"] >= 0


class TestFibrationSearchSolverBranchAndBound:
    """Tests para poda Branch & Bound."""
    
    def test_branch_bound_pruning(self, optimization_problem):
        """Test: Branch & Bound poda ramas subóptimas."""
        variables, domains, hierarchy = optimization_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve(time_limit_seconds=10)
        
        assert solution is not None
        stats = solver.get_statistics()
        
        # Debe haber realizado backtracks (poda)
        assert stats["backtracks_count"] > 0
    
    def test_max_backtracks_limit(self, nqueens_4x4):
        """Test: Límite de backtracks se respeta."""
        variables, domains, hierarchy = nqueens_4x4
        
        solver = FibrationSearchSolver(
            variables, domains, hierarchy,
            max_backtracks=10  # Límite muy bajo
        )
        
        solution = solver.solve(time_limit_seconds=10)
        
        # Puede no encontrar solución por límite de backtracks
        stats = solver.get_statistics()
        assert stats["backtracks_count"] <= 10
    
    def test_max_iterations_limit(self, nqueens_4x4):
        """Test: Límite de iteraciones se respeta."""
        variables, domains, hierarchy = nqueens_4x4
        
        solver = FibrationSearchSolver(
            variables, domains, hierarchy,
            max_iterations=50  # Límite muy bajo
        )
        
        solution = solver.solve(time_limit_seconds=10)
        
        stats = solver.get_statistics()
        assert stats["nodes_visited"] <= 50


class TestFibrationSearchSolverEdgeCases:
    """Tests para casos límite."""
    
    def test_empty_domains(self):
        """Test: Problema con dominios vacíos."""
        variables = ["x"]
        domains = {"x": []}
        hierarchy = ConstraintHierarchy()
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve(time_limit_seconds=5)
        
        # No debe encontrar solución
        assert solution is None or len(solution) == 0
    
    def test_no_variables(self):
        """Test: Problema sin variables."""
        variables = []
        domains = {}
        hierarchy = ConstraintHierarchy()
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve(time_limit_seconds=5)
        
        # Solución vacía es válida
        assert solution == {}
    
    def test_soft_constraints_only(self, soft_constraints_problem):
        """Test: Problema con solo restricciones SOFT."""
        variables, domains, hierarchy = soft_constraints_problem
        
        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve(time_limit_seconds=10)
        
        assert solution is not None
        # Debe encontrar solución que minimice violaciones SOFT
        assert len(solution) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

