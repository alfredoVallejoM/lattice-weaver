"""
Tests unitarios para SimpleOptimizationSolver

Pruebas exhaustivas del solver de optimización simple.
"""

import pytest
from lattice_weaver.fibration import (
    SimpleOptimizationSolver,
    ConstraintHierarchy,
    Hardness
)


class TestSimpleOptimizationSolverInit:
    """Tests para la inicialización del solver."""
    
    def test_init_basic(self, simple_csp_problem):
        """Test: Inicialización básica del solver."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        
        assert solver.variables == variables
        assert solver.hierarchy == hierarchy
        assert solver.max_solutions == 10
        assert solver.nodes_explored == 0
        assert solver.solutions_found == 0
        assert solver.best_solution is None
        assert solver.best_energy == float('inf')


class TestSimpleOptimizationSolverSolve:
    """Tests para el método solve()."""
    
    def test_solve_trivial_problem(self, trivial_problem):
        """Test: Resolver problema trivial."""
        variables, domains, hierarchy = trivial_problem
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve()
        
        assert solution is not None
        assert solution == {"x": 1}
    
    def test_solve_simple_csp(self, simple_csp_problem):
        """Test: Resolver problema CSP simple."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve()
        
        assert solution is not None
        assert len(solution) == 3
        # Verificar all_different
        values = list(solution.values())
        assert len(values) == len(set(values))
    
    def test_solve_optimization_problem(self, optimization_problem):
        """Test: Resolver problema de optimización."""
        variables, domains, hierarchy = optimization_problem
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve()
        
        assert solution is not None
        assert len(solution) == 3
        # Verificar restricción HARD: a != b
        assert solution["a"] != solution["b"]
        # Debe minimizar la suma
        assert sum(solution.values()) <= 6
    
    def test_solve_with_max_nodes_limit(self, simple_csp_problem):
        """Test: Límite de nodos se respeta."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve(max_nodes=10)
        
        stats = solver.get_statistics()
        assert stats['nodes_explored'] <= 10
    
    def test_solve_finds_multiple_solutions(self, simple_csp_problem):
        """Test: Encuentra múltiples soluciones."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve()
        
        # Debe encontrar al menos una solución
        assert solution is not None
        assert solver.solutions_found >= 1
        assert len(solver.all_solutions) >= 1


class TestSimpleOptimizationSolverHeuristics:
    """Tests para heurísticas del solver."""
    
    def test_select_variable_mrv(self, simple_csp_problem):
        """Test: Selección de variable."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        
        assignment = {"x": 0}
        var = solver._select_variable_mrv(assignment)
        
        # Debe seleccionar una variable no asignada
        assert var in ["y", "z"]
    
    def test_is_consistent_hard(self, simple_csp_problem):
        """Test: Verificación de consistencia HARD."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        
        # Asignación consistente
        assignment1 = {"x": 0, "y": 1}
        assert solver._is_consistent_hard(assignment1) is True
        
        # Asignación inconsistente
        assignment2 = {"x": 0, "y": 0}
        assert solver._is_consistent_hard(assignment2) is False
    
    def test_check_hard_constraints(self, simple_csp_problem):
        """Test: Verificación completa de restricciones HARD."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        
        # Solución válida
        solution1 = {"x": 0, "y": 1, "z": 2}
        assert solver._check_hard_constraints(solution1) is True
        
        # Solución inválida
        solution2 = {"x": 0, "y": 0, "z": 2}
        assert solver._check_hard_constraints(solution2) is False


class TestSimpleOptimizationSolverStatistics:
    """Tests para estadísticas del solver."""
    
    def test_get_statistics(self, simple_csp_problem):
        """Test: Obtener estadísticas."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        solver.solve()
        
        stats = solver.get_statistics()
        
        assert 'nodes_explored' in stats
        assert 'solutions_found' in stats
        assert 'best_energy' in stats
        assert 'landscape_stats' in stats
        
        assert stats['nodes_explored'] > 0
        assert stats['solutions_found'] >= 0


class TestSimpleOptimizationSolverEdgeCases:
    """Tests para casos límite."""
    
    def test_no_variables(self):
        """Test: Problema sin variables."""
        variables = []
        domains = {}
        hierarchy = ConstraintHierarchy()
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve()
        
        # Solución vacía es válida
        assert solution == {}
    
    def test_unsolvable_problem(self, unsolvable_problem):
        """Test: Problema sin solución."""
        variables, domains, hierarchy = unsolvable_problem
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve()
        
        # No debe encontrar solución
        assert solution is None or solver.best_energy == float('inf')
    
    def test_soft_constraints_only(self, soft_constraints_problem):
        """Test: Problema con solo restricciones SOFT."""
        variables, domains, hierarchy = soft_constraints_problem
        
        solver = SimpleOptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve()
        
        assert solution is not None
        assert len(solution) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

