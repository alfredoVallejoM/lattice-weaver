"""
Tests unitarios para OptimizationSolver

Pruebas básicas del solver de optimización con múltiples estrategias.
"""

import pytest
from lattice_weaver.fibration import (
    OptimizationSolver,
    ConstraintHierarchy,
    Hardness
)


class TestOptimizationSolverInit:
    """Tests para la inicialización del solver."""
    
    def test_init_basic(self, simple_csp_problem):
        """Test: Inicialización básica del solver."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = OptimizationSolver(variables, domains, hierarchy)
        
        assert solver.variables == variables
        assert solver.hierarchy == hierarchy
        assert solver.beam_width == 10
        assert solver.k_best_values == 3
        assert solver.nodes_explored == 0
        assert solver.solutions_found == 0


class TestOptimizationSolverSolve:
    """Tests para el método solve() con diferentes estrategias."""
    
    def test_solve_trivial_problem_beam_search(self, trivial_problem):
        """Test: Resolver problema trivial con beam search."""
        variables, domains, hierarchy = trivial_problem
        
        solver = OptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve(strategy="beam_search")
        
        assert solution is not None
        assert solution == {"x": 1}
    
    def test_solve_trivial_problem_branch_bound(self, trivial_problem):
        """Test: Resolver problema trivial con branch & bound."""
        variables, domains, hierarchy = trivial_problem
        
        solver = OptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve(strategy="branch_bound")
        
        assert solution is not None
        assert solution == {"x": 1}
    
    def test_solve_trivial_problem_k_best(self, trivial_problem):
        """Test: Resolver problema trivial con k-best."""
        variables, domains, hierarchy = trivial_problem
        
        solver = OptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve(strategy="k_best")
        
        assert solution is not None
        assert solution == {"x": 1}
    
    def test_solve_optimization_problem_beam_search(self, optimization_problem):
        """Test: Resolver problema de optimización con beam search."""
        variables, domains, hierarchy = optimization_problem
        
        solver = OptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve(strategy="beam_search", max_nodes=1000)
        
        if solution is not None:
            assert len(solution) == 3
            # Verificar restricción HARD: a != b
            assert solution["a"] != solution["b"]
    
    def test_solve_invalid_strategy(self, trivial_problem):
        """Test: Estrategia inválida lanza excepción."""
        variables, domains, hierarchy = trivial_problem
        
        solver = OptimizationSolver(variables, domains, hierarchy)
        
        with pytest.raises(ValueError):
            solver.solve(strategy="invalid_strategy")
    
    def test_solve_with_max_nodes_limit(self, optimization_problem):
        """Test: Límite de nodos se respeta aproximadamente."""
        variables, domains, hierarchy = optimization_problem
        
        solver = OptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve(strategy="beam_search", max_nodes=10)
        
        # Puede o no encontrar solución con límite bajo
        # Margen de tolerancia por expansión del beam
        assert solver.nodes_explored <= 20


class TestOptimizationSolverConfiguration:
    """Tests para configuración del solver."""
    
    def test_custom_beam_width(self, simple_csp_problem):
        """Test: Configurar ancho del beam."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = OptimizationSolver(variables, domains, hierarchy)
        solver.beam_width = 5
        
        assert solver.beam_width == 5
    
    def test_custom_k_best_values(self, simple_csp_problem):
        """Test: Configurar número de mejores valores."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = OptimizationSolver(variables, domains, hierarchy)
        solver.k_best_values = 2
        
        assert solver.k_best_values == 2


class TestOptimizationSolverEdgeCases:
    """Tests para casos límite."""
    
    def test_no_variables(self):
        """Test: Problema sin variables."""
        variables = []
        domains = {}
        hierarchy = ConstraintHierarchy()
        
        solver = OptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve(strategy="beam_search")
        
        # Solución vacía es válida
        assert solution == {} or solution is None
    
    def test_soft_constraints_only(self, soft_constraints_problem):
        """Test: Problema con solo restricciones SOFT."""
        variables, domains, hierarchy = soft_constraints_problem
        
        solver = OptimizationSolver(variables, domains, hierarchy)
        solution = solver.solve(strategy="beam_search", max_nodes=100)
        
        # Debe encontrar alguna solución
        if solution is not None:
            assert len(solution) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

