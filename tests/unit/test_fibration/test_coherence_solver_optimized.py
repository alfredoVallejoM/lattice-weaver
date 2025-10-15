"""
Tests unitarios para CoherenceSolverOptimized

Pruebas básicas del solver de coherencia optimizado.
"""

import pytest
from lattice_weaver.fibration import (
    CoherenceSolverOptimized,
    ConstraintHierarchy,
    Hardness
)


class TestCoherenceSolverOptimizedInit:
    """Tests para la inicialización del solver."""
    
    def test_init_basic(self, simple_csp_problem):
        """Test: Inicialización básica del solver."""
        variables, domains, _ = simple_csp_problem
        
        solver = CoherenceSolverOptimized(variables, domains)
        
        assert solver.variables == variables
        assert solver.nodes_explored == 0
        assert solver.nodes_pruned == 0
        assert solver.propagations == 0
        assert solver.conflicts_detected == 0


class TestCoherenceSolverOptimizedSolve:
    """Tests para el método solve()."""
    
    def test_solve_trivial_problem(self, trivial_problem):
        """Test: Resolver problema trivial."""
        variables, domains, _ = trivial_problem
        
        solver = CoherenceSolverOptimized(variables, domains)
        solution = solver.solve()
        
        assert solution is not None
        assert solution == {"x": 1}
    
    def test_solve_simple_problem(self):
        """Test: Resolver problema simple sin restricciones."""
        variables = ["x", "y"]
        domains = {"x": [1, 2], "y": [1, 2]}
        
        solver = CoherenceSolverOptimized(variables, domains)
        solution = solver.solve()
        
        # Sin restricciones, cualquier asignación es válida
        assert solution is not None
        assert len(solution) == 2
    
    def test_solve_with_max_nodes_limit(self):
        """Test: Límite de nodos se respeta."""
        variables = ["x", "y", "z"]
        domains = {"x": [1, 2, 3], "y": [1, 2, 3], "z": [1, 2, 3]}
        
        solver = CoherenceSolverOptimized(variables, domains)
        solution = solver.solve(max_nodes=5)
        
        # Puede o no encontrar solución con límite bajo
        assert solver.nodes_explored <= 5


class TestCoherenceSolverOptimizedEdgeCases:
    """Tests para casos límite."""
    
    def test_no_variables(self):
        """Test: Problema sin variables."""
        variables = []
        domains = {}
        
        solver = CoherenceSolverOptimized(variables, domains)
        solution = solver.solve()
        
        # Solución vacía es válida
        assert solution == {}
    
    def test_empty_domain(self):
        """Test: Dominio vacío."""
        variables = ["x"]
        domains = {"x": []}
        
        solver = CoherenceSolverOptimized(variables, domains)
        solution = solver.solve()
        
        # No debe encontrar solución
        assert solution is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

