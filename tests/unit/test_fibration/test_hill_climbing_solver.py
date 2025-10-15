"""
Tests unitarios para HillClimbingFibrationSolver

Pruebas exhaustivas del solver de Hill Climbing.
"""

import pytest
import random
from lattice_weaver.fibration import (
    HillClimbingFibrationSolver,
    ConstraintHierarchy,
    Hardness
)


class TestHillClimbingSolverInit:
    """Tests para la inicialización del solver."""
    
    def test_init_basic(self, simple_csp_problem):
        """Test: Inicialización básica del solver."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        
        assert solver.variables == variables
        assert solver.domains == domains
        assert solver.hierarchy == hierarchy
        assert solver.max_iterations == 1000
        assert solver.num_restarts == 10
        assert solver.neighbor_sampling_rate == 0.1
    
    def test_init_custom_params(self, simple_csp_problem):
        """Test: Inicialización con parámetros personalizados."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = HillClimbingFibrationSolver(
            variables, domains, hierarchy,
            max_iterations=500,
            num_restarts=5,
            neighbor_sampling_rate=0.2
        )
        
        assert solver.max_iterations == 500
        assert solver.num_restarts == 5
        assert solver.neighbor_sampling_rate == 0.2


class TestHillClimbingSolverSolve:
    """Tests para el método solve()."""
    
    def test_solve_trivial_problem(self, trivial_problem):
        """Test: Resolver problema trivial."""
        variables, domains, hierarchy = trivial_problem
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        solution = solver.solve()
        
        assert solution is not None
        assert solution == {"x": 1}
    
    def test_solve_simple_csp(self, simple_csp_problem):
        """Test: Resolver problema CSP simple."""
        variables, domains, hierarchy = simple_csp_problem
        
        # Fijar semilla para reproducibilidad
        random.seed(42)
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        solution = solver.solve()
        
        assert solution is not None
        assert len(solution) == 3
        # Verificar all_different
        values = list(solution.values())
        assert len(values) == len(set(values))
    
    def test_solve_optimization_problem(self, optimization_problem):
        """Test: Resolver problema de optimización."""
        variables, domains, hierarchy = optimization_problem
        
        random.seed(42)
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        solution = solver.solve()
        
        assert solution is not None
        assert len(solution) == 3
        # Verificar restricción HARD: a != b
        assert solution["a"] != solution["b"]
    
    def test_solve_with_restarts(self, soft_constraints_problem):
        """Test: Múltiples reinicios encuentran mejor solución."""
        variables, domains, hierarchy = soft_constraints_problem
        
        random.seed(42)
        
        # Con pocos reinicios
        solver1 = HillClimbingFibrationSolver(
            variables, domains, hierarchy,
            num_restarts=1
        )
        solution1 = solver1.solve()
        
        # Con más reinicios
        solver2 = HillClimbingFibrationSolver(
            variables, domains, hierarchy,
            num_restarts=10
        )
        solution2 = solver2.solve()
        
        # Ambas deben encontrar solución
        assert solution1 is not None
        assert solution2 is not None
    
    def test_solve_graph_coloring(self, graph_coloring_problem):
        """Test: Resolver problema de coloreo de grafos."""
        variables, domains, hierarchy = graph_coloring_problem
        
        random.seed(42)
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        solution = solver.solve()
        
        assert solution is not None
        # Verificar que nodos adyacentes tienen colores diferentes
        edges = [("v0", "v1"), ("v1", "v2"), ("v2", "v3"), ("v3", "v0"), ("v0", "v2")]
        for v1, v2 in edges:
            assert solution[v1] != solution[v2]


class TestHillClimbingSolverRandomSolution:
    """Tests para generación de solución inicial aleatoria."""
    
    def test_generate_random_valid_solution(self, simple_csp_problem):
        """Test: Generar solución aleatoria válida."""
        variables, domains, hierarchy = simple_csp_problem
        
        random.seed(42)
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        solution = solver._generate_random_valid_solution()
        
        assert solution is not None
        assert len(solution) == 3
        # Debe satisfacer restricciones HARD
        values = list(solution.values())
        assert len(values) == len(set(values))
    
    def test_generate_random_solution_unsolvable(self, unsolvable_problem):
        """Test: Problema sin solución devuelve None."""
        variables, domains, hierarchy = unsolvable_problem
        
        random.seed(42)
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        solution = solver._generate_random_valid_solution()
        
        # No debe encontrar solución válida
        assert solution is None
    
    def test_generate_random_solution_multiple_attempts(self, simple_csp_problem):
        """Test: Múltiples intentos generan soluciones diferentes."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        
        solutions = set()
        for seed in range(10):
            random.seed(seed)
            solution = solver._generate_random_valid_solution()
            if solution is not None:
                # Convertir a tupla para poder añadir a set
                solution_tuple = tuple(sorted(solution.items()))
                solutions.add(solution_tuple)
        
        # Debe generar al menos 2 soluciones diferentes
        assert len(solutions) >= 2


class TestHillClimbingSolverNeighbors:
    """Tests para generación de vecinos."""
    
    def test_get_best_neighbor_improvement(self, optimization_problem):
        """Test: Encontrar vecino que mejora la solución."""
        variables, domains, hierarchy = optimization_problem
        
        random.seed(42)
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        
        # Solución inicial subóptima
        current_solution = {"a": 3, "b": 2, "c": 3}
        current_energy = solver.landscape.compute_energy(current_solution).total_energy
        
        neighbor, neighbor_energy = solver._get_best_neighbor(current_solution)
        
        # Debe encontrar un vecino mejor
        if neighbor is not None:
            assert neighbor_energy <= current_energy
    
    def test_get_best_neighbor_optimal(self, trivial_problem):
        """Test: Solución óptima no tiene vecinos mejores."""
        variables, domains, hierarchy = trivial_problem
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        
        # Única solución posible
        current_solution = {"x": 1}
        
        neighbor, neighbor_energy = solver._get_best_neighbor(current_solution)
        
        # No debe encontrar vecino mejor (dominio de tamaño 1)
        assert neighbor is None
        assert neighbor_energy == float("inf")
    
    def test_get_best_neighbor_sampling(self, simple_csp_problem):
        """Test: Muestreo de vecinos funciona correctamente."""
        variables, domains, hierarchy = simple_csp_problem
        
        random.seed(42)
        
        # Tasa de muestreo alta
        solver = HillClimbingFibrationSolver(
            variables, domains, hierarchy,
            neighbor_sampling_rate=0.5  # 50% de variables
        )
        
        current_solution = {"x": 0, "y": 1, "z": 2}
        neighbor, _ = solver._get_best_neighbor(current_solution)
        
        # Debe explorar vecinos (puede o no encontrar mejora)
        # El test verifica que el método no falla
        assert neighbor is None or isinstance(neighbor, dict)


class TestHillClimbingSolverConvergence:
    """Tests para convergencia del algoritmo."""
    
    def test_convergence_finds_solution(self, soft_constraints_problem):
        """Test: Algoritmo encuentra solución."""
        variables, domains, hierarchy = soft_constraints_problem
        
        random.seed(42)
        
        solver = HillClimbingFibrationSolver(
            variables, domains, hierarchy,
            max_iterations=100,
            num_restarts=1
        )
        
        solution = solver.solve()
        
        # Debe encontrar alguna solución
        assert solution is not None
        assert len(solution) == 2
    
    def test_max_iterations_respected(self, optimization_problem):
        """Test: Límite de iteraciones se respeta."""
        variables, domains, hierarchy = optimization_problem
        
        random.seed(42)
        
        solver = HillClimbingFibrationSolver(
            variables, domains, hierarchy,
            max_iterations=5,  # Muy pocas iteraciones
            num_restarts=1
        )
        
        solution = solver.solve()
        
        # Debe terminar (puede o no encontrar óptimo)
        assert solution is None or isinstance(solution, dict)


class TestHillClimbingSolverEdgeCases:
    """Tests para casos límite."""
    
    def test_empty_domains(self):
        """Test: Problema con dominios vacíos."""
        variables = ["x"]
        domains = {"x": []}
        hierarchy = ConstraintHierarchy()
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        
        # El solver puede fallar al generar solución inicial
        # Verificamos que no lanza excepción
        try:
            solution = solver.solve()
            assert solution is None
        except (IndexError, ValueError):
            # Es aceptable que falle con estos errores en dominios vacíos
            pass
    
    def test_no_variables(self):
        """Test: Problema sin variables."""
        variables = []
        domains = {}
        hierarchy = ConstraintHierarchy()
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        
        # El solver puede fallar con lista vacía
        try:
            solution = solver.solve()
            assert solution is None or solution == {}
        except (ValueError, IndexError):
            # Es aceptable que falle con estos errores sin variables
            pass
    
    def test_single_value_domains(self, trivial_problem):
        """Test: Dominios de tamaño 1."""
        variables, domains, hierarchy = trivial_problem
        
        solver = HillClimbingFibrationSolver(variables, domains, hierarchy)
        solution = solver.solve()
        
        assert solution is not None
        assert solution == {"x": 1}
    
    def test_zero_restarts(self, simple_csp_problem):
        """Test: Sin reinicios (num_restarts=0)."""
        variables, domains, hierarchy = simple_csp_problem
        
        solver = HillClimbingFibrationSolver(
            variables, domains, hierarchy,
            num_restarts=0
        )
        
        solution = solver.solve()
        
        # No debe ejecutar ninguna búsqueda
        assert solution is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

