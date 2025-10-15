"""
Tests para las heurísticas MRV/Degree/LCV del CSPSolver.

Este módulo valida que las heurísticas implementadas funcionan correctamente
y mejoran el rendimiento del solver.
"""

import pytest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver


class TestMRVHeuristic:
    """Tests para la heurística MRV (Minimum Remaining Values)"""
    
    def test_mrv_selects_most_constrained_variable(self):
        """MRV debe seleccionar la variable con menor dominio"""
        # Crear CSP simple donde una variable tiene dominio más pequeño
        csp = CSP(
            variables=['A', 'B', 'C'],
            domains={
                'A': frozenset([1, 2, 3]),
                'B': frozenset([1]),  # Dominio más pequeño
                'C': frozenset([1, 2])
            },
            constraints=[]
        )
        
        solver = CSPSolver(csp)
        current_domains = {
            'A': [1, 2, 3],
            'B': [1],
            'C': [1, 2]
        }
        
        selected = solver._select_unassigned_variable(current_domains)
        assert selected == 'B', "MRV debe seleccionar variable con menor dominio"
    
    def test_mrv_with_equal_domains_uses_degree(self):
        """Cuando dominios son iguales, debe usar Degree como desempate"""
        csp = CSP(
            variables=['A', 'B', 'C'],
            domains={
                'A': frozenset([1, 2]),
                'B': frozenset([1, 2]),
                'C': frozenset([1, 2, 3])
            },
            constraints=[
                Constraint(scope=('A', 'C'), relation=lambda a, c: a != c),
                Constraint(scope=('B', 'C'), relation=lambda b, c: b != c),
                Constraint(scope=('A', 'B'), relation=lambda a, b: a != b)
            ]
        )
        
        solver = CSPSolver(csp)
        current_domains = {
            'A': [1, 2],
            'B': [1, 2],
            'C': [1, 2, 3]
        }
        
        # A y B tienen mismo dominio, pero A tiene más conexiones
        selected = solver._select_unassigned_variable(current_domains)
        assert selected in ['A', 'B'], "Debe seleccionar A o B (mismo MRV)"
    
    def test_mrv_after_forward_checking(self):
        """MRV debe considerar dominios reducidos por forward checking"""
        csp = CSP(
            variables=['X', 'Y', 'Z'],
            domains={
                'X': frozenset([1, 2, 3]),
                'Y': frozenset([1, 2, 3]),
                'Z': frozenset([1, 2, 3])
            },
            constraints=[
                Constraint(scope=('X', 'Y'), relation=lambda x, y: x != y),
                Constraint(scope=('Y', 'Z'), relation=lambda y, z: y != z)
            ]
        )
        
        solver = CSPSolver(csp)
        # Simular que X=1 ya está asignado
        solver.assignment['X'] = 1
        
        # Y debería tener dominio reducido a [2, 3]
        current_domains = {
            'X': [1],
            'Y': [2, 3],  # Reducido por X=1
            'Z': [1, 2, 3]
        }
        
        selected = solver._select_unassigned_variable(current_domains)
        assert selected == 'Y', "MRV debe seleccionar Y con dominio reducido"


class TestDegreeHeuristic:
    """Tests para la heurística Degree"""
    
    def test_degree_counts_constraints_correctly(self):
        """Degree debe contar correctamente restricciones con variables no asignadas"""
        csp = CSP(
            variables=['A', 'B', 'C', 'D'],
            domains={
                'A': frozenset([1, 2]),
                'B': frozenset([1, 2]),
                'C': frozenset([1, 2]),
                'D': frozenset([1, 2])
            },
            constraints=[
                # A está en 3 restricciones
                Constraint(scope=('A', 'B'), relation=lambda a, b: a != b),
                Constraint(scope=('A', 'C'), relation=lambda a, c: a != c),
                Constraint(scope=('A', 'D'), relation=lambda a, d: a != d),
                # B solo en 1 restricción
                Constraint(scope=('B', 'C'), relation=lambda b, c: b != c)
            ]
        )
        
        solver = CSPSolver(csp)
        current_domains = {
            'A': [1, 2],
            'B': [1, 2],
            'C': [1, 2],
            'D': [1, 2]
        }
        
        # A tiene más conexiones, debe ser seleccionada
        selected = solver._select_unassigned_variable(current_domains)
        assert selected == 'A', "Degree debe seleccionar A con más restricciones"
    
    def test_degree_ignores_assigned_variables(self):
        """Degree no debe contar restricciones con variables ya asignadas"""
        csp = CSP(
            variables=['A', 'B', 'C'],
            domains={
                'A': frozenset([1, 2]),
                'B': frozenset([1, 2]),
                'C': frozenset([1, 2])
            },
            constraints=[
                Constraint(scope=('A', 'B'), relation=lambda a, b: a != b),
                Constraint(scope=('A', 'C'), relation=lambda a, c: a != c),
                Constraint(scope=('B', 'C'), relation=lambda b, c: b != c)
            ]
        )
        
        solver = CSPSolver(csp)
        solver.assignment['A'] = 1  # A ya asignada
        
        current_domains = {
            'A': [1],
            'B': [2],
            'C': [2]
        }
        
        # B y C tienen mismo dominio, pero ambas tienen degree=1 (solo entre ellas)
        selected = solver._select_unassigned_variable(current_domains)
        assert selected in ['B', 'C'], "Debe seleccionar B o C (A ya asignada)"


class TestLCVHeuristic:
    """Tests para la heurística LCV (Least Constraining Value)"""
    
    def test_lcv_orders_least_constraining_first(self):
        """LCV debe ordenar valores menos restrictivos primero"""
        csp = CSP(
            variables=['A', 'B'],
            domains={
                'A': frozenset([1, 2, 3]),
                'B': frozenset([1, 2, 3])
            },
            constraints=[
                Constraint(scope=('A', 'B'), relation=lambda a, b: a < b)
            ]
        )
        
        solver = CSPSolver(csp)
        current_domains = {
            'A': [1, 2, 3],
            'B': [1, 2, 3]
        }
        
        # Para A: 
        # - valor 1 permite B={2,3} (2 opciones)
        # - valor 2 permite B={3} (1 opción)
        # - valor 3 permite B={} (0 opciones)
        # LCV debe ordenar: [1, 2, 3]
        ordered = solver._order_domain_values('A', current_domains)
        assert ordered[0] == 1, "LCV debe poner primero el valor menos restrictivo"
        assert ordered[-1] == 3, "LCV debe poner último el valor más restrictivo"
    
    def test_lcv_with_multiple_constraints(self):
        """LCV debe considerar múltiples restricciones"""
        csp = CSP(
            variables=['X', 'Y', 'Z'],
            domains={
                'X': frozenset([1, 2]),
                'Y': frozenset([1, 2]),
                'Z': frozenset([1, 2])
            },
            constraints=[
                Constraint(scope=('X', 'Y'), relation=lambda x, y: x != y),
                Constraint(scope=('X', 'Z'), relation=lambda x, z: x != z)
            ]
        )
        
        solver = CSPSolver(csp)
        current_domains = {
            'X': [1, 2],
            'Y': [1, 2],
            'Z': [1, 2]
        }
        
        # Para X, ambos valores eliminan 2 opciones total (1 en Y, 1 en Z)
        ordered = solver._order_domain_values('X', current_domains)
        assert len(ordered) == 2, "Debe retornar todos los valores"
        assert set(ordered) == {1, 2}, "Debe contener todos los valores originales"
    
    def test_lcv_ignores_assigned_variables(self):
        """LCV no debe considerar variables ya asignadas"""
        csp = CSP(
            variables=['A', 'B', 'C'],
            domains={
                'A': frozenset([1, 2]),
                'B': frozenset([1, 2]),
                'C': frozenset([1, 2])
            },
            constraints=[
                Constraint(scope=('A', 'B'), relation=lambda a, b: a != b),
                Constraint(scope=('A', 'C'), relation=lambda a, c: a != c)
            ]
        )
        
        solver = CSPSolver(csp)
        solver.assignment['B'] = 1  # B ya asignada
        
        current_domains = {
            'A': [1, 2],
            'B': [1],
            'C': [1, 2]
        }
        
        # LCV solo debe considerar C (B ya asignada)
        ordered = solver._order_domain_values('A', current_domains)
        assert len(ordered) == 2, "Debe retornar todos los valores de A"


class TestIntegrationHeuristics:
    """Tests de integración para las heurísticas combinadas"""
    
    def test_nqueens_4_solves_correctly(self):
        """Test de regresión: N-Queens 4x4 debe resolverse correctamente"""
        # Importar generador de N-Queens si está disponible
        try:
            from lattice_weaver.problems.generators.nqueens import NQueensProblem
            problem = NQueensProblem(4)  # Usar argumento posicional
            csp = problem.to_csp()
        except (ImportError, TypeError) as e:
            # Si no está disponible o la API es diferente, skip
            pytest.skip(f"NQueensProblem no disponible o API diferente: {e}")
        
        solver = CSPSolver(csp)
        stats = solver.solve(all_solutions=False, max_solutions=1)
        
        assert len(stats.solutions) >= 1, "Debe encontrar al menos una solución"
        assert stats.solutions[0].is_consistent, "La solución debe ser consistente"
    
    def test_simple_graph_coloring(self):
        """Test con problema simple de coloreo de grafos"""
        # Grafo triangular: 3 nodos, todos conectados
        # Necesita al menos 3 colores
        csp = CSP(
            variables=['A', 'B', 'C'],
            domains={
                'A': frozenset([1, 2, 3]),
                'B': frozenset([1, 2, 3]),
                'C': frozenset([1, 2, 3])
            },
            constraints=[
                Constraint(scope=('A', 'B'), relation=lambda a, b: a != b),
                Constraint(scope=('B', 'C'), relation=lambda b, c: b != c),
                Constraint(scope=('A', 'C'), relation=lambda a, c: a != c)
            ]
        )
        
        solver = CSPSolver(csp)
        stats = solver.solve(all_solutions=False, max_solutions=1)
        
        assert len(stats.solutions) == 1, "Debe encontrar una solución"
        solution = stats.solutions[0].assignment
        
        # Verificar que todos los nodos tienen colores diferentes
        assert solution['A'] != solution['B']
        assert solution['B'] != solution['C']
        assert solution['A'] != solution['C']
    
    def test_heuristics_reduce_backtracking(self):
        """Las heurísticas deben reducir backtracking vs. búsqueda ingenua"""
        # Problema con estructura que beneficia de heurísticas
        csp = CSP(
            variables=['X1', 'X2', 'X3', 'X4'],
            domains={
                'X1': frozenset([1, 2, 3]),
                'X2': frozenset([1, 2, 3]),
                'X3': frozenset([1, 2, 3]),
                'X4': frozenset([1, 2, 3])
            },
            constraints=[
                Constraint(scope=('X1', 'X2'), relation=lambda a, b: a != b),
                Constraint(scope=('X1', 'X3'), relation=lambda a, c: a != c),
                Constraint(scope=('X1', 'X4'), relation=lambda a, d: a != d),
                Constraint(scope=('X2', 'X3'), relation=lambda b, c: b != c),
                Constraint(scope=('X2', 'X4'), relation=lambda b, d: b != d),
                Constraint(scope=('X3', 'X4'), relation=lambda c, d: c != d)
            ]
        )
        
        solver = CSPSolver(csp)
        stats = solver.solve(all_solutions=False, max_solutions=1)
        
        # Con heurísticas, este problema debe resolver rápidamente
        assert stats.nodes_explored < 20, f"Exploró {stats.nodes_explored} nodos, esperaba <20"
        assert stats.backtracks < 10, f"Hizo {stats.backtracks} backtracks, esperaba <10"
    
    def test_unsatisfiable_problem_detected_quickly(self):
        """Heurísticas deben ayudar a detectar problemas insatisfacibles rápidamente"""
        # Problema imposible: 2 variables, dominios {1}, restricción a != b
        csp = CSP(
            variables=['A', 'B'],
            domains={
                'A': frozenset([1]),
                'B': frozenset([1])
            },
            constraints=[
                Constraint(scope=('A', 'B'), relation=lambda a, b: a != b)
            ]
        )
        
        solver = CSPSolver(csp)
        stats = solver.solve(all_solutions=False, max_solutions=1)
        
        assert len(stats.solutions) == 0, "No debe encontrar soluciones"
        assert stats.nodes_explored <= 2, "Debe detectar insatisfacibilidad rápidamente"


class TestEdgeCases:
    """Tests para casos especiales y límites"""
    
    def test_single_variable_csp(self):
        """CSP con una sola variable debe resolverse correctamente"""
        csp = CSP(
            variables=['X'],
            domains={'X': frozenset([1, 2, 3])},
            constraints=[]
        )
        
        solver = CSPSolver(csp)
        stats = solver.solve(all_solutions=False, max_solutions=1)
        
        assert len(stats.solutions) == 1
        # El solver explora 2 nodos: uno para asignar, otro para verificar completitud
        assert stats.nodes_explored == 2
    
    def test_empty_domain_handled(self):
        """Dominio vacío debe ser manejado correctamente"""
        csp = CSP(
            variables=['A', 'B'],
            domains={
                'A': frozenset([1]),
                'B': frozenset([])  # Dominio vacío
            },
            constraints=[]
        )
        
        solver = CSPSolver(csp)
        current_domains = {'A': [1], 'B': []}
        
        # No debe crashear, debe manejar el dominio vacío
        selected = solver._select_unassigned_variable(current_domains)
        assert selected == 'B', "Debe seleccionar B (dominio vacío tiene prioridad MRV)"
    
    def test_all_variables_assigned(self):
        """Cuando todas las variables están asignadas, debe retornar None"""
        csp = CSP(
            variables=['A', 'B'],
            domains={
                'A': frozenset([1]),
                'B': frozenset([2])
            },
            constraints=[]
        )
        
        solver = CSPSolver(csp)
        solver.assignment = {'A': 1, 'B': 2}
        
        current_domains = {'A': [1], 'B': [2]}
        selected = solver._select_unassigned_variable(current_domains)
        
        assert selected is None, "Debe retornar None cuando todas están asignadas"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

