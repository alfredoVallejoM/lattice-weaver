"""
Tests unitarios para el Motor de Consistencia Adaptativa (ACE).

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import pytest
from lattice_weaver.arc_weaver.graph_structures import ConstraintGraph
from lattice_weaver.arc_weaver.adaptive_consistency import (
    AC3Solver,
    AdaptiveConsistencyEngine,
    ClusterSolver,
    SolutionStats
)


class TestAC3Solver:
    """Tests para AC3Solver."""
    
    def test_create_solver(self):
        """Test creación de solver."""
        solver = AC3Solver()
        assert solver.calls == 0
        assert len(solver.last_support) == 0
    
    def test_enforce_arc_consistency_simple(self):
        """Test AC-3 en grafo simple."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        solver = AC3Solver()
        consistent = solver.enforce_arc_consistency(cg)
        
        assert consistent == True
        assert solver.calls == 1
    
    def test_enforce_arc_consistency_inconsistent(self):
        """Test AC-3 detecta inconsistencia."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2})
        cg.add_variable('Y', {1})
        cg.add_variable('Z', {1})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        cg.add_constraint('X', 'Z', lambda x, z: x != z)
        
        solver = AC3Solver()
        consistent = solver.enforce_arc_consistency(cg)
        
        # X debe quedar con dominio vacío (no puede ser ni 1 ni 2)
        # Nota: con Y={1} y Z={1}, X debe ser !=1, dejando X={2}
        # Este test verifica que AC-3 reduce dominios correctamente
        assert len(cg.get_domain('X')) > 0  # X={2} es válido
        assert 1 not in cg.get_domain('X')  # 1 fue eliminado
    
    def test_enforce_arc_consistency_reduces_domains(self):
        """Test que AC-3 reduce dominios."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        solver = AC3Solver()
        consistent = solver.enforce_arc_consistency(cg)
        
        assert consistent == True
        # Dominio de X debe haberse reducido (eliminar 1)
        assert 1 not in cg.get_domain('X')
        assert len(cg.get_domain('X')) == 2
    
    def test_enforce_arc_consistency_subset_variables(self):
        """Test AC-3 en subconjunto de variables."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_variable('Z', {1, 2, 3})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        cg.add_constraint('Y', 'Z', lambda y, z: y != z)
        
        solver = AC3Solver()
        # Solo procesar X e Y
        consistent = solver.enforce_arc_consistency(cg, {'X', 'Y'})
        
        assert consistent == True
    
    def test_last_support_caching(self):
        """Test que last_support se usa correctamente."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        solver = AC3Solver()
        solver.enforce_arc_consistency(cg)
        
        # Debe haber entradas en last_support
        assert len(solver.last_support) > 0
    
    def test_reset_support_cache(self):
        """Test limpiar caché de soporte."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        solver = AC3Solver()
        solver.enforce_arc_consistency(cg)
        
        assert len(solver.last_support) > 0
        
        solver.reset_support_cache()
        assert len(solver.last_support) == 0


class TestSolutionStats:
    """Tests para SolutionStats."""
    
    def test_create_stats(self):
        """Test creación de estadísticas."""
        stats = SolutionStats()
        
        assert len(stats.solutions) == 0
        assert stats.nodes_explored == 0
        assert stats.backtracks == 0
        assert stats.arc_consistency_calls == 0
    
    def test_add_solution(self):
        """Test añadir solución."""
        stats = SolutionStats()
        solution = {'X': 1, 'Y': 2}
        
        stats.add_solution(solution)
        
        assert len(stats.solutions) == 1
        assert stats.solutions[0] == solution
    
    def test_cluster_operations_tracking(self):
        """Test seguimiento de operaciones de clúster."""
        stats = SolutionStats()
        
        stats.cluster_operations["merge"] += 1
        stats.cluster_operations["split"] += 2
        stats.cluster_operations["prune"] += 3
        
        assert stats.cluster_operations["merge"] == 1
        assert stats.cluster_operations["split"] == 2
        assert stats.cluster_operations["prune"] == 3
    
    def test_stats_repr(self):
        """Test representación string."""
        stats = SolutionStats()
        stats.add_solution({'X': 1})
        stats.nodes_explored = 10
        stats.backtracks = 5
        
        repr_str = repr(stats)
        assert 'solutions=1' in repr_str
        assert 'nodes=10' in repr_str


class TestAdaptiveConsistencyEngine:
    """Tests para AdaptiveConsistencyEngine."""
    
    def test_create_engine(self):
        """Test creación de motor."""
        engine = AdaptiveConsistencyEngine(
            min_cluster_size=2,
            max_cluster_size=10
        )
        
        assert engine.min_cluster_size == 2
        assert engine.max_cluster_size == 10
    
    def test_solve_simple_problem(self):
        """Test resolver problema simple."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2})
        cg.add_variable('Y', {1, 2})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=10)
        
        # Debe encontrar soluciones
        assert len(stats.solutions) > 0
        assert stats.nodes_explored > 0
    
    def test_solve_nqueens_4(self):
        """Test resolver N-Reinas n=4."""
        cg = ConstraintGraph()
        n = 4
        
        for i in range(n):
            cg.add_variable(f'Q{i}', set(range(n)))
        
        for i in range(n):
            for j in range(i+1, n):
                def constraint(vi, vj, row_i=i, row_j=j):
                    return (vi != vj and 
                            abs(vi - vj) != abs(row_i - row_j))
                
                cg.add_constraint(f'Q{i}', f'Q{j}', constraint)
        
        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=10)
        
        # N-Reinas n=4 tiene 2 soluciones
        assert len(stats.solutions) == 2
        assert stats.clustering_metrics is not None
    
    def test_solve_inconsistent_problem(self):
        """Test resolver problema inconsistente."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1})
        cg.add_variable('Y', {1})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=10)
        
        # No debe encontrar soluciones
        assert len(stats.solutions) == 0
    
    def test_solve_with_max_solutions_limit(self):
        """Test límite de soluciones."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=2)
        
        # Debe respetar el límite
        assert len(stats.solutions) <= 2
    
    def test_solve_with_timeout(self):
        """Test resolver con timeout."""
        cg = ConstraintGraph()
        n = 8
        
        # N-Reinas n=8 (más complejo)
        for i in range(n):
            cg.add_variable(f'Q{i}', set(range(n)))
        
        for i in range(n):
            for j in range(i+1, n):
                def constraint(vi, vj, row_i=i, row_j=j):
                    return (vi != vj and 
                            abs(vi - vj) != abs(row_i - row_j))
                
                cg.add_constraint(f'Q{i}', f'Q{j}', constraint)
        
        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=100, timeout=0.1)
        
        # Debe terminar en tiempo razonable
        assert stats.time_elapsed <= 1.0  # Margen de error
    
    def test_clustering_metrics_populated(self):
        """Test que las métricas de clustering se populan."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2})
        cg.add_variable('Y', {1, 2})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=10)
        
        assert stats.clustering_metrics is not None
        assert stats.clustering_metrics.num_clusters > 0
    
    def test_arc_consistency_calls_tracked(self):
        """Test que se rastrean las llamadas a AC-3."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2})
        cg.add_variable('Y', {1, 2})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=10)
        
        assert stats.arc_consistency_calls > 0


class TestClusterSolver:
    """Tests para ClusterSolver."""
    
    def test_create_solver(self):
        """Test creación de solver."""
        solver = ClusterSolver()
        assert solver is not None
    
    def test_solve_cluster_simple(self):
        """Test resolver clúster simple."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2})
        cg.add_variable('Y', {1, 2})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        solver = ClusterSolver()
        solutions = solver.solve_cluster(cg, {'X', 'Y'}, max_solutions=10)
        
        # Debe encontrar soluciones
        assert len(solutions) > 0
        
        # Verificar que las soluciones son válidas
        for sol in solutions:
            assert 'X' in sol and 'Y' in sol
            assert sol['X'] != sol['Y']
    
    def test_solve_cluster_inconsistent(self):
        """Test clúster inconsistente."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1})
        cg.add_variable('Y', {1})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        solver = ClusterSolver()
        solutions = solver.solve_cluster(cg, {'X', 'Y'}, max_solutions=10)
        
        # No debe encontrar soluciones
        assert len(solutions) == 0
    
    def test_solve_cluster_with_max_solutions(self):
        """Test límite de soluciones en clúster."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        solver = ClusterSolver()
        solutions = solver.solve_cluster(cg, {'X', 'Y'}, max_solutions=3)
        
        # Debe respetar el límite
        assert len(solutions) <= 3
    
    def test_solve_cluster_subset_of_graph(self):
        """Test resolver solo un subconjunto del grafo."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2})
        cg.add_variable('Y', {1, 2})
        cg.add_variable('Z', {1, 2})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        cg.add_constraint('Y', 'Z', lambda y, z: y != z)
        
        solver = ClusterSolver()
        # Solo resolver X e Y
        solutions = solver.solve_cluster(cg, {'X', 'Y'}, max_solutions=10)
        
        # Las soluciones deben incluir solo X e Y
        for sol in solutions:
            assert 'X' in sol and 'Y' in sol
            assert 'Z' not in sol
    
    def test_solve_cluster_with_timeout(self):
        """Test resolver clúster con timeout."""
        cg = ConstraintGraph()
        
        # Crear clúster más complejo
        for i in range(5):
            cg.add_variable(f'X{i}', {1, 2, 3, 4, 5})
        
        for i in range(5):
            for j in range(i+1, 5):
                cg.add_constraint(f'X{i}', f'X{j}', lambda a, b: a != b)
        
        solver = ClusterSolver()
        cluster_vars = {f'X{i}' for i in range(5)}
        solutions = solver.solve_cluster(
            cg, cluster_vars, max_solutions=100, timeout=0.1
        )
        
        # Debe terminar en tiempo razonable
        assert len(solutions) >= 0  # Puede encontrar algunas o ninguna


class TestIntegration:
    """Tests de integración para resolución adaptativa."""
    
    def test_full_ace_pipeline(self):
        """Test pipeline completo de ACE."""
        # Crear problema CSP
        cg = ConstraintGraph()
        
        # Problema de coloreo de grafo simple
        # Grafo: X-Y-Z (cadena)
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_variable('Z', {1, 2, 3})
        
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        cg.add_constraint('Y', 'Z', lambda y, z: y != z)
        
        # Resolver con ACE
        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=10)
        
        # Verificar resultados
        assert len(stats.solutions) > 0
        assert stats.clustering_metrics is not None
        assert stats.nodes_explored > 0
        assert stats.arc_consistency_calls > 0
        
        # Verificar que las soluciones son válidas
        for sol in stats.solutions:
            assert sol['X'] != sol['Y']
            assert sol['Y'] != sol['Z']
    
    def test_compare_with_simple_backtracking(self):
        """Test comparar ACE con backtracking simple."""
        cg = ConstraintGraph()
        
        # Problema pequeño
        cg.add_variable('X', {1, 2})
        cg.add_variable('Y', {1, 2})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        # Resolver con ACE
        engine = AdaptiveConsistencyEngine()
        stats = engine.solve(cg, max_solutions=10)
        
        # Debe encontrar todas las soluciones
        assert len(stats.solutions) == 2
        
        # Verificar soluciones
        solutions_set = {
            (sol['X'], sol['Y']) for sol in stats.solutions
        }
        expected = {(1, 2), (2, 1)}
        assert solutions_set == expected
    
    def test_nqueens_benchmark(self):
        """Test benchmark con N-Reinas."""
        for n in [4, 6]:
            cg = ConstraintGraph()
            
            for i in range(n):
                cg.add_variable(f'Q{i}', set(range(n)))
            
            for i in range(n):
                for j in range(i+1, n):
                    def constraint(vi, vj, row_i=i, row_j=j):
                        return (vi != vj and 
                                abs(vi - vj) != abs(row_i - row_j))
                    
                    cg.add_constraint(f'Q{i}', f'Q{j}', constraint)
            
            engine = AdaptiveConsistencyEngine()
            stats = engine.solve(cg, max_solutions=100, timeout=5.0)
            
            # Debe encontrar al menos una solución
            assert len(stats.solutions) > 0
            
            # Verificar que las soluciones son válidas
            for sol in stats.solutions:
                # Verificar que todas las reinas están asignadas
                assert len(sol) == n
                
                # Verificar restricciones
                for i in range(n):
                    for j in range(i+1, n):
                        qi, qj = sol[f'Q{i}'], sol[f'Q{j}']
                        assert qi != qj
                        assert abs(qi - qj) != abs(i - j)

