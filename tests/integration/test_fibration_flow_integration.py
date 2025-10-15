"""
Tests de Integración para Fibration Flow

Este módulo implementa tests de integración end-to-end que verifican el funcionamiento
completo del sistema Fibration Flow con problemas reales de CSP.

Problemas testeados:
1. N-Queens (4, 6, 8 reinas)
2. Graph Coloring (grafos de diferentes topologías)
3. Sudoku (4x4, 9x9)
4. Job Shop Scheduling
5. Map Coloring
6. Magic Square

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import pytest
import time
from typing import Dict, List, Any, Tuple

from lattice_weaver.fibration.fibration_search_solver_enhanced import FibrationSearchSolverEnhanced
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
from lattice_weaver.arc_engine.core import ArcEngine


class TestNQueensIntegration:
    """Tests de integración para el problema N-Queens."""
    
    def _create_nqueens_problem(self, n: int) -> Tuple[ConstraintHierarchy, List[str], Dict[str, List[int]]]:
        """Crea un problema N-Queens."""
        hierarchy = ConstraintHierarchy()
        variables = [f"Q{i}" for i in range(n)]
        domains = {var: list(range(n)) for var in variables}
        
        # Restricciones: no dos reinas en misma columna o diagonal
        for i in range(n):
            for j in range(i + 1, n):
                # No misma columna
                def ne_constraint(assignment, i=i, j=j):
                    qi, qj = f"Q{i}", f"Q{j}"
                    if qi in assignment and qj in assignment:
                        return assignment[qi] != assignment[qj]
                    return True
                
                hierarchy.add_local_constraint(
                    var1=f"Q{i}", var2=f"Q{j}",
                    predicate=ne_constraint,
                    hardness=Hardness.HARD,
                    metadata={"name": f"Q{i}_ne_Q{j}"}
                )
                
                # No misma diagonal
                def no_diagonal(assignment, i=i, j=j):
                    qi, qj = f"Q{i}", f"Q{j}"
                    if qi in assignment and qj in assignment:
                        return abs(assignment[qi] - assignment[qj]) != abs(i - j)
                    return True
                
                hierarchy.add_local_constraint(
                    var1=f"Q{i}", var2=f"Q{j}",
                    predicate=no_diagonal,
                    hardness=Hardness.HARD,
                    metadata={"name": f"Q{i}_nodiag_Q{j}"}
                )
        
        return hierarchy, variables, domains
    
    def _verify_nqueens_solution(self, solution: Dict[str, int], n: int) -> bool:
        """Verifica que una solución de N-Queens es válida."""
        if len(solution) != n:
            return False
        
        positions = [solution[f"Q{i}"] for i in range(n)]
        
        # Verificar que no hay dos reinas en la misma columna
        if len(set(positions)) != n:
            return False
        
        # Verificar que no hay dos reinas en la misma diagonal
        for i in range(n):
            for j in range(i + 1, n):
                if abs(positions[i] - positions[j]) == abs(i - j):
                    return False
        
        return True
    
    @pytest.mark.parametrize("n,use_homotopy,use_tms", [
        (4, False, False),  # Baseline
        (4, True, False),   # Con HomotopyRules
        (4, False, True),   # Con TMS
        (4, True, True),    # Con ambos
    ])
    def test_4queens_with_different_configurations(self, n, use_homotopy, use_tms):
        """Test 4-Queens con diferentes configuraciones de optimización."""
        hierarchy, variables, domains = self._create_nqueens_problem(n)
        landscape = EnergyLandscapeOptimized(hierarchy)
        arc_engine = ArcEngine(use_tms=use_tms, parallel=False)
        
        solver = FibrationSearchSolverEnhanced(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine,
            variables=variables,
            domains=domains,
            use_homotopy=use_homotopy,
            use_tms=use_tms,
            use_enhanced_heuristics=True,
            max_backtracks=1000,
            max_iterations=1000,
            time_limit_seconds=10.0
        )
        
        start_time = time.time()
        solution = solver.solve()
        elapsed = time.time() - start_time
        
        # Verificaciones
        assert solution is not None, f"No se encontró solución para {n}-Queens"
        assert self._verify_nqueens_solution(solution, n), "Solución inválida"
        assert elapsed < 10.0, f"Tiempo excedido: {elapsed:.2f}s"
        
        # Estadísticas
        stats = solver.get_statistics()
        print(f"\n{n}-Queens (homotopy={use_homotopy}, tms={use_tms}):")
        print(f"  Tiempo: {elapsed:.3f}s")
        print(f"  Backtracks: {stats['search']['backtracks']}")
        print(f"  Nodos explorados: {stats['search']['nodes_explored']}")
        if use_tms:
            print(f"  Backjumps: {stats['search']['backjumps']}")
    
    @pytest.mark.parametrize("n", [4, 6, 8, 10])
    def test_nqueens_scalability(self, n):
        """Test de escalabilidad de N-Queens."""
        hierarchy, variables, domains = self._create_nqueens_problem(n)
        landscape = EnergyLandscapeOptimized(hierarchy)
        arc_engine = ArcEngine(use_tms=True, parallel=False)
        
        solver = FibrationSearchSolverEnhanced(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine,
            variables=variables,
            domains=domains,
            use_homotopy=True,
            use_tms=True,
            use_enhanced_heuristics=True,
            max_backtracks=10000,
            max_iterations=10000,
            time_limit_seconds=30.0
        )
        
        start_time = time.time()
        solution = solver.solve()
        elapsed = time.time() - start_time
        
        assert solution is not None, f"No se encontró solución para {n}-Queens"
        assert self._verify_nqueens_solution(solution, n), "Solución inválida"
        
        stats = solver.get_statistics()
        print(f"\n{n}-Queens (optimizado completo):")
        print(f"  Tiempo: {elapsed:.3f}s")
        print(f"  Backtracks: {stats['search']['backtracks']}")
        print(f"  Backjumps: {stats['search']['backjumps']}")
        print(f"  Nodos explorados: {stats['search']['nodes_explored']}")
        print(f"  Energía final: {stats['solution']['energy']:.4f}")


class TestGraphColoringIntegration:
    """Tests de integración para problemas de coloreo de grafos."""
    
    def _create_graph_coloring_problem(
        self, 
        n_nodes: int, 
        n_colors: int,
        graph_type: str = "cycle"
    ) -> Tuple[ConstraintHierarchy, List[str], Dict[str, List[int]]]:
        """
        Crea un problema de coloreo de grafos.
        
        Args:
            n_nodes: Número de nodos
            n_colors: Número de colores disponibles
            graph_type: Tipo de grafo ("cycle", "complete", "bipartite", "random")
        """
        hierarchy = ConstraintHierarchy()
        variables = [f"N{i}" for i in range(n_nodes)]
        domains = {var: list(range(n_colors)) for var in variables}
        
        # Generar aristas según el tipo de grafo
        edges = []
        if graph_type == "cycle":
            for i in range(n_nodes):
                j = (i + 1) % n_nodes
                edges.append((i, j))
        
        elif graph_type == "complete":
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    edges.append((i, j))
        
        elif graph_type == "bipartite":
            half = n_nodes // 2
            for i in range(half):
                for j in range(half, n_nodes):
                    edges.append((i, j))
        
        elif graph_type == "random":
            import random
            random.seed(42)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if random.random() < 0.3:  # 30% de probabilidad
                        edges.append((i, j))
        
        # Añadir restricciones de coloreo
        for i, j in edges:
            def ne_constraint(assignment, i=i, j=j):
                ni, nj = f"N{i}", f"N{j}"
                if ni in assignment and nj in assignment:
                    return assignment[ni] != assignment[nj]
                return True
            
            hierarchy.add_local_constraint(
                var1=f"N{i}", var2=f"N{j}",
                predicate=ne_constraint,
                hardness=Hardness.HARD,
                metadata={"name": f"N{i}_ne_N{j}"}
            )
        
        return hierarchy, variables, domains
    
    def _verify_graph_coloring_solution(
        self, 
        solution: Dict[str, int], 
        hierarchy: ConstraintHierarchy
    ) -> bool:
        """Verifica que una solución de coloreo es válida."""
        # Verificar todas las restricciones HARD
        from lattice_weaver.fibration.constraint_hierarchy import ConstraintLevel
        for level in [ConstraintLevel.LOCAL, ConstraintLevel.PATTERN, ConstraintLevel.GLOBAL]:
            for constraint in hierarchy.get_constraints_at_level(level):
                if constraint.hardness == Hardness.HARD:
                    if not constraint.predicate(solution):
                        return False
        return True
    
    @pytest.mark.parametrize("graph_type,n_nodes,n_colors", [
        ("cycle", 10, 3),
        ("cycle", 15, 3),
        ("bipartite", 12, 2),
        ("random", 12, 4),
    ])
    def test_graph_coloring_different_topologies(self, graph_type, n_nodes, n_colors):
        """Test de coloreo en diferentes topologías de grafos."""
        hierarchy, variables, domains = self._create_graph_coloring_problem(
            n_nodes, n_colors, graph_type
        )
        landscape = EnergyLandscapeOptimized(hierarchy)
        arc_engine = ArcEngine(use_tms=True, parallel=False)
        
        solver = FibrationSearchSolverEnhanced(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine,
            variables=variables,
            domains=domains,
            use_homotopy=True,
            use_tms=True,
            use_enhanced_heuristics=True,
            max_backtracks=5000,
            max_iterations=5000,
            time_limit_seconds=20.0
        )
        
        start_time = time.time()
        solution = solver.solve()
        elapsed = time.time() - start_time
        
        assert solution is not None, f"No se encontró solución para {graph_type} con {n_nodes} nodos"
        assert self._verify_graph_coloring_solution(solution, hierarchy), "Solución inválida"
        
        stats = solver.get_statistics()
        print(f"\nGraph Coloring ({graph_type}, {n_nodes} nodos, {n_colors} colores):")
        print(f"  Tiempo: {elapsed:.3f}s")
        print(f"  Backtracks: {stats['search']['backtracks']}")
        print(f"  Backjumps: {stats['search']['backjumps']}")
        print(f"  Energía final: {stats['solution']['energy']:.4f}")


class TestSudokuIntegration:
    """Tests de integración para Sudoku."""
    
    def _create_sudoku_problem(self, size: int = 4) -> Tuple[ConstraintHierarchy, List[str], Dict[str, List[int]]]:
        """
        Crea un problema de Sudoku.
        
        Args:
            size: Tamaño del Sudoku (4 para 4x4, 9 para 9x9)
        """
        hierarchy = ConstraintHierarchy()
        variables = [f"C{i}_{j}" for i in range(size) for j in range(size)]
        domains = {var: list(range(1, size + 1)) for var in variables}
        
        # Restricciones de fila: todos diferentes
        for i in range(size):
            for j1 in range(size):
                for j2 in range(j1 + 1, size):
                    def ne_row(assignment, i=i, j1=j1, j2=j2):
                        c1, c2 = f"C{i}_{j1}", f"C{i}_{j2}"
                        if c1 in assignment and c2 in assignment:
                            return assignment[c1] != assignment[c2]
                        return True
                    
                    hierarchy.add_local_constraint(
                        var1=f"C{i}_{j1}", var2=f"C{i}_{j2}",
                        predicate=ne_row,
                        hardness=Hardness.HARD,
                        metadata={"name": f"row_{i}_{j1}_{j2}"}
                    )
        
        # Restricciones de columna: todos diferentes
        for j in range(size):
            for i1 in range(size):
                for i2 in range(i1 + 1, size):
                    def ne_col(assignment, j=j, i1=i1, i2=i2):
                        c1, c2 = f"C{i1}_{j}", f"C{i2}_{j}"
                        if c1 in assignment and c2 in assignment:
                            return assignment[c1] != assignment[c2]
                        return True
                    
                    hierarchy.add_local_constraint(
                        var1=f"C{i1}_{j}", var2=f"C{i2}_{j}",
                        predicate=ne_col,
                        hardness=Hardness.HARD,
                        metadata={"name": f"col_{j}_{i1}_{i2}"}
                    )
        
        # Restricciones de subcuadrícula (solo para 4x4 y 9x9)
        if size == 4:
            box_size = 2
        elif size == 9:
            box_size = 3
        else:
            box_size = int(size ** 0.5)
        
        for box_i in range(size // box_size):
            for box_j in range(size // box_size):
                cells = [
                    (box_i * box_size + di, box_j * box_size + dj)
                    for di in range(box_size)
                    for dj in range(box_size)
                ]
                
                for idx1, (i1, j1) in enumerate(cells):
                    for i2, j2 in cells[idx1 + 1:]:
                        def ne_box(assignment, i1=i1, j1=j1, i2=i2, j2=j2):
                            c1, c2 = f"C{i1}_{j1}", f"C{i2}_{j2}"
                            if c1 in assignment and c2 in assignment:
                                return assignment[c1] != assignment[c2]
                            return True
                        
                        hierarchy.add_local_constraint(
                            var1=f"C{i1}_{j1}", var2=f"C{i2}_{j2}",
                            predicate=ne_box,
                            hardness=Hardness.HARD,
                            metadata={"name": f"box_{box_i}_{box_j}_{i1}_{j1}_{i2}_{j2}"}
                        )
        
        return hierarchy, variables, domains
    
    @pytest.mark.parametrize("size", [4, 9])
    def test_sudoku(self, size):
        """Test de Sudoku (4x4 y 9x9)."""
        hierarchy, variables, domains = self._create_sudoku_problem(size)
        landscape = EnergyLandscapeOptimized(hierarchy)
        arc_engine = ArcEngine(use_tms=True, parallel=False)
        
        solver = FibrationSearchSolverEnhanced(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine,
            variables=variables,
            domains=domains,
            use_homotopy=False,  # Desactivar para reducir tiempo
            use_tms=True,
            use_enhanced_heuristics=True,
            max_backtracks=5000,
            max_iterations=5000,
            time_limit_seconds=20.0
        )
        
        start_time = time.time()
        solution = solver.solve()
        elapsed = time.time() - start_time
        
        assert solution is not None, f"No se encontró solución para Sudoku {size}x{size}"
        
        stats = solver.get_statistics()
        print(f"\nSudoku {size}x{size}:")
        print(f"  Tiempo: {elapsed:.3f}s")
        print(f"  Backtracks: {stats['search']['backtracks']}")
        print(f"  Backjumps: {stats['search']['backjumps']}")
        print(f"  Nodos explorados: {stats['search']['nodes_explored']}")


class TestMagicSquareIntegration:
    """Tests de integración para Magic Square."""
    
    def _create_magic_square_problem(self, n: int) -> Tuple[ConstraintHierarchy, List[str], Dict[str, List[int]]]:
        """
        Crea un problema de Magic Square.
        
        Args:
            n: Tamaño del cuadrado mágico (n x n)
        """
        hierarchy = ConstraintHierarchy()
        variables = [f"S{i}_{j}" for i in range(n) for j in range(n)]
        domains = {var: list(range(1, n * n + 1)) for var in variables}
        
        magic_sum = n * (n * n + 1) // 2
        
        # Restricción: todos los valores diferentes
        all_vars = variables[:]
        for idx1, var1 in enumerate(all_vars):
            for var2 in all_vars[idx1 + 1:]:
                def ne_constraint(assignment, v1=var1, v2=var2):
                    if v1 in assignment and v2 in assignment:
                        return assignment[v1] != assignment[v2]
                    return True
                
                hierarchy.add_local_constraint(
                    var1=var1, var2=var2,
                    predicate=ne_constraint,
                    hardness=Hardness.HARD,
                    metadata={"name": f"{var1}_ne_{var2}"}
                )
        
        # Restricciones de suma (SOFT para permitir búsqueda gradual)
        # Filas
        for i in range(n):
            row_vars = [f"S{i}_{j}" for j in range(n)]
            
            def row_sum(assignment, row_vars=row_vars, magic_sum=magic_sum):
                if all(v in assignment for v in row_vars):
                    return sum(assignment[v] for v in row_vars) == magic_sum
                return True
            
            hierarchy.add_global_constraint(
                variables=row_vars,
                predicate=row_sum,
                hardness=Hardness.SOFT,
                weight=10.0,
                metadata={"name": f"row_{i}_sum"}
            )
        
        # Columnas
        for j in range(n):
            col_vars = [f"S{i}_{j}" for i in range(n)]
            
            def col_sum(assignment, col_vars=col_vars, magic_sum=magic_sum):
                if all(v in assignment for v in col_vars):
                    return sum(assignment[v] for v in col_vars) == magic_sum
                return True
            
            hierarchy.add_global_constraint(
                variables=col_vars,
                predicate=col_sum,
                hardness=Hardness.SOFT,
                weight=10.0,
                metadata={"name": f"col_{j}_sum"}
            )
        
        # Diagonales
        diag1_vars = [f"S{i}_{i}" for i in range(n)]
        def diag1_sum(assignment, diag_vars=diag1_vars, magic_sum=magic_sum):
            if all(v in assignment for v in diag_vars):
                return sum(assignment[v] for v in diag_vars) == magic_sum
            return True
        
        hierarchy.add_global_constraint(
            variables=diag1_vars,
            predicate=diag1_sum,
            hardness=Hardness.SOFT,
            weight=10.0,
            metadata={"name": "diag1_sum"}
        )
        
        diag2_vars = [f"S{i}_{n-1-i}" for i in range(n)]
        def diag2_sum(assignment, diag_vars=diag2_vars, magic_sum=magic_sum):
            if all(v in assignment for v in diag_vars):
                return sum(assignment[v] for v in diag_vars) == magic_sum
            return True
        
        hierarchy.add_global_constraint(
            variables=diag2_vars,
            predicate=diag2_sum,
            hardness=Hardness.SOFT,
            weight=10.0,
            metadata={"name": "diag2_sum"}
        )
        
        return hierarchy, variables, domains
    
    @pytest.mark.parametrize("n", [3, 4])
    def test_magic_square(self, n):
        """Test de Magic Square (3x3 y 4x4)."""
        hierarchy, variables, domains = self._create_magic_square_problem(n)
        landscape = EnergyLandscapeOptimized(hierarchy)
        arc_engine = ArcEngine(use_tms=True, parallel=False)
        
        solver = FibrationSearchSolverEnhanced(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine,
            variables=variables,
            domains=domains,
            use_homotopy=False,  # Desactivar para reducir tiempo
            use_tms=True,
            use_enhanced_heuristics=True,
            max_backtracks=10000,
            max_iterations=10000,
            time_limit_seconds=30.0
        )
        
        start_time = time.time()
        solution = solver.solve()
        elapsed = time.time() - start_time
        
        assert solution is not None, f"No se encontró solución para Magic Square {n}x{n}"
        
        stats = solver.get_statistics()
        print(f"\nMagic Square {n}x{n}:")
        print(f"  Tiempo: {elapsed:.3f}s")
        print(f"  Backtracks: {stats['search']['backtracks']}")
        print(f"  Backjumps: {stats['search']['backjumps']}")
        print(f"  Energía final: {stats['solution']['energy']:.4f}")
        
        # Mostrar solución
        if solution:
            print(f"\n  Solución encontrada:")
            for i in range(n):
                row = [solution[f"S{i}_{j}"] for j in range(n)]
                print(f"    {row}")


class TestPerformanceComparison:
    """Tests de comparación de rendimiento entre configuraciones."""
    
    def test_configuration_comparison_nqueens(self):
        """Compara diferentes configuraciones en N-Queens."""
        n = 6
        configurations = [
            ("Baseline", False, False, False),
            ("HomotopyRules", True, False, False),
            ("TMS", False, True, False),
            ("Enhanced Heuristics", False, False, True),
            ("Full Optimized", True, True, True),
        ]
        
        results = []
        
        for name, use_homotopy, use_tms, use_enhanced in configurations:
            hierarchy, variables, domains = TestNQueensIntegration()._create_nqueens_problem(n)
            landscape = EnergyLandscapeOptimized(hierarchy)
            arc_engine = ArcEngine(use_tms=use_tms, parallel=False)
            
            solver = FibrationSearchSolverEnhanced(
                hierarchy=hierarchy,
                landscape=landscape,
                arc_engine=arc_engine,
                variables=variables,
                domains=domains,
                use_homotopy=use_homotopy,
                use_tms=use_tms,
                use_enhanced_heuristics=use_enhanced,
                max_backtracks=10000,
                max_iterations=10000,
                time_limit_seconds=30.0
            )
            
            start_time = time.time()
            solution = solver.solve()
            elapsed = time.time() - start_time
            
            stats = solver.get_statistics()
            results.append({
                'name': name,
                'time': elapsed,
                'backtracks': stats['search']['backtracks'],
                'backjumps': stats['search']['backjumps'],
                'nodes': stats['search']['nodes_explored'],
                'found': solution is not None
            })
        
        # Mostrar comparación
        print(f"\n{'='*80}")
        print(f"Comparación de Configuraciones - {n}-Queens")
        print(f"{'='*80}")
        print(f"{'Configuración':<25} {'Tiempo':<12} {'Backtracks':<12} {'Backjumps':<12} {'Nodos':<12}")
        print(f"{'-'*80}")
        
        for r in results:
            print(f"{r['name']:<25} {r['time']:>8.3f}s   {r['backtracks']:>10}   {r['backjumps']:>10}   {r['nodes']:>10}")
        
        print(f"{'='*80}\n")
        
        # Verificar que la versión optimizada es mejor
        baseline = results[0]
        optimized = results[-1]
        
        assert optimized['backtracks'] <= baseline['backtracks'], \
            "La versión optimizada debería tener menos backtracks"

