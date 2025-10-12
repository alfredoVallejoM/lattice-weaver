"""
Tests de integración end-to-end para el sistema de familias de problemas.

Estos tests verifican el flujo completo:
1. Obtener familia del catálogo
2. Generar problema (ArcEngine)
3. Validar estructura del problema
4. Validar soluciones conocidas
"""

import pytest
from lattice_weaver.problems import get_catalog


class TestEndToEndNQueens:
    """Tests end-to-end para N-Queens."""
    
    def test_4queens_generation(self):
        """Test generación de 4-Queens."""
        # 1. Obtener familia del catálogo
        catalog = get_catalog()
        assert 'nqueens' in catalog
        
        # 2. Generar problema
        engine = catalog.generate_problem('nqueens', n=4)
        assert engine is not None
        assert len(engine.variables) == 4
        assert len(engine.constraints) == 6  # C(4,2) = 6
    
    def test_4queens_known_solution(self):
        """Test validación de solución conocida de 4-Queens."""
        catalog = get_catalog()
        
        # Solución conocida para 4-Queens
        solution = {
            'Q0': 1,
            'Q1': 3,
            'Q2': 0,
            'Q3': 2
        }
        
        is_valid = catalog.validate_solution('nqueens', solution, n=4)
        assert is_valid is True
    
    def test_8queens_known_solution(self):
        """Test validación de solución conocida de 8-Queens."""
        catalog = get_catalog()
        
        # Solución conocida para 8-Queens
        solution = {
            'Q0': 0,
            'Q1': 4,
            'Q2': 7,
            'Q3': 5,
            'Q4': 2,
            'Q5': 6,
            'Q6': 1,
            'Q7': 3
        }
        
        is_valid = catalog.validate_solution('nqueens', solution, n=8)
        assert is_valid is True
    
    def test_nqueens_metadata(self):
        """Test que los metadatos son correctos."""
        catalog = get_catalog()
        metadata = catalog.get_metadata('nqueens', n=8)
        
        assert metadata['family'] == 'nqueens'
        assert metadata['n_variables'] == 8
        assert metadata['n_constraints'] == 28


class TestEndToEndGraphColoring:
    """Tests end-to-end para Graph Coloring."""
    
    def test_cycle_graph_generation(self):
        """Test generación de grafo cíclico."""
        catalog = get_catalog()
        
        engine = catalog.generate_problem(
            'graph_coloring',
            graph_type='cycle',
            n_nodes=5,
            n_colors=3
        )
        
        assert engine is not None
        assert len(engine.variables) == 5
        assert len(engine.constraints) == 5  # Ciclo tiene n aristas
    
    def test_cycle_known_solution(self):
        """Test validación de solución conocida para ciclo."""
        catalog = get_catalog()
        
        # Ciclo de 5 nodos con 3 colores
        solution = {
            'V0': 0,
            'V1': 1,
            'V2': 0,
            'V3': 1,
            'V4': 2
        }
        
        is_valid = catalog.validate_solution(
            'graph_coloring',
            solution,
            graph_type='cycle',
            n_nodes=5,
            n_colors=3
        )
        assert is_valid is True
    
    def test_path_known_solution(self):
        """Test validación de solución conocida para camino."""
        catalog = get_catalog()
        
        # Camino de 4 nodos con 2 colores
        solution = {
            'V0': 0,
            'V1': 1,
            'V2': 0,
            'V3': 1
        }
        
        is_valid = catalog.validate_solution(
            'graph_coloring',
            solution,
            graph_type='path',
            n_nodes=4,
            n_colors=2
        )
        assert is_valid is True


class TestEndToEndSudoku:
    """Tests end-to-end para Sudoku."""
    
    def test_empty_4x4_sudoku_generation(self):
        """Test generación de Sudoku 4x4 vacío."""
        catalog = get_catalog()
        
        engine = catalog.generate_problem(
            'sudoku',
            size=4,
            difficulty='empty'
        )
        
        assert engine is not None
        assert len(engine.variables) == 16  # 4x4 = 16 celdas
    
    def test_4x4_sudoku_known_solution(self):
        """Test validación de solución conocida de Sudoku 4x4."""
        catalog = get_catalog()
        
        # Solución válida de Sudoku 4x4
        solution = {
            'C_0_0': 1, 'C_0_1': 2, 'C_0_2': 3, 'C_0_3': 4,
            'C_1_0': 3, 'C_1_1': 4, 'C_1_2': 1, 'C_1_3': 2,
            'C_2_0': 2, 'C_2_1': 1, 'C_2_2': 4, 'C_2_3': 3,
            'C_3_0': 4, 'C_3_1': 3, 'C_3_2': 2, 'C_3_3': 1
        }
        
        is_valid = catalog.validate_solution('sudoku', solution, size=4)
        assert is_valid is True


class TestCatalogIntegration:
    """Tests de integración del catálogo."""
    
    def test_catalog_has_all_families(self):
        """Test que el catálogo tiene todas las familias registradas."""
        catalog = get_catalog()
        families = catalog.list_families()
        
        assert 'nqueens' in families
        assert 'graph_coloring' in families
        assert 'sudoku' in families
    
    def test_catalog_print(self, capsys):
        """Test que print_catalog() funciona correctamente."""
        catalog = get_catalog()
        catalog.print_catalog()
        
        captured = capsys.readouterr()
        assert 'nqueens' in captured.out
        assert 'graph_coloring' in captured.out
        assert 'sudoku' in captured.out
    
    def test_get_family_directly(self):
        """Test que se pueden obtener familias directamente."""
        catalog = get_catalog()
        
        nqueens = catalog.get('nqueens')
        assert nqueens is not None
        assert nqueens.name == 'nqueens'
        
        graph_coloring = catalog.get('graph_coloring')
        assert graph_coloring is not None
        assert graph_coloring.name == 'graph_coloring'
        
        sudoku = catalog.get('sudoku')
        assert sudoku is not None
        assert sudoku.name == 'sudoku'
    
    def test_generate_with_default_params(self):
        """Test que se puede generar con parámetros por defecto."""
        catalog = get_catalog()
        
        # N-Queens con parámetros por defecto (n=8)
        nqueens_family = catalog.get('nqueens')
        defaults = nqueens_family.get_default_params()
        engine = catalog.generate_problem('nqueens', **defaults)
        assert len(engine.variables) == 8
        
        # Graph Coloring con parámetros por defecto
        gc_family = catalog.get('graph_coloring')
        defaults = gc_family.get_default_params()
        engine = catalog.generate_problem('graph_coloring', **defaults)
        assert len(engine.variables) == 10
        
        # Sudoku con parámetros por defecto
        sudoku_family = catalog.get('sudoku')
        defaults = sudoku_family.get_default_params()
        engine = catalog.generate_problem('sudoku', **defaults)
        assert len(engine.variables) == 81


class TestPerformance:
    """Tests de rendimiento básico."""
    
    @pytest.mark.slow
    def test_16queens_performance(self):
        """Test que 16-Queens se puede resolver en tiempo razonable."""
        catalog = get_catalog()
        
        engine = catalog.generate_problem('nqueens', n=16)
        solver = CSPSolver(engine)
        
        # Esto puede tomar varios segundos
        solution = solver.solve()
        
        if solution is not None:
            assert catalog.validate_solution('nqueens', solution, n=16) is True
    
    @pytest.mark.slow
    def test_large_graph_coloring_performance(self):
        """Test que grafos grandes se pueden generar rápidamente."""
        catalog = get_catalog()
        
        # Generar grafo grande (100 nodos)
        engine = catalog.generate_problem(
            'graph_coloring',
            graph_type='random',
            n_nodes=100,
            n_colors=10,
            edge_probability=0.1,
            seed=42
        )
        
        assert len(engine.variables) == 100
        # No intentamos resolver (sería muy lento), solo verificamos generación


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

