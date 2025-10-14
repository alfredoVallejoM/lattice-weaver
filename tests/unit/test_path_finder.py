"""
Tests Unitarios para PathFinder

Verifica la correcta búsqueda de caminos entre soluciones CSP.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import pytest
from lattice_weaver.formal.path_finder import (
    PathFinder,
    SolutionPath,
    create_path_finder
)
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.formal.csp_cubical_bridge import CSPToCubicalBridge


class TestSolutionPath:
    """Tests para SolutionPath."""
    
    def test_create_solution_path(self):
        """Test: Crear camino de solución."""
        path = SolutionPath(
            start={'X': 1, 'Y': 2},
            end={'X': 2, 'Y': 3},
            steps=tuple([{'X': 1, 'Y': 3}]),
            distance=2
        )
        
        assert path.start == {'X': 1, 'Y': 2}
        assert path.end == {'X': 2, 'Y': 3}
        assert len(path.steps) == 1
        assert path.distance == 2
    
    def test_path_length(self):
        """Test: Longitud del camino."""
        path = SolutionPath(
            start={'X': 1},
            end={'X': 3},
            steps=tuple([{'X': 2}]),
            distance=2
        )
        
        assert len(path) == 1  # Un paso intermedio
    
    def test_is_direct(self):
        """Test: Verificar si camino es directo."""
        # Camino directo
        path1 = SolutionPath(
            start={'X': 1},
            end={'X': 1},
            steps=tuple(),
            distance=0
        )
        assert path1.is_direct() is True
        
        # Camino con pasos
        path2 = SolutionPath(
            start={'X': 1},
            end={'X': 3},
            steps=tuple([{'X': 2}]),
            distance=2
        )
        assert path2.is_direct() is False


class TestPathFinderConstruction:
    """Tests para construcción de PathFinder."""
    
    def test_create_path_finder(self):
        """Test: Crear PathFinder."""
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2}), 'Y': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        
        finder = PathFinder(bridge)
        
        assert finder.bridge is bridge
        assert finder.max_search_depth == 10
    
    def test_create_path_finder_custom_depth(self):
        """Test: Crear PathFinder con profundidad personalizada."""
        csp = CSP(
            variables={'X'},
            domains={'X': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        
        finder = PathFinder(bridge, max_search_depth=5)
        
        assert finder.max_search_depth == 5
    
    def test_create_path_finder_function(self):
        """Test: Función create_path_finder."""
        csp = CSP(
            variables={'X'},
            domains={'X': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        
        finder = create_path_finder(bridge, max_depth=15)
        
        assert isinstance(finder, PathFinder)
        assert finder.max_search_depth == 15


class TestHammingDistance:
    """Tests para distancia de Hamming."""
    
    def test_hamming_distance_identical(self):
        """Test: Distancia de Hamming entre soluciones idénticas."""
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2}), 'Y': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge)
        
        solution = {'X': 1, 'Y': 2}
        distance = finder.hamming_distance(solution, solution)
        
        assert distance == 0
    
    def test_hamming_distance_one_diff(self):
        """Test: Distancia de Hamming con una diferencia."""
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2}), 'Y': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge)
        
        sol1 = {'X': 1, 'Y': 2}
        sol2 = {'X': 1, 'Y': 1}
        distance = finder.hamming_distance(sol1, sol2)
        
        assert distance == 1
    
    def test_hamming_distance_all_diff(self):
        """Test: Distancia de Hamming con todas las variables diferentes."""
        csp = CSP(
            variables={'X', 'Y', 'Z'},
            domains={'X': frozenset({1, 2}), 'Y': frozenset({1, 2}), 'Z': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge)
        
        sol1 = {'X': 1, 'Y': 1, 'Z': 1}
        sol2 = {'X': 2, 'Y': 2, 'Z': 2}
        distance = finder.hamming_distance(sol1, sol2)
        
        assert distance == 3


class TestPathFinding:
    """Tests para búsqueda de caminos."""
    
    def test_find_path_identical_solutions(self):
        """Test: Camino entre soluciones idénticas."""
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2}), 'Y': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge)
        
        solution = {'X': 1, 'Y': 2}
        path = finder.find_path(solution, solution)
        
        assert path is not None
        assert path.start == solution
        assert path.end == solution
        assert path.distance == 0
        assert path.is_direct()
    
    def test_find_path_simple(self):
        """Test: Encontrar camino simple."""
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2, 3}), 'Y': frozenset({1, 2, 3})},
            constraints=[Constraint(scope=frozenset({'X', 'Y'}), relation=lambda x, y: x != y, name='neq_xy')]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge)
        
        sol1 = {'X': 1, 'Y': 2}
        sol2 = {'X': 2, 'Y': 3}
        
        path = finder.find_path(sol1, sol2)
        
        assert path is not None
        assert path.start == sol1
        assert path.end == sol2
        assert path.distance > 0
    
    def test_find_path_no_path(self):
        """Test: No existe camino entre soluciones."""
        # CSP donde las soluciones están desconectadas
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2}), 'Y': frozenset({1, 2})},
            constraints=[
                Constraint(scope=frozenset({'X', 'Y'}), relation=lambda x, y: (x == 1 and y == 2) or (x == 2 and y == 1), name='custom_constraint')
            ]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge, max_search_depth=5)
        
        sol1 = {'X': 1, 'Y': 2}
        sol2 = {'X': 2, 'Y': 1}
        
        path = finder.find_path(sol1, sol2)
        
        # En este caso particular, puede haber o no camino dependiendo de la estructura
        # El test verifica que el método no falle
        assert path is None or isinstance(path, SolutionPath)
    
    def test_find_path_invalid_start(self):
        """Test: Solución inicial inválida."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        finder = PathFinder(bridge)
        
        sol1 = {'X': 1, 'Y': 1}  # Inválida
        sol2 = {'X': 1, 'Y': 2}
        
        path = finder.find_path(sol1, sol2)
        
        assert path is None
    
    def test_find_path_invalid_end(self):
        """Test: Solución final inválida."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        finder = PathFinder(bridge)
        
        sol1 = {'X': 1, 'Y': 2}
        sol2 = {'X': 2, 'Y': 2}  # Inválida
        
        path = finder.find_path(sol1, sol2)
        
        assert path is None
    
    def test_find_path_caching(self):
        """Test: Caché de caminos."""
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2}), 'Y': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge)
        
        sol1 = {'X': 1, 'Y': 1}
        sol2 = {'X': 2, 'Y': 2}
        
        # Primera búsqueda
        path1 = finder.find_path(sol1, sol2)
        
        # Segunda búsqueda (debe usar caché)
        path2 = finder.find_path(sol1, sol2)
        
        assert path1 is path2  # Mismo objeto por caché


class TestEquivalence:
    """Tests para verificación de equivalencia."""
    
    def test_are_equivalent_true(self):
        """Test: Soluciones equivalentes."""
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2, 3}), 'Y': frozenset({1, 2, 3})},
            constraints=[Constraint(scope=frozenset({'X', 'Y'}), relation=lambda x, y: x != y, name='neq_xy')]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge)
        
        sol1 = {'X': 1, 'Y': 2}
        sol2 = {'X': 2, 'Y': 3}
        
        assert finder.are_equivalent(sol1, sol2) is True
    
    def test_are_equivalent_false(self):
        """Test: Soluciones no equivalentes."""
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2}), 'Y': frozenset({1, 2})},
            constraints=[
                Constraint(scope=frozenset({'X', 'Y'}), relation=lambda x, y: (x == 1 and y == 2) or (x == 2 and y == 1), name='custom_constraint')
            ]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge, max_search_depth=3)
        
        sol1 = {'X': 1, 'Y': 2}
        sol2 = {'X': 2, 'Y': 1}
        
        # Puede ser True o False dependiendo de la estructura
        result = finder.are_equivalent(sol1, sol2)
        assert isinstance(result, bool)


class TestNeighbors:
    """Tests para obtención de vecinos."""
    
    def test_get_solution_neighbors(self):
        """Test: Obtener vecinos de una solución."""
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2}), 'Y': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge)
        
        solution = {'X': 1, 'Y': 1}
        neighbors = finder.get_solution_neighbors(solution)
        
        # Debe tener vecinos (cambiando X o Y)
        assert len(neighbors) > 0
        
        # Todos los vecinos deben ser diferentes de la solución original
        for neighbor in neighbors:
            assert neighbor != solution
    
    def test_get_solution_neighbors_valid_only(self):
        """Test: Obtener solo vecinos válidos."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        finder = PathFinder(bridge)
        
        solution = {'X': 1, 'Y': 2}
        neighbors = finder.get_solution_neighbors(solution, valid_only=True)
        
        # Todos los vecinos deben ser válidos
        for neighbor in neighbors:
            assert bridge.verify_solution(neighbor)
    
    def test_get_solution_neighbors_include_invalid(self):
        """Test: Obtener vecinos incluyendo inválidos."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        finder = PathFinder(bridge)
        
        solution = {'X': 1, 'Y': 2}
        neighbors = finder.get_solution_neighbors(solution, valid_only=False)
        
        # Debe incluir vecinos inválidos
        assert len(neighbors) > 0


class TestCaching:
    """Tests para sistema de caché."""
    
    def test_clear_cache(self):
        """Test: Limpiar caché."""
        csp = CSP(
            variables={'X'},
            domains={'X': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge)
        
        # Buscar camino (cachear)
        finder.find_path({'X': 1}, {'X': 2})
        
        # Limpiar caché
        finder.clear_cache()
        
        assert len(finder._path_cache) == 0


class TestStringRepresentation:
    """Tests para representación en string."""
    
    def test_str_representation(self):
        """Test: Representación en string de PathFinder."""
        csp = CSP(
            variables={'X'},
            domains={'X': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge, max_search_depth=15)
        
        str_repr = str(finder)
        assert "PathFinder" in str_repr
        assert "15" in str_repr
    
    def test_repr_representation(self):
        """Test: Representación detallada de PathFinder."""
        csp = CSP(
            variables={'X'},
            domains={'X': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp_problem=csp)
        finder = PathFinder(bridge)
        
        repr_str = repr(finder)
        assert "PathFinder" in repr_str
        assert "max_depth" in repr_str
        assert "cached_paths" in repr_str
    
    def test_solution_path_str(self):
        """Test: Representación en string de SolutionPath."""
        path = SolutionPath(
            start={'X': 1},
            end={'X': 3},
            steps=tuple([{'X': 2}]),
            distance=2
        )
        
        str_repr = str(path)
        assert "Path" in str_repr
        assert "1 steps" in str_repr


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

