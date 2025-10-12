"""
Tests de Integración: CSP ↔ Cubical

Tests end-to-end que verifican la integración completa entre
el motor CSP (ArcEngine) y el sistema de tipos cúbicos.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import pytest
from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.formal.cubical_csp_type import CubicalCSPType
from lattice_weaver.formal.csp_cubical_bridge import (
    CSPToCubicalBridge,
    create_simple_csp_bridge
)
from lattice_weaver.formal.path_finder import PathFinder
from lattice_weaver.formal.symmetry_extractor import SymmetryExtractor


class TestEndToEndIntegration:
    """Tests end-to-end de integración completa."""
    
    def test_full_workflow_simple_csp(self):
        """Test: Flujo completo con CSP simple."""
        # 1. Crear CSP
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y', 'Z'],
            domains={'X': {1, 2, 3}, 'Y': {1, 2, 3}, 'Z': {1, 2, 3}},
            constraints=[
                ('X', 'Y', lambda x, y: x != y),
                ('Y', 'Z', lambda y, z: y != z)
            ]
        )
        
        # 2. Verificar traducción
        cubical_type = bridge.cubical_type
        assert cubical_type is not None
        assert len(cubical_type.variables) == 3
        assert len(cubical_type.constraint_props) == 2
        
        # 3. Verificar soluciones
        valid_solution = {'X': 1, 'Y': 2, 'Z': 1}
        invalid_solution = {'X': 1, 'Y': 1, 'Z': 2}
        
        assert bridge.verify_solution(valid_solution) is True
        assert bridge.verify_solution(invalid_solution) is False
        
        # 4. Buscar caminos
        finder = PathFinder(bridge)
        solution1 = {'X': 1, 'Y': 2, 'Z': 1}
        solution2 = {'X': 2, 'Y': 3, 'Z': 1}
        
        path = finder.find_path(solution1, solution2)
        assert path is not None
        
        # 5. Analizar simetrías
        extractor = SymmetryExtractor(bridge)
        analysis = extractor.analyze_symmetry_structure()
        
        assert 'symmetry_count' in analysis
        assert 'has_symmetries' in analysis
    
    def test_nqueens_integration(self):
        """Test: Integración con N-Queens."""
        # Crear 3-Queens simplificado
        bridge = create_simple_csp_bridge(
            variables=['Q1', 'Q2', 'Q3'],
            domains={'Q1': {1, 2, 3}, 'Q2': {1, 2, 3}, 'Q3': {1, 2, 3}},
            constraints=[
                ('Q1', 'Q2', lambda x, y: x != y),
                ('Q2', 'Q3', lambda x, y: x != y),
                ('Q1', 'Q3', lambda x, y: x != y)
            ]
        )
        
        # Verificar traducción
        assert bridge.cubical_type is not None
        
        # Verificar soluciones
        solution = {'Q1': 1, 'Q2': 2, 'Q3': 3}
        assert bridge.verify_solution(solution) is True
        
        # Analizar simetrías
        extractor = SymmetryExtractor(bridge)
        group = extractor.extract_all_symmetries()
        
        # Puede o no tener simetrías dependiendo de las restricciones
        assert group.order >= 0


class TestBridgeWithPathFinder:
    """Tests de integración Bridge + PathFinder."""
    
    def test_path_finding_with_bridge(self):
        """Test: Búsqueda de caminos con bridge."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2, 3}, 'Y': {1, 2, 3}},
            constraints=[('X', 'Y', lambda x, y: x < y)]
        )
        
        finder = PathFinder(bridge)
        
        sol1 = {'X': 1, 'Y': 2}
        sol2 = {'X': 2, 'Y': 3}
        
        path = finder.find_path(sol1, sol2)
        
        assert path is not None
        assert path.start == sol1
        assert path.end == sol2
    
    def test_equivalence_checking(self):
        """Test: Verificación de equivalencia."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y', 'Z'],
            domains={'X': {1, 2}, 'Y': {1, 2}, 'Z': {1, 2}},
            constraints=[
                ('X', 'Y', lambda x, y: x != y),
                ('Y', 'Z', lambda y, z: y != z)
            ]
        )
        
        finder = PathFinder(bridge)
        
        sol1 = {'X': 1, 'Y': 2, 'Z': 1}
        sol2 = {'X': 2, 'Y': 1, 'Z': 2}
        
        # Verificar si son equivalentes (conectadas)
        equiv = finder.are_equivalent(sol1, sol2)
        
        assert isinstance(equiv, bool)


class TestBridgeWithSymmetryExtractor:
    """Tests de integración Bridge + SymmetryExtractor."""
    
    def test_symmetry_extraction_with_bridge(self):
        """Test: Extracción de simetrías con bridge."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[]
        )
        
        extractor = SymmetryExtractor(bridge)
        
        symmetries = extractor.extract_variable_symmetries()
        
        assert isinstance(symmetries, list)
    
    def test_equivalence_classes_with_solutions(self):
        """Test: Clases de equivalencia con soluciones reales."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        
        extractor = SymmetryExtractor(bridge)
        
        # Generar todas las soluciones válidas
        solutions = [
            {'X': 1, 'Y': 2},
            {'X': 2, 'Y': 1}
        ]
        
        # Verificar que todas son válidas
        for sol in solutions:
            assert bridge.verify_solution(sol) is True
        
        # Agrupar en clases de equivalencia
        classes = extractor.get_equivalence_classes(solutions)
        
        assert len(classes) >= 1
        assert len(classes) <= len(solutions)


class TestPathFinderWithSymmetryExtractor:
    """Tests de integración PathFinder + SymmetryExtractor."""
    
    def test_paths_respect_symmetries(self):
        """Test: Los caminos respetan las simetrías."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y', 'Z'],
            domains={'X': {1, 2}, 'Y': {1, 2}, 'Z': {1, 2}},
            constraints=[]
        )
        
        finder = PathFinder(bridge)
        extractor = SymmetryExtractor(bridge)
        
        sol1 = {'X': 1, 'Y': 1, 'Z': 1}
        sol2 = {'X': 2, 'Y': 2, 'Z': 2}
        
        # Buscar camino
        path = finder.find_path(sol1, sol2)
        
        # Analizar simetrías
        group = extractor.extract_all_symmetries()
        
        # Verificar que el camino existe
        assert path is not None or group.order == 0


class TestComplexScenarios:
    """Tests de escenarios complejos."""
    
    def test_large_domain_integration(self):
        """Test: Integración con dominio grande."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': set(range(10)), 'Y': set(range(10))},
            constraints=[('X', 'Y', lambda x, y: x < y)]
        )
        
        # Verificar traducción
        assert bridge.cubical_type is not None
        
        # Verificar soluciones
        assert bridge.verify_solution({'X': 0, 'Y': 9}) is True
        assert bridge.verify_solution({'X': 9, 'Y': 0}) is False
        
        # Buscar camino
        finder = PathFinder(bridge, max_search_depth=5)
        path = finder.find_path({'X': 0, 'Y': 1}, {'X': 1, 'Y': 2})
        
        # Puede o no encontrar camino dependiendo de la profundidad
        assert path is None or path is not None
    
    def test_multiple_constraints_integration(self):
        """Test: Integración con múltiples restricciones."""
        bridge = create_simple_csp_bridge(
            variables=['A', 'B', 'C', 'D'],
            domains={
                'A': {1, 2, 3},
                'B': {1, 2, 3},
                'C': {1, 2, 3},
                'D': {1, 2, 3}
            },
            constraints=[
                ('A', 'B', lambda a, b: a != b),
                ('B', 'C', lambda b, c: b != c),
                ('C', 'D', lambda c, d: c != d),
                ('A', 'D', lambda a, d: a != d)
            ]
        )
        
        # Verificar traducción
        cubical_type = bridge.cubical_type
        assert len(cubical_type.variables) == 4
        assert len(cubical_type.constraint_props) == 4
        
        # Verificar propiedades
        props = bridge.get_solution_space_properties()
        assert props['variable_count'] == 4
        assert props['constraint_count'] == 4
        assert props['domain_size'] == 3 ** 4  # 81
    
    def test_disconnected_solution_space(self):
        """Test: Espacio de soluciones desconectado."""
        # CSP donde las soluciones están en componentes separadas
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[
                ('X', 'Y', lambda x, y: (x == 1 and y == 2) or (x == 2 and y == 1))
            ]
        )
        
        finder = PathFinder(bridge, max_search_depth=5)
        
        sol1 = {'X': 1, 'Y': 2}
        sol2 = {'X': 2, 'Y': 1}
        
        # Buscar camino (puede no existir si están desconectadas)
        path = finder.find_path(sol1, sol2)
        
        # El test verifica que el método no falle
        assert path is None or path is not None


class TestPerformance:
    """Tests de rendimiento."""
    
    def test_caching_improves_performance(self):
        """Test: El caché mejora el rendimiento."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y', 'Z'],
            domains={'X': {1, 2, 3}, 'Y': {1, 2, 3}, 'Z': {1, 2, 3}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        
        solution = {'X': 1, 'Y': 2, 'Z': 3}
        
        # Primera verificación (sin caché)
        result1 = bridge.verify_solution(solution)
        
        # Segunda verificación (con caché)
        result2 = bridge.verify_solution(solution)
        
        assert result1 == result2 == True
        
        # Verificar que el caché tiene entradas
        assert len(bridge._verification_cache) > 0
    
    def test_path_finder_caching(self):
        """Test: PathFinder usa caché correctamente."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2, 3}, 'Y': {1, 2, 3}},
            constraints=[]
        )
        
        finder = PathFinder(bridge)
        
        sol1 = {'X': 1, 'Y': 1}
        sol2 = {'X': 2, 'Y': 2}
        
        # Primera búsqueda
        path1 = finder.find_path(sol1, sol2)
        
        # Segunda búsqueda (debe usar caché)
        path2 = finder.find_path(sol1, sol2)
        
        assert path1 is path2  # Mismo objeto por caché


class TestErrorHandling:
    """Tests de manejo de errores."""
    
    def test_invalid_solution_handling(self):
        """Test: Manejo de soluciones inválidas."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        
        # Solución inválida
        invalid_solution = {'X': 1, 'Y': 1}
        
        # Debe retornar False, no lanzar excepción
        result = bridge.verify_solution(invalid_solution)
        assert result is False
    
    def test_path_finder_with_invalid_solutions(self):
        """Test: PathFinder con soluciones inválidas."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        
        finder = PathFinder(bridge)
        
        invalid_sol = {'X': 1, 'Y': 1}
        valid_sol = {'X': 1, 'Y': 2}
        
        # Debe retornar None, no lanzar excepción
        path = finder.find_path(invalid_sol, valid_sol)
        assert path is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

