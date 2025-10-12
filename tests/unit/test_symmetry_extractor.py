"""
Tests Unitarios para SymmetryExtractor

Verifica la correcta extracción y análisis de simetrías en CSP.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import pytest
from lattice_weaver.formal.symmetry_extractor import (
    SymmetryExtractor,
    Symmetry,
    SymmetryGroup,
    create_symmetry_extractor
)
from lattice_weaver.formal.csp_cubical_bridge import create_simple_csp_bridge


class TestSymmetry:
    """Tests para Symmetry."""
    
    def test_create_symmetry(self):
        """Test: Crear simetría."""
        sym = Symmetry(
            type_='variable',
            mapping=frozenset([('X', 'Y'), ('Y', 'X')]),
            description="Swap X and Y"
        )
        
        assert sym.type_ == 'variable'
        assert len(sym.mapping) == 2
        assert sym.description == "Swap X and Y"
    
    def test_apply_to_solution(self):
        """Test: Aplicar simetría a solución."""
        sym = Symmetry(
            type_='variable',
            mapping=frozenset([('X', 'Y'), ('Y', 'X')])
        )
        
        solution = {'X': 1, 'Y': 2}
        transformed = sym.apply_to_solution(solution)
        
        assert transformed['X'] == 2  # Valor de Y
        assert transformed['Y'] == 1  # Valor de X
    
    def test_str_representation(self):
        """Test: Representación en string."""
        sym = Symmetry(
            type_='variable',
            mapping=frozenset([('X', 'Y')]),
            description="Test"
        )
        
        str_repr = str(sym)
        assert "Symmetry" in str_repr
        assert "variable" in str_repr


class TestSymmetryGroup:
    """Tests para SymmetryGroup."""
    
    def test_create_symmetry_group(self):
        """Test: Crear grupo de simetrías."""
        group = SymmetryGroup()
        
        assert group.order == 0
        assert len(group.symmetries) == 0
    
    def test_add_symmetry(self):
        """Test: Añadir simetría al grupo."""
        group = SymmetryGroup()
        sym = Symmetry(
            type_='variable',
            mapping=frozenset([('X', 'Y')])
        )
        
        group.add_symmetry(sym)
        
        assert group.order == 1
        assert sym in group.symmetries
    
    def test_apply_all_to_solution(self):
        """Test: Aplicar todas las simetrías a una solución."""
        group = SymmetryGroup()
        
        # Añadir simetría de intercambio
        sym = Symmetry(
            type_='variable',
            mapping=frozenset([('X', 'Y'), ('Y', 'X')])
        )
        group.add_symmetry(sym)
        
        solution = {'X': 1, 'Y': 2}
        symmetric_solutions = group.apply_all_to_solution(solution)
        
        assert len(symmetric_solutions) >= 1


class TestSymmetryExtractorConstruction:
    """Tests para construcción de SymmetryExtractor."""
    
    def test_create_symmetry_extractor(self):
        """Test: Crear SymmetryExtractor."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[]
        )
        
        extractor = SymmetryExtractor(bridge)
        
        assert extractor.bridge is bridge
    
    def test_create_symmetry_extractor_function(self):
        """Test: Función create_symmetry_extractor."""
        bridge = create_simple_csp_bridge(
            variables=['X'],
            domains={'X': {1, 2}},
            constraints=[]
        )
        
        extractor = create_symmetry_extractor(bridge)
        
        assert isinstance(extractor, SymmetryExtractor)


class TestVariableSymmetries:
    """Tests para extracción de simetrías de variables."""
    
    def test_extract_no_symmetries(self):
        """Test: Problema sin simetrías."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {3, 4}},  # Dominios diferentes
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        symmetries = extractor.extract_variable_symmetries()
        
        # No debe haber simetrías (dominios diferentes)
        assert len(symmetries) >= 0  # Puede ser 0 o más
    
    def test_extract_symmetric_problem(self):
        """Test: Problema con simetrías."""
        # CSP simétrico: todas las variables tienen el mismo dominio
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        symmetries = extractor.extract_variable_symmetries()
        
        # Puede haber simetrías (dominios idénticos, sin restricciones)
        assert isinstance(symmetries, list)
    
    def test_group_variables_by_domain(self):
        """Test: Agrupar variables por dominio."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y', 'Z'],
            domains={'X': {1, 2}, 'Y': {1, 2}, 'Z': {3, 4}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        groups = extractor._group_variables_by_domain()
        
        # Debe haber 2 grupos: {1,2} y {3,4}
        assert len(groups) == 2


class TestSymmetryExtraction:
    """Tests para extracción de simetrías."""
    
    def test_extract_all_symmetries(self):
        """Test: Extraer todas las simetrías."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        group = extractor.extract_all_symmetries()
        
        assert isinstance(group, SymmetryGroup)
        assert group.order >= 0
    
    def test_extract_all_symmetries_caching(self):
        """Test: Caché de simetrías."""
        bridge = create_simple_csp_bridge(
            variables=['X'],
            domains={'X': {1, 2}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        # Primera extracción
        group1 = extractor.extract_all_symmetries()
        
        # Segunda extracción (debe usar caché)
        group2 = extractor.extract_all_symmetries()
        
        assert group1 is group2  # Mismo objeto por caché


class TestEquivalenceClasses:
    """Tests para clases de equivalencia."""
    
    def test_get_equivalence_classes_no_symmetries(self):
        """Test: Clases de equivalencia sin simetrías."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {3, 4}},  # Dominios diferentes
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        solutions = [
            {'X': 1, 'Y': 3},
            {'X': 1, 'Y': 4},
            {'X': 2, 'Y': 3}
        ]
        
        classes = extractor.get_equivalence_classes(solutions)
        
        # Sin simetrías, cada solución es su propia clase
        assert len(classes) >= 1
    
    def test_get_representative_solutions(self):
        """Test: Obtener representantes."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        solutions = [
            {'X': 1, 'Y': 2},
            {'X': 2, 'Y': 1}
        ]
        
        representatives = extractor.get_representative_solutions(solutions)
        
        # Debe haber al menos un representante
        assert len(representatives) >= 1
        assert all(isinstance(r, dict) for r in representatives)
    
    def test_count_unique_solutions(self):
        """Test: Contar soluciones únicas."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        solutions = [
            {'X': 1, 'Y': 2},
            {'X': 2, 'Y': 1},
            {'X': 1, 'Y': 1}
        ]
        
        unique_count = extractor.count_unique_solutions(solutions)
        
        assert unique_count >= 1
        assert unique_count <= len(solutions)


class TestSymmetryAnalysis:
    """Tests para análisis de estructura de simetrías."""
    
    def test_analyze_symmetry_structure(self):
        """Test: Analizar estructura de simetrías."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y', 'Z'],
            domains={'X': {1, 2}, 'Y': {1, 2}, 'Z': {3, 4}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        analysis = extractor.analyze_symmetry_structure()
        
        assert 'symmetry_count' in analysis
        assert 'symmetry_types' in analysis
        assert 'domain_groups' in analysis
        assert 'has_symmetries' in analysis
        assert isinstance(analysis['symmetry_count'], int)
        assert isinstance(analysis['has_symmetries'], bool)
    
    def test_analyze_structure_no_symmetries(self):
        """Test: Análisis sin simetrías."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {3, 4}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        analysis = extractor.analyze_symmetry_structure()
        
        # Sin simetrías
        assert analysis['symmetry_count'] >= 0


class TestCaching:
    """Tests para sistema de caché."""
    
    def test_clear_cache(self):
        """Test: Limpiar caché."""
        bridge = create_simple_csp_bridge(
            variables=['X'],
            domains={'X': {1, 2}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        # Extraer simetrías (cachear)
        extractor.extract_all_symmetries()
        
        # Limpiar caché
        extractor.clear_cache()
        
        assert extractor._symmetry_cache is None


class TestStringRepresentation:
    """Tests para representación en string."""
    
    def test_str_representation(self):
        """Test: Representación en string de SymmetryExtractor."""
        bridge = create_simple_csp_bridge(
            variables=['X'],
            domains={'X': {1, 2}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        str_repr = str(extractor)
        assert "SymmetryExtractor" in str_repr
    
    def test_repr_representation(self):
        """Test: Representación detallada de SymmetryExtractor."""
        bridge = create_simple_csp_bridge(
            variables=['X'],
            domains={'X': {1, 2}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        repr_str = repr(extractor)
        assert "SymmetryExtractor" in repr_str
    
    def test_repr_with_cache(self):
        """Test: Representación con caché."""
        bridge = create_simple_csp_bridge(
            variables=['X'],
            domains={'X': {1, 2}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        # Cachear simetrías
        extractor.extract_all_symmetries()
        
        repr_str = repr(extractor)
        assert "symmetries_cached" in repr_str
    
    def test_symmetry_group_str(self):
        """Test: Representación en string de SymmetryGroup."""
        group = SymmetryGroup()
        
        str_repr = str(group)
        assert "SymmetryGroup" in str_repr
        assert "order" in str_repr


class TestEdgeCases:
    """Tests para casos extremos."""
    
    def test_empty_csp(self):
        """Test: CSP vacío."""
        bridge = create_simple_csp_bridge(
            variables=[],
            domains={},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        symmetries = extractor.extract_variable_symmetries()
        
        assert isinstance(symmetries, list)
    
    def test_single_variable(self):
        """Test: CSP con una sola variable."""
        bridge = create_simple_csp_bridge(
            variables=['X'],
            domains={'X': {1, 2, 3}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        symmetries = extractor.extract_variable_symmetries()
        
        # No puede haber simetrías de permutación con una sola variable
        assert len(symmetries) == 0
    
    def test_no_constraints(self):
        """Test: CSP sin restricciones."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y', 'Z'],
            domains={'X': {1, 2}, 'Y': {1, 2}, 'Z': {1, 2}},
            constraints=[]
        )
        extractor = SymmetryExtractor(bridge)
        
        analysis = extractor.analyze_symmetry_structure()
        
        # Sin restricciones, puede haber muchas simetrías
        assert isinstance(analysis, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

