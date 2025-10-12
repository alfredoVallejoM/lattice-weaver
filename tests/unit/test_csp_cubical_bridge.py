"""
Tests Unitarios para CSPToCubicalBridge

Verifica la correcta integración entre el motor CSP (ArcEngine) y
el sistema de tipos cúbicos.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import pytest
from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.formal.csp_cubical_bridge import (
    CSPToCubicalBridge,
    create_bridge_from_arc_engine,
    create_simple_csp_bridge
)
from lattice_weaver.formal.cubical_csp_type import CubicalCSPType


class TestCSPToCubicalBridgeConstruction:
    """Tests para construcción del bridge."""
    
    def test_create_bridge_from_arc_engine(self):
        """Test: Crear bridge desde ArcEngine."""
        engine = ArcEngine()
        engine.add_variable('X', {1, 2, 3})
        engine.add_variable('Y', {1, 2, 3})
        
        bridge = CSPToCubicalBridge(engine)
        
        assert bridge.arc_engine is engine
        assert bridge.cubical_type is not None
        assert isinstance(bridge.cubical_type, CubicalCSPType)
    
    def test_create_bridge_with_constraints(self):
        """Test: Crear bridge con restricciones."""
        engine = ArcEngine()
        engine.add_variable('X', {1, 2, 3})
        engine.add_variable('Y', {1, 2, 3})
        engine.add_constraint('X', 'Y', lambda x, y: x < y, cid='X_lt_Y')
        
        bridge = CSPToCubicalBridge(engine)
        
        assert bridge.cubical_type is not None
        assert len(bridge.cubical_type.variables) == 2
        assert len(bridge.cubical_type.constraint_props) == 1
    
    def test_create_simple_csp_bridge(self):
        """Test: Crear bridge con función de utilidad."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        
        assert bridge.cubical_type is not None
        assert len(bridge.cubical_type.variables) == 2
        assert len(bridge.cubical_type.constraint_props) == 1
    
    def test_create_bridge_from_arc_engine_function(self):
        """Test: Función create_bridge_from_arc_engine."""
        engine = ArcEngine()
        engine.add_variable('X', {1, 2})
        
        bridge = create_bridge_from_arc_engine(engine)
        
        assert isinstance(bridge, CSPToCubicalBridge)
        assert bridge.arc_engine is engine


class TestTranslation:
    """Tests para traducción CSP → Tipo Cúbico."""
    
    def test_translate_simple_csp(self):
        """Test: Traducir CSP simple."""
        engine = ArcEngine()
        engine.add_variable('X', {1, 2, 3})
        engine.add_variable('Y', {1, 2, 3})
        
        bridge = CSPToCubicalBridge(engine)
        cubical_type = bridge.translate_to_cubical_type()
        
        assert cubical_type is not None
        assert len(cubical_type.variables) == 2
        assert 'X' in cubical_type.variables
        assert 'Y' in cubical_type.variables
    
    def test_translate_with_constraints(self):
        """Test: Traducir CSP con restricciones."""
        engine = ArcEngine()
        engine.add_variable('X', {1, 2, 3})
        engine.add_variable('Y', {1, 2, 3})
        engine.add_constraint('X', 'Y', lambda x, y: x < y, cid='X_lt_Y')
        
        bridge = CSPToCubicalBridge(engine)
        cubical_type = bridge.translate_to_cubical_type()
        
        assert len(cubical_type.constraint_props) == 1
        assert cubical_type.constraint_props[0].constraint_name == 'X_lt_Y'
    
    def test_translate_caching(self):
        """Test: Caché de traducción."""
        engine = ArcEngine()
        engine.add_variable('X', {1, 2})
        
        bridge = CSPToCubicalBridge(engine)
        
        # Primera traducción
        type1 = bridge.translate_to_cubical_type()
        
        # Segunda traducción (debe usar caché)
        type2 = bridge.translate_to_cubical_type()
        
        # Deben ser el mismo objeto (por caché)
        assert type1 is type2
    
    def test_translate_different_domains(self):
        """Test: Traducir con dominios diferentes."""
        engine = ArcEngine()
        engine.add_variable('X', {1, 2, 3})
        engine.add_variable('Y', {4, 5, 6})
        
        bridge = CSPToCubicalBridge(engine)
        cubical_type = bridge.translate_to_cubical_type()
        
        domain_x = cubical_type.domain_types['X']
        domain_y = cubical_type.domain_types['Y']
        
        assert 1 in domain_x.values
        assert 4 in domain_y.values
        assert 1 not in domain_y.values


class TestSolutionVerification:
    """Tests para verificación de soluciones."""
    
    def test_verify_valid_solution(self):
        """Test: Verificar solución válida."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2, 3}, 'Y': {1, 2, 3}},
            constraints=[('X', 'Y', lambda x, y: x < y)]
        )
        
        solution = {'X': 1, 'Y': 2}
        assert bridge.verify_solution(solution) is True
    
    def test_verify_invalid_solution(self):
        """Test: Rechazar solución inválida."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2, 3}, 'Y': {1, 2, 3}},
            constraints=[('X', 'Y', lambda x, y: x < y)]
        )
        
        solution = {'X': 2, 'Y': 1}
        assert bridge.verify_solution(solution) is False
    
    def test_verify_solution_caching(self):
        """Test: Caché de verificación."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        
        solution = {'X': 1, 'Y': 2}
        
        # Primera verificación
        result1 = bridge.verify_solution(solution)
        
        # Segunda verificación (debe usar caché)
        result2 = bridge.verify_solution(solution)
        
        assert result1 == result2 == True
    
    def test_verify_multiple_constraints(self):
        """Test: Verificar con múltiples restricciones."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y', 'Z'],
            domains={'X': {1, 2, 3}, 'Y': {1, 2, 3}, 'Z': {1, 2, 3}},
            constraints=[
                ('X', 'Y', lambda x, y: x < y),
                ('Y', 'Z', lambda y, z: y < z)
            ]
        )
        
        # Solución válida
        solution1 = {'X': 1, 'Y': 2, 'Z': 3}
        assert bridge.verify_solution(solution1) is True
        
        # Solución inválida (primera restricción falla)
        solution2 = {'X': 2, 'Y': 1, 'Z': 3}
        assert bridge.verify_solution(solution2) is False
        
        # Solución inválida (segunda restricción falla)
        solution3 = {'X': 1, 'Y': 2, 'Z': 1}
        assert bridge.verify_solution(solution3) is False
    
    def test_verify_solution_with_proof(self):
        """Test: Verificar solución con prueba."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        
        solution = {'X': 1, 'Y': 2}
        is_valid, proof = bridge.verify_solution_with_proof(solution)
        
        assert is_valid is True
        assert proof is not None
        assert str(proof) == "(1, (2, ()))"
    
    def test_verify_invalid_solution_no_proof(self):
        """Test: Solución inválida no tiene prueba."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        
        solution = {'X': 1, 'Y': 1}
        is_valid, proof = bridge.verify_solution_with_proof(solution)
        
        assert is_valid is False
        assert proof is None


class TestSolutionToTerm:
    """Tests para conversión Solución → Término."""
    
    def test_solution_to_term_simple(self):
        """Test: Convertir solución simple a término."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[]
        )
        
        solution = {'X': 1, 'Y': 2}
        term = bridge.solution_to_term(solution)
        
        assert term is not None
        assert str(term) == "(1, (2, ()))"
    
    def test_solution_to_term_invalid(self):
        """Test: Convertir solución inválida debe fallar."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        
        solution = {'X': 1, 'Y': 1}  # Inválida
        
        with pytest.raises(ValueError):
            bridge.solution_to_term(solution)


class TestSolutionSpaceProperties:
    """Tests para extracción de propiedades del espacio."""
    
    def test_get_solution_space_properties(self):
        """Test: Obtener propiedades del espacio de soluciones."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2, 3}, 'Y': {1, 2, 3}},
            constraints=[('X', 'Y', lambda x, y: x < y)]
        )
        
        props = bridge.get_solution_space_properties()
        
        assert props['domain_size'] == 9  # 3 * 3
        assert props['constraint_count'] == 1
        assert props['variable_count'] == 2
        assert props['type_complexity'] == 3  # 2 vars + 1 constraint
        assert props['variables'] == ['X', 'Y']
        assert 'X' in props['domain_types']
        assert 'Y' in props['domain_types']
    
    def test_properties_large_domain(self):
        """Test: Propiedades con dominio grande."""
        bridge = create_simple_csp_bridge(
            variables=['A', 'B', 'C'],
            domains={'A': {1, 2, 3, 4}, 'B': {1, 2, 3}, 'C': {1, 2}},
            constraints=[]
        )
        
        props = bridge.get_solution_space_properties()
        
        assert props['domain_size'] == 4 * 3 * 2  # 24
        assert props['constraint_count'] == 0
        assert props['variable_count'] == 3


class TestCaching:
    """Tests para sistema de caché."""
    
    def test_clear_cache(self):
        """Test: Limpiar caché."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[]
        )
        
        # Verificar solución (cachear)
        solution = {'X': 1, 'Y': 2}
        bridge.verify_solution(solution)
        
        # Limpiar caché
        bridge.clear_cache()
        
        # Caché debe estar vacía
        assert len(bridge._verification_cache) == 0
        assert len(bridge._translation_cache) == 0
    
    def test_hash_csp_different_domains(self):
        """Test: Hash diferente para dominios diferentes."""
        engine1 = ArcEngine()
        engine1.add_variable('X', {1, 2})
        bridge1 = CSPToCubicalBridge(engine1)
        hash1 = bridge1._hash_csp()
        
        engine2 = ArcEngine()
        engine2.add_variable('X', {1, 2, 3})  # Dominio diferente
        bridge2 = CSPToCubicalBridge(engine2)
        hash2 = bridge2._hash_csp()
        
        assert hash1 != hash2
    
    def test_hash_csp_different_constraints(self):
        """Test: Hash diferente para restricciones diferentes."""
        engine1 = ArcEngine()
        engine1.add_variable('X', {1, 2})
        engine1.add_variable('Y', {1, 2})
        engine1.add_constraint('X', 'Y', lambda x, y: x < y, cid='C1')
        bridge1 = CSPToCubicalBridge(engine1)
        hash1 = bridge1._hash_csp()
        
        engine2 = ArcEngine()
        engine2.add_variable('X', {1, 2})
        engine2.add_variable('Y', {1, 2})
        engine2.add_constraint('X', 'Y', lambda x, y: x != y, cid='C2')  # Diferente
        bridge2 = CSPToCubicalBridge(engine2)
        hash2 = bridge2._hash_csp()
        
        assert hash1 != hash2


class TestEdgeCases:
    """Tests para casos extremos."""
    
    def test_empty_csp(self):
        """Test: CSP vacío."""
        engine = ArcEngine()
        bridge = CSPToCubicalBridge(engine)
        
        assert bridge.cubical_type is not None
        assert len(bridge.cubical_type.variables) == 0
    
    def test_single_variable(self):
        """Test: CSP con una sola variable."""
        engine = ArcEngine()
        engine.add_variable('X', {1, 2, 3})
        
        bridge = CSPToCubicalBridge(engine)
        
        assert len(bridge.cubical_type.variables) == 1
        assert bridge.cubical_type.get_domain_size() == 3
    
    def test_no_constraints(self):
        """Test: CSP sin restricciones."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[]
        )
        
        assert len(bridge.cubical_type.constraint_props) == 0
        
        # Todas las soluciones deben ser válidas
        assert bridge.verify_solution({'X': 1, 'Y': 1}) is True
        assert bridge.verify_solution({'X': 1, 'Y': 2}) is True
        assert bridge.verify_solution({'X': 2, 'Y': 1}) is True
        assert bridge.verify_solution({'X': 2, 'Y': 2}) is True
    
    def test_large_domain(self):
        """Test: Dominio grande."""
        bridge = create_simple_csp_bridge(
            variables=['X'],
            domains={'X': set(range(100))},
            constraints=[]
        )
        
        assert bridge.cubical_type.get_domain_size() == 100


class TestStringRepresentation:
    """Tests para representación en string."""
    
    def test_str_representation(self):
        """Test: Representación en string del bridge."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[]
        )
        
        str_repr = str(bridge)
        assert "CSPToCubicalBridge" in str_repr
    
    def test_repr_representation(self):
        """Test: Representación detallada del bridge."""
        bridge = create_simple_csp_bridge(
            variables=['X', 'Y'],
            domains={'X': {1, 2}, 'Y': {1, 2}},
            constraints=[('X', 'Y', lambda x, y: x != y)]
        )
        
        repr_str = repr(bridge)
        assert "CSPToCubicalBridge" in repr_str
        assert "variables=2" in repr_str
        assert "constraints=1" in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

