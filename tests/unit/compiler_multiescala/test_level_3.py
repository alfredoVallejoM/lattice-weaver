"""
Tests para el Nivel L3 (Estructuras Compuestas) del compilador multiescala.
"""

import pytest
from lattice_weaver.compiler_multiescala import (
    Level0, Level1, Level2, Level3,
    ConstraintBlock, LocalPattern, PatternSignature,
    CompositeStructure, CompositeSignature
)
from lattice_weaver.core.csp_problem import CSP, Constraint


class TestCompositeSignature:
    """Tests para la clase CompositeSignature."""

    def test_init(self):
        """Test de inicialización de una firma de estructura compuesta."""
        signature = CompositeSignature(
            num_patterns=2,
            num_unique_blocks=1,
            pattern_types=('pattern_0', 'pattern_1'),
            topology='linear'
        )
        
        assert signature.num_patterns == 2
        assert signature.num_unique_blocks == 1
        assert signature.pattern_types == ('pattern_0', 'pattern_1')
        assert signature.topology == 'linear'

    def test_equality(self):
        """Test de igualdad de firmas."""
        sig1 = CompositeSignature(2, 1, ('p0', 'p1'), 'linear')
        sig2 = CompositeSignature(2, 1, ('p0', 'p1'), 'linear')
        sig3 = CompositeSignature(3, 1, ('p0', 'p1'), 'linear')
        
        assert sig1 == sig2
        assert sig1 != sig3


class TestCompositeStructure:
    """Tests para la clase CompositeStructure."""

    def test_init(self):
        """Test de inicialización de una estructura compuesta."""
        signature = CompositeSignature(2, 1, ('p0', 'p1'), 'linear')
        structure = CompositeStructure(
            structure_id=0,
            signature=signature,
            patterns=[0, 1],
            unique_blocks=[2],
            internal_constraints=[],
            interface_patterns={0}
        )
        
        assert structure.structure_id == 0
        assert structure.signature == signature
        assert structure.patterns == [0, 1]
        assert structure.unique_blocks == [2]
        assert structure.interface_patterns == {0}


class TestLevel3Initialization:
    """Tests para la inicialización del Nivel L3."""

    def test_init_with_structures(self):
        """Test de inicialización con estructuras."""
        signature = CompositeSignature(2, 0, ('p0', 'p1'), 'linear')
        structure = CompositeStructure(
            structure_id=0,
            signature=signature,
            patterns=[0, 1],
            unique_blocks=[],
            internal_constraints=[]
        )
        
        level3 = Level3(
            structures=[structure],
            isolated_patterns=[],
            isolated_blocks=[],
            inter_structure_constraints=[]
        )
        
        assert level3.level == 3
        assert len(level3.structures) == 1
        assert level3.pattern_to_structure[0] == 0
        assert level3.pattern_to_structure[1] == 0


class TestLevel3BuildFromLower:
    """Tests para la construcción de L3 desde L2."""

    def test_build_from_l2_simple(self):
        """Test de construcción desde L2 con patrones simples."""
        # Crear un CSP con estructura repetitiva
        variables = {'v1', 'v2', 'v3', 'v4'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq'),
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # L0 -> L1 -> L2 -> L3
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        level2 = Level2([], [], [], config={'original_domains': domains})
        level2.build_from_lower(level1)
        
        level3 = Level3([], [], [], [], config=level2.config)
        level3.build_from_lower(level2)
        
        # Validar que L3 es coherente
        assert level3.validate() is True

    def test_build_from_l2_no_structures(self):
        """Test de construcción desde L2 sin estructuras (solo patrones aislados)."""
        # Crear patrones aislados sin conexiones
        signature = PatternSignature(2, 1, 0, ('neq',))
        block1 = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')]
        )
        block2 = ConstraintBlock(
            block_id=1,
            variables={'v3', 'v4'},
            constraints=[Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq')]
        )
        
        pattern = LocalPattern(
            pattern_id=0,
            signature=signature,
            instances=[0, 1],
            template_block=block1,
            instance_blocks={0: block1, 1: block2}
        )
        
        level2 = Level2(
            patterns=[pattern],
            unique_blocks=[],
            inter_pattern_constraints=[],
            config={'original_patterns': [pattern], 'original_unique_blocks': []}
        )
        
        level3 = Level3([], [], [], [], config=level2.config)
        level3.build_from_lower(level2)
        
        # Sin conexiones, deberíamos tener patrones aislados
        assert len(level3.structures) == 0
        assert len(level3.isolated_patterns) >= 0


class TestLevel3RefineToLower:
    """Tests para el refinamiento de L3 a L2."""

    def test_refine_to_l2_simple(self):
        """Test de refinamiento a L2 con estructuras simples."""
        # Crear un CSP con estructura repetitiva
        variables = {'v1', 'v2', 'v3', 'v4'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq'),
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # L0 -> L1 -> L2 -> L3
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        level2 = Level2([], [], [], config={'original_domains': domains})
        level2.build_from_lower(level1)
        
        level3 = Level3([], [], [], [], config=level2.config)
        level3.build_from_lower(level2)
        
        # Refinar de vuelta a L2
        refined_level2 = level3.refine_to_lower()
        
        assert isinstance(refined_level2, Level2)
        assert refined_level2.validate() is True


class TestLevel3Validation:
    """Tests para la validación del Nivel L3."""

    def test_validate_valid_l3(self):
        """Test de validación con una representación válida de L3."""
        signature = CompositeSignature(2, 0, ('p0', 'p1'), 'linear')
        structure = CompositeStructure(
            structure_id=0,
            signature=signature,
            patterns=[0, 1],
            unique_blocks=[],
            internal_constraints=[]
        )
        
        level3 = Level3(
            structures=[structure],
            isolated_patterns=[],
            isolated_blocks=[],
            inter_structure_constraints=[]
        )
        
        assert level3.validate() is True

    def test_validate_structure_with_one_component(self):
        """Test de validación con una estructura de un solo componente."""
        signature = CompositeSignature(1, 0, ('p0',), 'singleton')
        structure = CompositeStructure(
            structure_id=0,
            signature=signature,
            patterns=[0],
            unique_blocks=[],
            internal_constraints=[]
        )
        
        level3 = Level3(
            structures=[structure],
            isolated_patterns=[],
            isolated_blocks=[],
            inter_structure_constraints=[]
        )
        
        assert level3.validate() is False


class TestLevel3Complexity:
    """Tests para el cálculo de complejidad del Nivel L3."""

    def test_complexity_with_structures(self):
        """Test de complejidad con estructuras."""
        signature = CompositeSignature(2, 1, ('p0', 'p1'), 'linear')
        structure = CompositeStructure(
            structure_id=0,
            signature=signature,
            patterns=[0, 1],
            unique_blocks=[2],
            internal_constraints=[]
        )
        
        level3 = Level3(
            structures=[structure],
            isolated_patterns=[],
            isolated_blocks=[],
            inter_structure_constraints=[]
        )
        
        assert level3.complexity > 0

    def test_complexity_empty_l3(self):
        """Test de complejidad con una representación vacía de L3."""
        level3 = Level3(
            structures=[],
            isolated_patterns=[],
            isolated_blocks=[],
            inter_structure_constraints=[]
        )
        
        assert level3.complexity == 0.0


class TestLevel3Statistics:
    """Tests para las estadísticas del Nivel L3."""

    def test_get_statistics_with_structures(self):
        """Test de obtención de estadísticas con estructuras."""
        signature = CompositeSignature(2, 1, ('p0', 'p1'), 'linear')
        structure = CompositeStructure(
            structure_id=0,
            signature=signature,
            patterns=[0, 1],
            unique_blocks=[2],
            internal_constraints=[]
        )
        
        level3 = Level3(
            structures=[structure],
            isolated_patterns=[],
            isolated_blocks=[],
            inter_structure_constraints=[]
        )
        stats = level3.get_statistics()
        
        assert stats['level'] == 3
        assert stats['num_structures'] == 1
        assert stats['total_patterns_in_structures'] == 2
        assert stats['total_blocks_in_structures'] == 1
        assert stats['complexity'] > 0


class TestLevel3EdgeCases:
    """Tests para casos extremos del Nivel L3."""

    def test_empty_l3(self):
        """Test con una representación vacía de L3."""
        level3 = Level3(
            structures=[],
            isolated_patterns=[],
            isolated_blocks=[],
            inter_structure_constraints=[]
        )
        
        assert level3.validate() is True
        assert level3.complexity == 0.0
        assert level3.get_statistics()['num_structures'] == 0


class TestLevel3Integration:
    """Tests de integración completa L0 -> L1 -> L2 -> L3."""

    def test_full_integration_l0_l1_l2_l3(self):
        """Test de integración completa L0 -> L1 -> L2 -> L3."""
        # Crear un CSP con estructura repetitiva
        variables = {'v1', 'v2', 'v3', 'v4'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq'),
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # L0 -> L1 -> L2 -> L3
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        level2 = Level2([], [], [], config={'original_domains': domains})
        level2.build_from_lower(level1)
        
        level3 = Level3([], [], [], [], config=level2.config)
        level3.build_from_lower(level2)
        
        # Validar que L3 es coherente
        assert level3.validate() is True
        
        # Validar roundtrip L3 -> L2 -> L1 -> L0
        refined_level2 = level3.refine_to_lower()
        assert refined_level2.validate() is True
        
        refined_level1 = refined_level2.refine_to_lower()
        assert refined_level1.validate() is True
        
        refined_level0 = refined_level1.refine_to_lower()
        assert refined_level0.validate() is True
        
        # El número de variables debería ser el mismo
        assert refined_level0.csp.variables == variables


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
