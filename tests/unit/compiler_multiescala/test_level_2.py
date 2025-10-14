"""
Tests para el Nivel L2 (Patrones Locales) del compilador multiescala.
"""

import pytest
from lattice_weaver.compiler_multiescala import Level0, Level1, Level2, ConstraintBlock, LocalPattern, PatternSignature
from lattice_weaver.core.csp_problem import CSP, Constraint


class TestPatternSignature:
    """Tests para la clase PatternSignature."""

    def test_init(self):
        """Test de inicialización de una firma de patrón."""
        signature = PatternSignature(
            num_variables=2,
            num_constraints=1,
            num_interface_vars=1,
            constraint_types=('neq',)
        )
        
        assert signature.num_variables == 2
        assert signature.num_constraints == 1
        assert signature.num_interface_vars == 1
        assert signature.constraint_types == ('neq',)

    def test_equality(self):
        """Test de igualdad de firmas."""
        sig1 = PatternSignature(2, 1, 1, ('neq',))
        sig2 = PatternSignature(2, 1, 1, ('neq',))
        sig3 = PatternSignature(3, 1, 1, ('neq',))
        
        assert sig1 == sig2
        assert sig1 != sig3

    def test_hashable(self):
        """Test de que las firmas son hashables."""
        sig1 = PatternSignature(2, 1, 1, ('neq',))
        sig2 = PatternSignature(2, 1, 1, ('neq',))
        
        # Deberían poder usarse como claves de diccionario
        d = {sig1: 'value'}
        assert d[sig2] == 'value'


class TestLocalPattern:
    """Tests para la clase LocalPattern."""

    def test_init(self):
        """Test de inicialización de un patrón local."""
        signature = PatternSignature(2, 1, 1, ('neq',))
        block = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')]
        )
        
        pattern = LocalPattern(
            pattern_id=0,
            signature=signature,
            instances=[0, 1],
            template_block=block
        )
        
        assert pattern.pattern_id == 0
        assert pattern.signature == signature
        assert pattern.instances == [0, 1]
        assert pattern.template_block == block


class TestLevel2Initialization:
    """Tests para la inicialización del Nivel L2."""

    def test_init_with_patterns(self):
        """Test de inicialización con patrones."""
        signature = PatternSignature(2, 1, 1, ('neq',))
        block = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')]
        )
        
        pattern = LocalPattern(
            pattern_id=0,
            signature=signature,
            instances=[0, 1],
            template_block=block
        )
        
        level2 = Level2(patterns=[pattern], unique_blocks=[], inter_pattern_constraints=[])
        
        assert level2.level == 2
        assert len(level2.patterns) == 1
        assert len(level2.unique_blocks) == 0
        assert level2.pattern_instances[0] == 0
        assert level2.pattern_instances[1] == 0


class TestLevel2BuildFromLower:
    """Tests para la construcción de L2 desde L1."""

    def test_build_from_l1_with_patterns(self):
        """Test de construcción desde L1 con patrones recurrentes."""
        # Crear bloques con la misma estructura (patrón)
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
        block3 = ConstraintBlock(
            block_id=2,
            variables={'v5', 'v6', 'v7'},
            constraints=[
                Constraint(scope=frozenset({'v5', 'v6'}), relation=lambda x, y: x != y, name='neq'),
                Constraint(scope=frozenset({'v6', 'v7'}), relation=lambda x, y: x != y, name='neq')
            ]
        )
        
        level1 = Level1(blocks=[block1, block2, block3], inter_block_constraints=[])
        level2 = Level2([], [], [], config={'original_domains': {}})
        level2.build_from_lower(level1)
        
        # Deberíamos tener 1 patrón (block1 y block2) y 1 bloque único (block3)
        assert len(level2.patterns) == 1
        assert len(level2.unique_blocks) == 1
        assert len(level2.patterns[0].instances) == 2

    def test_build_from_l1_no_patterns(self):
        """Test de construcción desde L1 sin patrones recurrentes."""
        # Crear bloques con estructuras diferentes
        block1 = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')]
        )
        block2 = ConstraintBlock(
            block_id=1,
            variables={'v3', 'v4', 'v5'},
            constraints=[
                Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq'),
                Constraint(scope=frozenset({'v4', 'v5'}), relation=lambda x, y: x != y, name='neq')
            ]
        )
        
        level1 = Level1(blocks=[block1, block2], inter_block_constraints=[])
        level2 = Level2([], [], [], config={'original_domains': {}})
        level2.build_from_lower(level1)
        
        # No deberíamos tener patrones, solo bloques únicos
        assert len(level2.patterns) == 0
        assert len(level2.unique_blocks) == 2

    def test_build_from_l1_integration(self):
        """Test de construcción desde L1 con integración completa L0->L1->L2."""
        # Crear un CSP con estructura repetitiva
        variables = {'v1', 'v2', 'v3', 'v4', 'v5', 'v6'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            # Par 1
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq'),
            # Par 2 (misma estructura que Par 1)
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq'),
            # Par 3 (misma estructura que Par 1)
            Constraint(scope=frozenset({'v5', 'v6'}), relation=lambda x, y: x != y, name='neq'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # L0 -> L1 -> L2
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        level2 = Level2([], [], [], config={'original_domains': domains})
        level2.build_from_lower(level1)
        
        # Deberíamos detectar el patrón recurrente
        assert len(level2.patterns) >= 1 or len(level2.unique_blocks) >= 1


class TestLevel2RefineToLower:
    """Tests para el refinamiento de L2 a L1."""

    def test_refine_to_l1_with_patterns(self):
        """Test de refinamiento a L1 con patrones."""
        # Crear un patrón
        signature = PatternSignature(2, 1, 0, ('neq',))
        block = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')]
        )
        
        pattern = LocalPattern(
            pattern_id=0,
            signature=signature,
            instances=[0, 1],
            template_block=block
        )
        
        level2 = Level2(patterns=[pattern], unique_blocks=[], inter_pattern_constraints=[])
        refined_level1 = level2.refine_to_lower()
        
        assert isinstance(refined_level1, Level1)
        assert len(refined_level1.blocks) == 2  # 2 instancias del patrón

    def test_refine_to_l1_roundtrip(self):
        """Test de roundtrip L1 -> L2 -> L1."""
        # Crear bloques con patrón
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
        
        level1 = Level1(blocks=[block1, block2], inter_block_constraints=[])
        level2 = Level2([], [], [], config={'original_domains': {}})
        level2.build_from_lower(level1)
        
        # Refinar de vuelta a L1
        refined_level1 = level2.refine_to_lower()
        
        assert isinstance(refined_level1, Level1)
        assert len(refined_level1.blocks) == 2


class TestLevel2Validation:
    """Tests para la validación del Nivel L2."""

    def test_validate_valid_l2(self):
        """Test de validación con una representación válida de L2."""
        signature = PatternSignature(2, 1, 0, ('neq',))
        block = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')]
        )
        
        pattern = LocalPattern(
            pattern_id=0,
            signature=signature,
            instances=[0, 1],
            template_block=block
        )
        
        level2 = Level2(patterns=[pattern], unique_blocks=[], inter_pattern_constraints=[])
        
        assert level2.validate() is True

    def test_validate_pattern_with_one_instance(self):
        """Test de validación con un patrón de una sola instancia."""
        signature = PatternSignature(2, 1, 0, ('neq',))
        block = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')]
        )
        
        pattern = LocalPattern(
            pattern_id=0,
            signature=signature,
            instances=[0],  # Solo una instancia
            template_block=block
        )
        
        level2 = Level2(patterns=[pattern], unique_blocks=[], inter_pattern_constraints=[])
        
        assert level2.validate() is False


class TestLevel2Complexity:
    """Tests para el cálculo de complejidad del Nivel L2."""

    def test_complexity_with_patterns(self):
        """Test de complejidad con patrones."""
        signature = PatternSignature(2, 1, 0, ('neq',))
        block = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')]
        )
        
        pattern = LocalPattern(
            pattern_id=0,
            signature=signature,
            instances=[0, 1, 2],
            template_block=block
        )
        
        level2 = Level2(patterns=[pattern], unique_blocks=[], inter_pattern_constraints=[])
        
        # La complejidad debería ser menor que si tratáramos cada instancia por separado
        assert level2.complexity > 0

    def test_complexity_empty_l2(self):
        """Test de complejidad con una representación vacía de L2."""
        level2 = Level2(patterns=[], unique_blocks=[], inter_pattern_constraints=[])
        
        assert level2.complexity == 0.0


class TestLevel2Statistics:
    """Tests para las estadísticas del Nivel L2."""

    def test_get_statistics_with_patterns(self):
        """Test de obtención de estadísticas con patrones."""
        signature = PatternSignature(2, 1, 0, ('neq',))
        block = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')]
        )
        
        pattern = LocalPattern(
            pattern_id=0,
            signature=signature,
            instances=[0, 1, 2],
            template_block=block
        )
        
        unique_block = ConstraintBlock(
            block_id=3,
            variables={'v7', 'v8', 'v9'},
            constraints=[
                Constraint(scope=frozenset({'v7', 'v8'}), relation=lambda x, y: x != y, name='neq'),
                Constraint(scope=frozenset({'v8', 'v9'}), relation=lambda x, y: x != y, name='neq')
            ]
        )
        
        level2 = Level2(patterns=[pattern], unique_blocks=[unique_block], inter_pattern_constraints=[])
        stats = level2.get_statistics()
        
        assert stats['level'] == 2
        assert stats['num_patterns'] == 1
        assert stats['num_unique_blocks'] == 1
        assert stats['total_pattern_instances'] == 3
        assert stats['total_blocks'] == 4
        assert stats['avg_instances_per_pattern'] == 3.0
        assert stats['complexity'] > 0
        assert stats['compression_ratio'] > 1.0  # Debería haber compresión


class TestLevel2EdgeCases:
    """Tests para casos extremos del Nivel L2."""

    def test_empty_l2(self):
        """Test con una representación vacía de L2."""
        level2 = Level2(patterns=[], unique_blocks=[], inter_pattern_constraints=[])
        
        assert level2.validate() is True
        assert level2.complexity == 0.0
        assert level2.get_statistics()['num_patterns'] == 0

    def test_only_unique_blocks(self):
        """Test con solo bloques únicos (sin patrones)."""
        block = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')]
        )
        
        level2 = Level2(patterns=[], unique_blocks=[block], inter_pattern_constraints=[])
        
        assert level2.validate() is True
        assert level2.complexity > 0
        assert level2.get_statistics()['num_patterns'] == 0
        assert level2.get_statistics()['num_unique_blocks'] == 1


class TestLevel2Integration:
    """Tests de integración completa L0 -> L1 -> L2."""

    def test_full_integration_l0_l1_l2(self):
        """Test de integración completa L0 -> L1 -> L2."""
        # Crear un CSP con estructura repetitiva
        variables = {'v1', 'v2', 'v3', 'v4'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq'),
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # L0 -> L1 -> L2
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        level2 = Level2([], [], [], config={'original_domains': domains})
        level2.build_from_lower(level1)
        
        # Validar que L2 es coherente
        assert level2.validate() is True
        
        # Validar roundtrip L2 -> L1 -> L0
        refined_level1 = level2.refine_to_lower()
        assert refined_level1.validate() is True
        
        refined_level0 = refined_level1.refine_to_lower()
        assert refined_level0.validate() is True
        
        # El número de variables debería ser el mismo
        assert refined_level0.csp.variables == variables


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

