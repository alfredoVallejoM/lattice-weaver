"""
Tests para el Nivel L1 (Bloques de Restricciones) del compilador multiescala.
"""

import pytest
from lattice_weaver.compiler_multiescala import Level0, Level1, ConstraintBlock
from lattice_weaver.core.csp_problem import CSP, Constraint, generate_nqueens


class TestConstraintBlock:
    """Tests para la clase ConstraintBlock."""

    def test_init(self):
        """Test de inicialización de un bloque de restricciones."""
        variables = {'v1', 'v2'}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')
        ]
        block = ConstraintBlock(block_id=0, variables=variables, constraints=constraints)
        
        assert block.block_id == 0
        assert block.variables == variables
        assert block.constraints == constraints
        assert block.interface_variables == set()


class TestLevel1Initialization:
    """Tests para la inicialización del Nivel L1."""

    def test_init_with_simple_blocks(self):
        """Test de inicialización con bloques simples."""
        # Crear dos bloques simples
        block1 = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2')]
        )
        block2 = ConstraintBlock(
            block_id=1,
            variables={'v3', 'v4'},
            constraints=[Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq_v3_v4')]
        )
        
        # Restricción inter-bloque
        inter_constraint = Constraint(scope=frozenset({'v2', 'v3'}), relation=lambda x, y: x != y, name='neq_v2_v3')
        
        level1 = Level1(blocks=[block1, block2], inter_block_constraints=[inter_constraint])
        
        assert level1.level == 1
        assert len(level1.blocks) == 2
        assert len(level1.inter_block_constraints) == 1
        assert level1.variable_to_block['v1'] == 0
        assert level1.variable_to_block['v3'] == 1


class TestLevel1BuildFromLower:
    """Tests para la construcción de L1 desde L0."""

    def test_build_from_l0_simple(self):
        """Test de construcción desde L0 con un CSP simple."""
        # Crear un CSP con dos componentes desconectadas
        variables = {'v1', 'v2', 'v3', 'v4'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2'),
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq_v3_v4'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        # Deberíamos tener 2 bloques
        assert len(level1.blocks) == 2
        # No deberían haber restricciones inter-bloque
        assert len(level1.inter_block_constraints) == 0

    def test_build_from_l0_connected(self):
        """Test de construcción desde L0 con un CSP conectado."""
        # Crear un CSP conectado
        variables = {'v1', 'v2', 'v3'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2'),
            Constraint(scope=frozenset({'v2', 'v3'}), relation=lambda x, y: x != y, name='neq_v2_v3'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        # Deberíamos tener al menos 1 bloque
        assert len(level1.blocks) >= 1
        # Todas las variables deberían estar en algún bloque
        all_vars_in_blocks = set()
        for block in level1.blocks:
            all_vars_in_blocks.update(block.variables)
        assert all_vars_in_blocks == variables

    def test_build_from_l0_nqueens(self):
        """Test de construcción desde L0 con N-Reinas."""
        csp = generate_nqueens(4)
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': csp.domains})
        level1.build_from_lower(level0)
        
        # Deberíamos tener al menos 1 bloque
        assert len(level1.blocks) >= 1
        # Todas las variables deberían estar en algún bloque
        all_vars_in_blocks = set()
        for block in level1.blocks:
            all_vars_in_blocks.update(block.variables)
        assert all_vars_in_blocks == csp.variables


class TestLevel1RefineToLower:
    """Tests para el refinamiento de L1 a L0."""

    def test_refine_to_l0_simple(self):
        """Test de refinamiento a L0 con bloques simples."""
        # Crear un CSP original
        variables = {'v1', 'v2', 'v3', 'v4'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2'),
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq_v3_v4'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # Construir L1 desde L0
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        # Refinar de vuelta a L0
        refined_level0 = level1.refine_to_lower()
        
        assert isinstance(refined_level0, Level0)
        assert refined_level0.csp.variables == variables
        assert len(refined_level0.csp.constraints) == len(constraints)

    def test_refine_to_l0_nqueens(self):
        """Test de refinamiento a L0 con N-Reinas."""
        csp = generate_nqueens(4)
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': csp.domains})
        level1.build_from_lower(level0)
        
        # Refinar de vuelta a L0
        refined_level0 = level1.refine_to_lower()
        
        assert isinstance(refined_level0, Level0)
        assert refined_level0.csp.variables == csp.variables
        # El número de restricciones debería ser el mismo
        assert len(refined_level0.csp.constraints) == len(csp.constraints)


class TestLevel1Validation:
    """Tests para la validación del Nivel L1."""

    def test_validate_valid_l1(self):
        """Test de validación con una representación válida de L1."""
        # Crear bloques válidos
        block1 = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2')]
        )
        block2 = ConstraintBlock(
            block_id=1,
            variables={'v3', 'v4'},
            constraints=[Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq_v3_v4')]
        )
        
        inter_constraint = Constraint(scope=frozenset({'v2', 'v3'}), relation=lambda x, y: x != y, name='neq_v2_v3')
        
        level1 = Level1(blocks=[block1, block2], inter_block_constraints=[inter_constraint])
        
        assert level1.validate() is True

    def test_validate_empty_block(self):
        """Test de validación con un bloque vacío."""
        block1 = ConstraintBlock(block_id=0, variables=set(), constraints=[])
        block2 = ConstraintBlock(
            block_id=1,
            variables={'v3', 'v4'},
            constraints=[Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq_v3_v4')]
        )
        
        level1 = Level1(blocks=[block1, block2], inter_block_constraints=[])
        
        assert level1.validate() is False

    def test_validate_overlapping_blocks(self):
        """Test de validación con bloques solapados."""
        block1 = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2')]
        )
        block2 = ConstraintBlock(
            block_id=1,
            variables={'v2', 'v3'},  # v2 está en ambos bloques
            constraints=[Constraint(scope=frozenset({'v2', 'v3'}), relation=lambda x, y: x != y, name='neq_v2_v3')]
        )
        
        level1 = Level1(blocks=[block1, block2], inter_block_constraints=[])
        
        assert level1.validate() is False


class TestLevel1Complexity:
    """Tests para el cálculo de complejidad del Nivel L1."""

    def test_complexity_simple_l1(self):
        """Test de complejidad con una representación simple de L1."""
        block1 = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2')]
        )
        block2 = ConstraintBlock(
            block_id=1,
            variables={'v3', 'v4'},
            constraints=[Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq_v3_v4')]
        )
        
        level1 = Level1(blocks=[block1, block2], inter_block_constraints=[])
        
        # La complejidad debería ser mayor que 0
        assert level1.complexity > 0

    def test_complexity_empty_l1(self):
        """Test de complejidad con una representación vacía de L1."""
        level1 = Level1(blocks=[], inter_block_constraints=[])
        
        assert level1.complexity == 0.0


class TestLevel1Renormalization:
    """Tests para la renormalización en el Nivel L1."""

    def test_renormalize_simple_l1(self):
        """Test de renormalización con una representación simple de L1."""
        # Crear un CSP original
        variables = {'v1', 'v2', 'v3', 'v4'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2'),
            Constraint(scope=frozenset({'v2', 'v3'}), relation=lambda x, y: x != y, name='neq_v2_v3'),
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq_v3_v4'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # Construir L1 desde L0
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        # Renormalizar
        original_complexity = level1.complexity
        renormalized_level1 = level1.renormalize(partitioner='simple', k=2)
        
        assert isinstance(renormalized_level1, Level1)
        # La renormalización debería reducir o mantener la complejidad
        assert renormalized_level1.complexity <= original_complexity


class TestLevel1Statistics:
    """Tests para las estadísticas del Nivel L1."""

    def test_get_statistics_simple_l1(self):
        """Test de obtención de estadísticas con una representación simple de L1."""
        block1 = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2'},
            constraints=[Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2')]
        )
        block2 = ConstraintBlock(
            block_id=1,
            variables={'v3', 'v4'},
            constraints=[Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq_v3_v4')]
        )
        
        inter_constraint = Constraint(scope=frozenset({'v2', 'v3'}), relation=lambda x, y: x != y, name='neq_v2_v3')
        
        level1 = Level1(blocks=[block1, block2], inter_block_constraints=[inter_constraint])
        stats = level1.get_statistics()
        
        assert stats['level'] == 1
        assert stats['num_blocks'] == 2
        assert stats['num_variables'] == 4
        assert stats['num_internal_constraints'] == 2
        assert stats['num_inter_block_constraints'] == 1
        assert stats['avg_block_size'] == 2.0
        assert stats['complexity'] > 0


class TestLevel1EdgeCases:
    """Tests para casos extremos del Nivel L1."""

    def test_empty_l1(self):
        """Test con una representación vacía de L1."""
        level1 = Level1(blocks=[], inter_block_constraints=[])
        
        assert level1.validate() is True
        assert level1.complexity == 0.0
        assert level1.get_statistics()['num_blocks'] == 0

    def test_single_block_l1(self):
        """Test con un solo bloque."""
        block = ConstraintBlock(
            block_id=0,
            variables={'v1', 'v2', 'v3'},
            constraints=[
                Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2'),
                Constraint(scope=frozenset({'v2', 'v3'}), relation=lambda x, y: x != y, name='neq_v2_v3'),
            ]
        )
        
        level1 = Level1(blocks=[block], inter_block_constraints=[])
        
        assert level1.validate() is True
        assert level1.complexity > 0
        assert level1.get_statistics()['num_blocks'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

