"""
Tests para el Nivel L0 (Primitivas CSP) del compilador multiescala.
"""

import pytest
from lattice_weaver.compiler_multiescala import Level0
from lattice_weaver.core.csp_problem import CSP, Constraint, generate_nqueens, generate_random_csp


class TestLevel0Initialization:
    """Tests para la inicialización del Nivel L0."""

    def test_init_with_simple_csp(self):
        """Test de inicialización con un CSP simple."""
        variables = {'v1', 'v2'}
        domains = {'v1': frozenset([1, 2]), 'v2': frozenset([1, 2])}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        level0 = Level0(csp)
        
        assert level0.level == 0
        assert level0.csp == csp
        assert level0.data == csp
        assert level0.constraint_graph is not None

    def test_init_with_nqueens(self):
        """Test de inicialización con el problema de N-Reinas."""
        csp = generate_nqueens(4)
        level0 = Level0(csp)
        
        assert level0.level == 0
        assert len(level0.csp.variables) == 4
        assert level0.constraint_graph is not None


class TestLevel0Validation:
    """Tests para la validación del Nivel L0."""

    def test_validate_valid_csp(self):
        """Test de validación con un CSP válido."""
        csp = generate_nqueens(4)
        level0 = Level0(csp)
        
        assert level0.validate() is True

    def test_validate_empty_domain(self):
        """Test de validación con un dominio vacío."""
        variables = {'v1', 'v2'}
        domains = {'v1': frozenset(), 'v2': frozenset([1, 2])}
        constraints = []
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        level0 = Level0(csp)
        
        assert level0.validate() is False

    def test_validate_missing_domain(self):
        """Test de validación con un dominio faltante."""
        variables = {'v1', 'v2'}
        domains = {'v1': frozenset([1, 2])}
        constraints = []
        
        # CSP.__post_init__ debería lanzar una excepción
        with pytest.raises(ValueError):
            csp = CSP(variables=variables, domains=domains, constraints=constraints)


class TestLevel0Complexity:
    """Tests para el cálculo de complejidad del Nivel L0."""

    def test_complexity_simple_csp(self):
        """Test de complejidad con un CSP simple."""
        variables = {'v1', 'v2'}
        domains = {'v1': frozenset([1, 2]), 'v2': frozenset([1, 2])}
        constraints = []
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        level0 = Level0(csp)
        
        # Complejidad = log(3) + log(3) = 2 * log(3) ≈ 2.197
        import math
        expected_complexity = 2 * math.log(3)
        assert abs(level0.complexity - expected_complexity) < 0.01

    def test_complexity_nqueens(self):
        """Test de complejidad con N-Reinas."""
        csp = generate_nqueens(4)
        level0 = Level0(csp)
        
        # Complejidad = 4 * log(5) ≈ 6.44
        import math
        expected_complexity = 4 * math.log(5)
        assert abs(level0.complexity - expected_complexity) < 0.01

    def test_complexity_empty_csp(self):
        """Test de complejidad con un CSP vacío."""
        variables = set()
        domains = {}
        constraints = []
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        level0 = Level0(csp)
        
        assert level0.complexity == 0.0


class TestLevel0Renormalization:
    """Tests para la renormalización en el Nivel L0."""

    def test_renormalize_simple_csp(self):
        """Test de renormalización con un CSP simple."""
        variables = {'v1', 'v2', 'v3', 'v4'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2'),
            Constraint(scope=frozenset({'v2', 'v3'}), relation=lambda x, y: x != y, name='neq_v2_v3'),
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq_v3_v4'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        level0 = Level0(csp)
        renormalized_level0 = level0.renormalize(partitioner='simple', k=2)
        
        assert isinstance(renormalized_level0, Level0)
        assert renormalized_level0.level == 0
        # La renormalización debería reducir la complejidad
        assert renormalized_level0.complexity <= level0.complexity

    def test_renormalize_nqueens(self):
        """Test de renormalización con N-Reinas."""
        csp = generate_nqueens(4)
        level0 = Level0(csp)
        
        original_complexity = level0.complexity
        renormalized_level0 = level0.renormalize(partitioner='simple', k=2)
        
        assert isinstance(renormalized_level0, Level0)
        # La renormalización debería reducir o mantener la complejidad
        assert renormalized_level0.complexity <= original_complexity


class TestLevel0ConstraintBlocks:
    """Tests para la detección de bloques de restricciones."""

    def test_detect_blocks_simple_csp(self):
        """Test de detección de bloques con un CSP simple."""
        variables = {'v1', 'v2', 'v3', 'v4'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2'),
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq_v3_v4'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        level0 = Level0(csp)
        blocks = level0.detect_constraint_blocks()
        
        # Deberíamos tener 2 bloques: {v1, v2} y {v3, v4}
        assert len(blocks) == 2
        assert any({'v1', 'v2'} == block for block in blocks)
        assert any({'v3', 'v4'} == block for block in blocks)

    def test_detect_blocks_connected_csp(self):
        """Test de detección de bloques con un CSP completamente conectado."""
        variables = {'v1', 'v2', 'v3'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq_v1_v2'),
            Constraint(scope=frozenset({'v2', 'v3'}), relation=lambda x, y: x != y, name='neq_v2_v3'),
            Constraint(scope=frozenset({'v1', 'v3'}), relation=lambda x, y: x != y, name='neq_v1_v3'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        level0 = Level0(csp)
        blocks = level0.detect_constraint_blocks()
        
        # Deberíamos tener 1 bloque con todas las variables
        assert len(blocks) >= 1
        # Todas las variables deberían estar en algún bloque
        all_vars_in_blocks = set().union(*blocks)
        assert all_vars_in_blocks == variables


class TestLevel0Statistics:
    """Tests para las estadísticas del Nivel L0."""

    def test_get_statistics_simple_csp(self):
        """Test de obtención de estadísticas con un CSP simple."""
        variables = {'v1', 'v2'}
        domains = {'v1': frozenset([1, 2]), 'v2': frozenset([1, 2])}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq')
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        level0 = Level0(csp)
        stats = level0.get_statistics()
        
        assert stats['level'] == 0
        assert stats['num_variables'] == 2
        assert stats['num_constraints'] == 1
        assert stats['avg_domain_size'] == 2.0
        assert stats['complexity'] > 0

    def test_get_statistics_nqueens(self):
        """Test de obtención de estadísticas con N-Reinas."""
        csp = generate_nqueens(4)
        level0 = Level0(csp)
        
        stats = level0.get_statistics()
        
        assert stats['level'] == 0
        assert stats['num_variables'] == 4
        assert stats['num_constraints'] > 0
        assert stats['avg_domain_size'] == 4.0
        assert stats['complexity'] > 0


class TestLevel0EdgeCases:
    """Tests para casos extremos del Nivel L0."""

    def test_empty_csp(self):
        """Test con un CSP vacío."""
        variables = set()
        domains = {}
        constraints = []
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        level0 = Level0(csp)
        
        assert level0.validate() is True
        assert level0.complexity == 0.0
        assert level0.get_statistics()['num_variables'] == 0

    def test_single_variable_csp(self):
        """Test con un CSP de una sola variable."""
        variables = {'v1'}
        domains = {'v1': frozenset([1, 2, 3])}
        constraints = []
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        level0 = Level0(csp)
        
        assert level0.validate() is True
        assert level0.complexity > 0
        assert level0.get_statistics()['num_variables'] == 1

    def test_build_from_lower_raises_error(self):
        """Test que build_from_lower lanza una excepción."""
        csp = generate_nqueens(4)
        level0 = Level0(csp)
        
        with pytest.raises(NotImplementedError):
            level0.build_from_lower(None)

    def test_refine_to_lower_raises_error(self):
        """Test que refine_to_lower lanza una excepción."""
        csp = generate_nqueens(4)
        level0 = Level0(csp)
        
        with pytest.raises(NotImplementedError):
            level0.refine_to_lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

