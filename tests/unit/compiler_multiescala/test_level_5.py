"""
Tests para el Nivel L5 (Meta-patrones) del compilador multiescala.
"""

import pytest
from lattice_weaver.compiler_multiescala import (
    Level0, Level1, Level2, Level3, Level4, Level5,
    ConstraintBlock, LocalPattern, PatternSignature,
    CompositeStructure, CompositeSignature, DomainConcept, DomainConceptSignature,
    MetaPattern, MetaPatternSignature
)
from lattice_weaver.core.csp_problem import CSP, Constraint


class TestMetaPatternSignature:
    """Tests para la clase MetaPatternSignature."""

    def test_init(self):
        """Test de inicialización de una firma de meta-patrón."""
        signature = MetaPatternSignature(
            pattern_type='pipeline',
            num_concepts=2,
            concept_types=('task_0', 'task_1'),
            properties=frozenset([('flow', 'sequential')])
        )
        
        assert signature.pattern_type == 'pipeline'
        assert signature.num_concepts == 2
        assert signature.concept_types == ('task_0', 'task_1')
        assert signature.properties == frozenset([('flow', 'sequential')])

    def test_equality(self):
        """Test de igualdad de firmas."""
        sig1 = MetaPatternSignature('pipeline', 2, ('t0', 't1'), frozenset([('f', 's')]))
        sig2 = MetaPatternSignature('pipeline', 2, ('t0', 't1'), frozenset([('f', 's')]))
        sig3 = MetaPatternSignature('pipeline', 3, ('t0', 't1'), frozenset([('f', 's')]))
        
        assert sig1 == sig2
        assert sig1 != sig3


class TestMetaPattern:
    """Tests para la clase MetaPattern."""

    def test_init(self):
        """Test de inicialización de un meta-patrón."""
        signature = MetaPatternSignature('pipeline', 2, ('t0', 't1'), frozenset())
        meta_pattern = MetaPattern(
            meta_pattern_id=0,
            signature=signature,
            concepts=[0, 1],
            internal_constraints=[],
            meta_pattern_properties={'flow': 'sequential'}
        )
        
        assert meta_pattern.meta_pattern_id == 0
        assert meta_pattern.signature == signature
        assert meta_pattern.concepts == [0, 1]
        assert meta_pattern.meta_pattern_properties == {'flow': 'sequential'}


class TestLevel5Initialization:
    """Tests para la inicialización del Nivel L5."""

    def test_init_with_meta_patterns(self):
        """Test de inicialización con meta-patrones."""
        signature = MetaPatternSignature('pipeline', 2, ('t0', 't1'), frozenset())
        meta_pattern = MetaPattern(
            meta_pattern_id=0,
            signature=signature,
            concepts=[0, 1],
            internal_constraints=[]
        )
        
        level5 = Level5(
            meta_patterns=[meta_pattern],
            isolated_concepts=[],
            inter_meta_pattern_constraints=[]
        )
        
        assert level5.level == 5
        assert len(level5.meta_patterns) == 1
        assert level5.concept_to_meta_pattern[0] == 0
        assert level5.concept_to_meta_pattern[1] == 0


class TestLevel5BuildFromLower:
    """Tests para la construcción de L5 desde L4."""

    def test_build_from_l4_simple(self):
        """Test de construcción desde L4 con conceptos simples."""
        # Crear un CSP con estructura repetitiva
        variables = {f'v{i}' for i in range(8)}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({f'v{i}' for i in range(2)}), relation=lambda *args: True, name='c0'),
            Constraint(scope=frozenset({f'v{i}' for i in range(2, 4)}), relation=lambda *args: True, name='c1'),
            Constraint(scope=frozenset({f'v{i}' for i in range(4, 6)}), relation=lambda *args: True, name='c2'),
            Constraint(scope=frozenset({f'v{i}' for i in range(6, 8)}), relation=lambda *args: True, name='c3'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # L0 -> L1 -> L2 -> L3 -> L4 -> L5
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        level2 = Level2([], [], [], config={'original_domains': domains})
        level2.build_from_lower(level1)
        
        level3 = Level3([], [], [], [], config=level2.config)
        level3.build_from_lower(level2)
        
        level4 = Level4([], [], [], config=level3.config)
        level4.build_from_lower(level3)
        
        level5 = Level5([], [], [], config=level4.config)
        level5.build_from_lower(level4)
        
        # Validar que L5 es coherente
        assert level5.validate() is True

    def test_build_from_l4_no_meta_patterns(self):
        """Test de construcción desde L4 sin meta-patrones (solo conceptos aislados)."""
        # Crear un L4 vacío
        level4 = Level4(
            concepts=[],
            isolated_structures=[],
            inter_concept_constraints=[]
        )
        
        level5 = Level5([], [], [], config=level4.config)
        level5.build_from_lower(level4)
        
        # Sin conceptos, deberíamos tener L5 vacío
        assert len(level5.meta_patterns) == 0
        assert len(level5.isolated_concepts) == 0


class TestLevel5RefineToLower:
    """Tests para el refinamiento de L5 a L4."""

    def test_refine_to_l4_simple(self):
        """Test de refinamiento a L4 con meta-patrones simples."""
        # Crear un CSP con estructura repetitiva
        variables = {f'v{i}' for i in range(8)}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({f'v{i}' for i in range(2)}), relation=lambda *args: True, name='c0'),
            Constraint(scope=frozenset({f'v{i}' for i in range(2, 4)}), relation=lambda *args: True, name='c1'),
            Constraint(scope=frozenset({f'v{i}' for i in range(4, 6)}), relation=lambda *args: True, name='c2'),
            Constraint(scope=frozenset({f'v{i}' for i in range(6, 8)}), relation=lambda *args: True, name='c3'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # L0 -> L1 -> L2 -> L3 -> L4 -> L5
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        level2 = Level2([], [], [], config={'original_domains': domains})
        level2.build_from_lower(level1)
        
        level3 = Level3([], [], [], [], config=level2.config)
        level3.build_from_lower(level2)
        
        level4 = Level4([], [], [], config=level3.config)
        level4.build_from_lower(level3)
        
        level5 = Level5([], [], [], config=level4.config)
        level5.build_from_lower(level4)
        
        # Refinar de vuelta a L4
        refined_level4 = level5.refine_to_lower()
        
        assert isinstance(refined_level4, Level4)
        assert refined_level4.validate() is True


class TestLevel5Validation:
    """Tests para la validación del Nivel L5."""

    def test_validate_valid_l5(self):
        """Test de validación con una representación válida de L5."""
        signature = MetaPatternSignature('pipeline', 2, ('t0', 't1'), frozenset())
        meta_pattern = MetaPattern(
            meta_pattern_id=0,
            signature=signature,
            concepts=[0, 1],
            internal_constraints=[]
        )
        
        level5 = Level5(
            meta_patterns=[meta_pattern],
            isolated_concepts=[],
            inter_meta_pattern_constraints=[]
        )
        
        assert level5.validate() is True

    def test_validate_meta_pattern_with_no_concepts(self):
        """Test de validación con un meta-patrón sin conceptos."""
        signature = MetaPatternSignature('pipeline', 0, (), frozenset())
        meta_pattern = MetaPattern(
            meta_pattern_id=0,
            signature=signature,
            concepts=[],
            internal_constraints=[]
        )
        
        level5 = Level5(
            meta_patterns=[meta_pattern],
            isolated_concepts=[],
            inter_meta_pattern_constraints=[]
        )
        
        assert level5.validate() is False


class TestLevel5Complexity:
    """Tests para el cálculo de complejidad del Nivel L5."""

    def test_complexity_with_meta_patterns(self):
        """Test de complejidad con meta-patrones."""
        signature = MetaPatternSignature('pipeline', 2, ('t0', 't1'), frozenset())
        meta_pattern = MetaPattern(
            meta_pattern_id=0,
            signature=signature,
            concepts=[0, 1],
            internal_constraints=[]
        )
        
        level5 = Level5(
            meta_patterns=[meta_pattern],
            isolated_concepts=[],
            inter_meta_pattern_constraints=[]
        )
        
        assert level5.complexity > 0

    def test_complexity_empty_l5(self):
        """Test de complejidad con una representación vacía de L5."""
        level5 = Level5(
            meta_patterns=[],
            isolated_concepts=[],
            inter_meta_pattern_constraints=[]
        )
        
        assert level5.complexity == 0.0


class TestLevel5Statistics:
    """Tests para las estadísticas del Nivel L5."""

    def test_get_statistics_with_meta_patterns(self):
        """Test de obtención de estadísticas con meta-patrones."""
        signature = MetaPatternSignature('pipeline', 2, ('t0', 't1'), frozenset())
        meta_pattern = MetaPattern(
            meta_pattern_id=0,
            signature=signature,
            concepts=[0, 1],
            internal_constraints=[]
        )
        
        level5 = Level5(
            meta_patterns=[meta_pattern],
            isolated_concepts=[],
            inter_meta_pattern_constraints=[]
        )
        stats = level5.get_statistics()
        
        assert stats['level'] == 5
        assert stats['num_meta_patterns'] == 1
        assert stats['total_concepts_in_meta_patterns'] == 2
        assert stats['complexity'] > 0


class TestLevel5EdgeCases:
    """Tests para casos extremos del Nivel L5."""

    def test_empty_l5(self):
        """Test con una representación vacía de L5."""
        level5 = Level5(
            meta_patterns=[],
            isolated_concepts=[],
            inter_meta_pattern_constraints=[]
        )
        
        assert level5.validate() is True
        assert level5.complexity == 0.0
        assert level5.get_statistics()['num_meta_patterns'] == 0


class TestLevel5Integration:
    """Tests de integración completa L0 -> L1 -> L2 -> L3 -> L4 -> L5."""

    def test_full_integration_l0_l1_l2_l3_l4_l5(self):
        """Test de integración completa L0 -> L1 -> L2 -> L3 -> L4 -> L5."""
        # Crear un CSP con estructura repetitiva
        variables = {f'v{i}' for i in range(8)}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({f'v{i}' for i in range(2)}), relation=lambda *args: True, name='c0'),
            Constraint(scope=frozenset({f'v{i}' for i in range(2, 4)}), relation=lambda *args: True, name='c1'),
            Constraint(scope=frozenset({f'v{i}' for i in range(4, 6)}), relation=lambda *args: True, name='c2'),
            Constraint(scope=frozenset({f'v{i}' for i in range(6, 8)}), relation=lambda *args: True, name='c3'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # L0 -> L1 -> L2 -> L3 -> L4 -> L5
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        level2 = Level2([], [], [], config={'original_domains': domains})
        level2.build_from_lower(level1)
        
        level3 = Level3([], [], [], [], config=level2.config)
        level3.build_from_lower(level2)
        
        level4 = Level4([], [], [], config=level3.config)
        level4.build_from_lower(level3)
        
        level5 = Level5([], [], [], config=level4.config)
        level5.build_from_lower(level4)
        
        # Validar que L5 es coherente
        assert level5.validate() is True
        
        # Validar roundtrip L5 -> L4 -> L3 -> L2 -> L1 -> L0
        refined_level4 = level5.refine_to_lower()
        assert refined_level4.validate() is True
        
        refined_level3 = refined_level4.refine_to_lower()
        assert refined_level3.validate() is True
        
        refined_level2 = refined_level3.refine_to_lower()
        assert refined_level2.validate() is True
        
        refined_level1 = refined_level2.refine_to_lower()
        assert refined_level1.validate() is True
        
        refined_level0 = refined_level1.refine_to_lower()
        assert refined_level0.validate() is True
        
        # El número de variables debería ser el mismo
        assert refined_level0.csp.variables == variables

if __name__ == '__main__':    pytest.main([__file__, '-v'])