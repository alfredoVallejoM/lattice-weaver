"""
Tests para el Nivel L4 (Abstracciones de Dominio) del compilador multiescala.
"""

import pytest
from lattice_weaver.compiler_multiescala import (
    Level0, Level1, Level2, Level3, Level4,
    ConstraintBlock, LocalPattern, PatternSignature,
    CompositeStructure, CompositeSignature, DomainConcept, DomainConceptSignature
)
from lattice_weaver.core.csp_problem import CSP, Constraint


class TestDomainConceptSignature:
    """Tests para la clase DomainConceptSignature."""

    def test_init(self):
        """Test de inicialización de una firma de concepto de dominio."""
        signature = DomainConceptSignature(
            concept_type='scheduling_task',
            num_structures=2,
            structure_types=('structure_0', 'structure_1'),
            properties=frozenset([('priority', 'high')])
        )
        
        assert signature.concept_type == 'scheduling_task'
        assert signature.num_structures == 2
        assert signature.structure_types == ('structure_0', 'structure_1')
        assert signature.properties == frozenset([('priority', 'high')])

    def test_equality(self):
        """Test de igualdad de firmas."""
        sig1 = DomainConceptSignature('task', 2, ('s0', 's1'), frozenset([('p', 1)]))
        sig2 = DomainConceptSignature('task', 2, ('s0', 's1'), frozenset([('p', 1)]))
        sig3 = DomainConceptSignature('task', 3, ('s0', 's1'), frozenset([('p', 1)]))
        
        assert sig1 == sig2
        assert sig1 != sig3


class TestDomainConcept:
    """Tests para la clase DomainConcept."""

    def test_init(self):
        """Test de inicialización de un concepto de dominio."""
        signature = DomainConceptSignature('task', 2, ('s0', 's1'), frozenset([('p', 1)]))
        concept = DomainConcept(
            concept_id=0,
            signature=signature,
            structures=[0, 1],
            internal_constraints=[],
            domain_properties={'priority': 'high'}
        )
        
        assert concept.concept_id == 0
        assert concept.signature == signature
        assert concept.structures == [0, 1]
        assert concept.domain_properties == {'priority': 'high'}


class TestLevel4Initialization:
    """Tests para la inicialización del Nivel L4."""

    def test_init_with_concepts(self):
        """Test de inicialización con conceptos."""
        signature = DomainConceptSignature('task', 2, ('s0', 's1'), frozenset())
        concept = DomainConcept(
            concept_id=0,
            signature=signature,
            structures=[0, 1],
            internal_constraints=[]
        )
        
        level4 = Level4(
            concepts=[concept],
            isolated_structures=[],
            inter_concept_constraints=[]
        )
        
        assert level4.level == 4
        assert len(level4.concepts) == 1
        assert level4.structure_to_concept[0] == 0
        assert level4.structure_to_concept[1] == 0


class TestLevel4BuildFromLower:
    """Tests para la construcción de L4 desde L3."""

    def test_build_from_l3_simple(self):
        """Test de construcción desde L3 con estructuras simples."""
        # Crear un CSP con estructura repetitiva
        variables = {'v1', 'v2', 'v3', 'v4'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq'),
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # L0 -> L1 -> L2 -> L3 -> L4
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        level2 = Level2([], [], [], config={'original_domains': domains})
        level2.build_from_lower(level1)
        
        level3 = Level3([], [], [], [], config=level2.config)
        level3.build_from_lower(level2)
        
        level4 = Level4([], [], [], config=level3.config)
        level4.build_from_lower(level3)
        
        # Validar que L4 es coherente
        assert level4.validate() is True

    def test_build_from_l3_no_concepts(self):
        """Test de construcción desde L3 sin conceptos (solo estructuras aisladas)."""
        # Crear un L3 vacío
        level3 = Level3(
            structures=[],
            isolated_patterns=[],
            isolated_blocks=[],
            inter_structure_constraints=[]
        )
        
        level4 = Level4([], [], [], config=level3.config)
        level4.build_from_lower(level3)
        
        # Sin estructuras, deberíamos tener L4 vacío
        assert len(level4.concepts) == 0
        assert len(level4.isolated_structures) == 0


class TestLevel4RefineToLower:
    """Tests para el refinamiento de L4 a L3."""

    def test_refine_to_l3_simple(self):
        """Test de refinamiento a L3 con conceptos simples."""
        # Crear un CSP con estructura repetitiva
        variables = {'v1', 'v2', 'v3', 'v4'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq'),
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # L0 -> L1 -> L2 -> L3 -> L4
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        level2 = Level2([], [], [], config={'original_domains': domains})
        level2.build_from_lower(level1)
        
        level3 = Level3([], [], [], [], config=level2.config)
        level3.build_from_lower(level2)
        
        level4 = Level4([], [], [], config=level3.config)
        level4.build_from_lower(level3)
        
        # Refinar de vuelta a L3
        refined_level3 = level4.refine_to_lower()
        
        assert isinstance(refined_level3, Level3)
        assert refined_level3.validate() is True


class TestLevel4Validation:
    """Tests para la validación del Nivel L4."""

    def test_validate_valid_l4(self):
        """Test de validación con una representación válida de L4."""
        signature = DomainConceptSignature('task', 2, ('s0', 's1'), frozenset())
        concept = DomainConcept(
            concept_id=0,
            signature=signature,
            structures=[0, 1],
            internal_constraints=[]
        )
        
        level4 = Level4(
            concepts=[concept],
            isolated_structures=[],
            inter_concept_constraints=[]
        )
        
        assert level4.validate() is True

    def test_validate_concept_with_no_structures(self):
        """Test de validación con un concepto sin estructuras."""
        signature = DomainConceptSignature('task', 0, (), frozenset())
        concept = DomainConcept(
            concept_id=0,
            signature=signature,
            structures=[],
            internal_constraints=[]
        )
        
        level4 = Level4(
            concepts=[concept],
            isolated_structures=[],
            inter_concept_constraints=[]
        )
        
        assert level4.validate() is False


class TestLevel4Complexity:
    """Tests para el cálculo de complejidad del Nivel L4."""

    def test_complexity_with_concepts(self):
        """Test de complejidad con conceptos."""
        signature = DomainConceptSignature('task', 2, ('s0', 's1'), frozenset())
        concept = DomainConcept(
            concept_id=0,
            signature=signature,
            structures=[0, 1],
            internal_constraints=[]
        )
        
        level4 = Level4(
            concepts=[concept],
            isolated_structures=[],
            inter_concept_constraints=[]
        )
        
        assert level4.complexity > 0

    def test_complexity_empty_l4(self):
        """Test de complejidad con una representación vacía de L4."""
        level4 = Level4(
            concepts=[],
            isolated_structures=[],
            inter_concept_constraints=[]
        )
        
        assert level4.complexity == 0.0


class TestLevel4Statistics:
    """Tests para las estadísticas del Nivel L4."""

    def test_get_statistics_with_concepts(self):
        """Test de obtención de estadísticas con conceptos."""
        signature = DomainConceptSignature('task', 2, ('s0', 's1'), frozenset())
        concept = DomainConcept(
            concept_id=0,
            signature=signature,
            structures=[0, 1],
            internal_constraints=[]
        )
        
        level4 = Level4(
            concepts=[concept],
            isolated_structures=[],
            inter_concept_constraints=[]
        )
        stats = level4.get_statistics()
        
        assert stats['level'] == 4
        assert stats['num_concepts'] == 1
        assert stats['total_structures_in_concepts'] == 2
        assert stats['complexity'] > 0


class TestLevel4EdgeCases:
    """Tests para casos extremos del Nivel L4."""

    def test_empty_l4(self):
        """Test con una representación vacía de L4."""
        level4 = Level4(
            concepts=[],
            isolated_structures=[],
            inter_concept_constraints=[]
        )
        
        assert level4.validate() is True
        assert level4.complexity == 0.0
        assert level4.get_statistics()['num_concepts'] == 0


class TestLevel4Integration:
    """Tests de integración completa L0 -> L1 -> L2 -> L3 -> L4."""

    def test_full_integration_l0_l1_l2_l3_l4(self):
        """Test de integración completa L0 -> L1 -> L2 -> L3 -> L4."""
        # Crear un CSP con estructura repetitiva
        variables = {'v1', 'v2', 'v3', 'v4'}
        domains = {var: frozenset([1, 2]) for var in variables}
        constraints = [
            Constraint(scope=frozenset({'v1', 'v2'}), relation=lambda x, y: x != y, name='neq'),
            Constraint(scope=frozenset({'v3', 'v4'}), relation=lambda x, y: x != y, name='neq'),
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)
        
        # L0 -> L1 -> L2 -> L3 -> L4
        level0 = Level0(csp)
        level1 = Level1([], [], config={'original_domains': domains})
        level1.build_from_lower(level0)
        
        level2 = Level2([], [], [], config={'original_domains': domains})
        level2.build_from_lower(level1)
        
        level3 = Level3([], [], [], [], config=level2.config)
        level3.build_from_lower(level2)
        
        level4 = Level4([], [], [], config=level3.config)
        level4.build_from_lower(level3)
        
        # Validar que L4 es coherente
        assert level4.validate() is True
        
        # Validar roundtrip L4 -> L3 -> L2 -> L1 -> L0
        refined_level3 = level4.refine_to_lower()
        assert refined_level3.validate() is True
        
        refined_level2 = refined_level3.refine_to_lower()
        assert refined_level2.validate() is True
        
        refined_level1 = refined_level2.refine_to_lower()
        assert refined_level1.validate() is True
        
        refined_level0 = refined_level1.refine_to_lower()
        assert refined_level0.validate() is True
        
        # El número de variables debería ser el mismo
        assert refined_level0.csp.variables == variables


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
