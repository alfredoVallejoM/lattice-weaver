"""
Tests de integración para el puente CSP-Cúbico.
"""

import pytest
from lattice_weaver.core.csp_problem import CSP, Constraint, AllDifferentConstraint
from lattice_weaver.formal.csp_cubical_bridge_refactored import CSPToCubicalBridge
from lattice_weaver.formal.cubical_types import CubicalSubtype, CubicalSigmaType, CubicalNegation, CubicalPredicate, VariableTerm, ValueTerm, CubicalAnd, CubicalPath

class TestCSPToCubicalBridge:
    def test_translate_simple_csp(self):
        # CSP simple: 2 variables, dominios {1,2,3}, sin restricciones
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2, 3}), 'Y': frozenset({1, 2, 3})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge()
        cubical_type = bridge.to_cubical(csp)
        
        assert isinstance(cubical_type, CubicalSubtype)
        assert isinstance(cubical_type.base_type, CubicalSigmaType)
        assert len(cubical_type.base_type.components) == 2
        # El predicado es un placeholder, así que esperamos el valor por defecto
        expected_predicate = CubicalPath(ValueTerm(True), ValueTerm(True))
        assert str(cubical_type.predicate) == str(expected_predicate)

    def test_translate_search_space(self):
        csp = CSP(
            variables={'A', 'B', 'C'},
            domains={'A': frozenset({1, 2}), 'B': frozenset({3, 4, 5}), 'C': frozenset({6})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge()
        search_space = bridge._translate_search_space(csp)
        
        assert isinstance(search_space, CubicalSigmaType)
        assert len(search_space.components) == 3
        # Verificar que los tamaños de los dominios son correctos
        sizes = {name: comp.size for name, comp in search_space.components}
        assert sizes['A'] == 2
        assert sizes['B'] == 3
        assert sizes['C'] == 1

    def test_translate_alldifferent_constraint(self):
        csp = CSP(variables=set(), domains=dict(), constraints=[])
        csp.add_variable("A", [1, 2])
        csp.add_variable("B", [1, 2])
        csp.add_variable("C", [1, 2])
        csp.add_constraint(AllDifferentConstraint(["A", "B", "C"]))

        bridge = CSPToCubicalBridge()
        cubical_subtype = bridge.to_cubical(csp)

        # Esperamos un CubicalAnd de negaciones de igualdad para todos los pares de variables
        assert isinstance(cubical_subtype.predicate, CubicalAnd)
        
        expected_predicates = frozenset([
            CubicalNegation(CubicalPath(VariableTerm("A"), VariableTerm("B"))),
            CubicalNegation(CubicalPath(VariableTerm("A"), VariableTerm("C"))),
            CubicalNegation(CubicalPath(VariableTerm("B"), VariableTerm("C"))),
        ])


        generated_predicates = cubical_subtype.predicate.predicates

        assert generated_predicates == expected_predicates

    def test_translate_sum_constraint(self):
        csp = CSP(variables=set(), domains=dict(), constraints=[])
        csp.add_variable("X", [1, 2, 3])
        csp.add_variable("Y", [1, 2, 3])
        csp.add_constraint(SumConstraint(scope=frozenset({"X", "Y"}), target_sum=3))

        bridge = CSPToCubicalBridge()
        cubical_subtype = bridge.to_cubical(csp)

        expected_sum_expression = CubicalArithmetic("sum", (VariableTerm("X"), VariableTerm("Y")))
        expected_target_value = ValueTerm(3)
        expected_predicate = CubicalComparison(expected_sum_expression, "==", expected_target_value)

        assert isinstance(cubical_subtype.predicate, CubicalComparison)
        assert cubical_subtype.predicate == expected_predicate
