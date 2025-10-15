"""
Tests de integración para el puente CSP-Cúbico.
"""

import pytest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.formal.csp_cubical_bridge_refactored import CSPToCubicalBridge
from lattice_weaver.formal.cubical_types import CubicalSubtype, CubicalSigmaType

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

