import pytest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.formal.cubical_csp_type import CubicalCSPType
from lattice_weaver.formal.csp_cubical_bridge import CSPToCubicalBridge
from lattice_weaver.formal.path_finder import PathFinder
from lattice_weaver.formal.symmetry_extractor import SymmetryExtractor


class TestEndToEndIntegration:
    def test_full_workflow_simple_csp(self):
        csp = CSP(
            variables={'X', 'Y', 'Z'},
            domains={'X': frozenset({1, 2, 3}), 'Y': frozenset({1, 2, 3}), 'Z': frozenset({1, 2, 3})},
            constraints=[
                Constraint(scope=frozenset({'X', 'Y'}), relation=lambda x, y: x != y, name='neq_xy'),
                Constraint(scope=frozenset({'Y', 'Z'}), relation=lambda y, z: y != z, name='neq_yz')
            ]
        )
        bridge = CSPToCubicalBridge(csp)

        cubical_type = bridge.cubical_type
        assert cubical_type is not None
        assert len(cubical_type.variables) == 3
        assert len(cubical_type.constraint_props) == 2

        valid_solution = {'X': 1, 'Y': 2, 'Z': 1}
        invalid_solution = {'X': 1, 'Y': 1, 'Z': 2}

        assert bridge.verify_solution(valid_solution) is True
        assert bridge.verify_solution(invalid_solution) is False

        finder = PathFinder(bridge)
        solution1 = {'X': 1, 'Y': 2, 'Z': 1}
        solution2 = {'X': 2, 'Y': 3, 'Z': 1}

        path = finder.find_path(solution1, solution2)
        assert path is not None

        extractor = SymmetryExtractor(bridge)
        analysis = extractor.analyze_symmetry_structure()

        assert 'symmetry_count' in analysis
        assert 'has_symmetries' in analysis

    def test_nqueens_integration(self):
        csp = CSP(
            variables={'Q1', 'Q2', 'Q3'},
            domains={'Q1': frozenset({1, 2, 3}), 'Q2': frozenset({1, 2, 3}), 'Q3': frozenset({1, 2, 3})},
            constraints=[
                Constraint(scope=frozenset({'Q1', 'Q2'}), relation=lambda x, y: x != y, name='neq_q1q2'),
                Constraint(scope=frozenset({'Q2', 'Q3'}), relation=lambda x, y: x != y, name='neq_q2q3'),
                Constraint(scope=frozenset({'Q1', 'Q3'}), relation=lambda x, y: x != y, name='neq_q1q3')
            ]
        )
        bridge = CSPToCubicalBridge(csp)

        assert bridge.cubical_type is not None

        solution = {'Q1': 1, 'Q2': 2, 'Q3': 3}
        assert bridge.verify_solution(solution) is True

        extractor = SymmetryExtractor(bridge)
        group = extractor.extract_all_symmetries()

        assert group.order >= 0


class TestBridgeWithPathFinder:
    def test_path_finding_with_bridge(self):
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2, 3}), 'Y': frozenset({1, 2, 3})},
            constraints=[Constraint(scope=frozenset({'X', 'Y'}), relation=lambda x, y: x < y, name='lt_xy')]
        )
        bridge = CSPToCubicalBridge(csp)

        finder = PathFinder(bridge)

        sol1 = {'X': 1, 'Y': 2}
        sol2 = {'X': 2, 'Y': 3}

        path = finder.find_path(sol1, sol2)

        assert path is not None
        assert path.start == sol1
        assert path.end == sol2

    def test_equivalence_checking(self):
        csp = CSP(
            variables={'X', 'Y', 'Z'},
            domains={'X': frozenset({1, 2}), 'Y': frozenset({1, 2}), 'Z': frozenset({1, 2})},
            constraints=[
                Constraint(scope=frozenset({'X', 'Y'}), relation=lambda x, y: x != y, name='neq_xy'),
                Constraint(scope=frozenset({'Y', 'Z'}), relation=lambda y, z: y != z, name='neq_yz')
            ]
        )
        bridge = CSPToCubicalBridge(csp)

        finder = PathFinder(bridge)

        sol1 = {'X': 1, 'Y': 2, 'Z': 1}
        sol2 = {'X': 2, 'Y': 1, 'Z': 2}

        equiv = finder.are_equivalent(sol1, sol2)

        assert isinstance(equiv, bool)


class TestBridgeWithSymmetryExtractor:
    def test_symmetry_extraction_with_bridge(self):
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2}), 'Y': frozenset({1, 2})},
            constraints=[]
        )
        bridge = CSPToCubicalBridge(csp)

        extractor = SymmetryExtractor(bridge)

        symmetries = extractor.extract_variable_symmetries()

        assert isinstance(symmetries, list)

    def test_equivalence_classes_with_solutions(self):
        csp = CSP(
            variables={'X', 'Y'},
            domains={'X': frozenset({1, 2}), 'Y': frozenset({1, 2})},
            constraints=[Constraint(scope=frozenset({'X', 'Y'}), relation=lambda x, y: x != y, name='neq_xy')]
        )
        bridge = CSPToCubicalBridge(csp)

        extractor = SymmetryExtractor(bridge)

        solutions = [
            {'X': 1, 'Y': 2},
            {'X': 2, 'Y': 1}
        ]

        for sol in solutions:
            assert bridge.verify_solution(sol) is True

        classes = extractor.get_equivalence_classes(solutions)

        assert len(classes) >= 1
        assert len(classes) <= len(solutions)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

