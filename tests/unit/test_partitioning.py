"""
# tests/unit/test_partitioning.py

Pruebas para las estrategias de particionamiento de variables.
"""

import unittest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.renormalization.partition import VariablePartitioner

def create_symmetric_csp():
    """Crea un CSP con simetrías obvias."""
    variables = {"v0", "v1", "v2", "v3"}
    domains = {var: {0, 1} for var in variables}
    constraints = [
        Constraint(scope=frozenset({"v0", "v1"}), relation=lambda a, b: a != b),
        Constraint(scope=frozenset({"v2", "v3"}), relation=lambda a, b: a != b),
    ]
    return CSP(variables, domains, constraints, name="Symmetric")

def create_asymmetric_csp():
    """Crea un CSP sin simetrías."""
    variables = {"v0", "v1", "v2"}
    domains = {
        "v0": {0, 1},
        "v1": {0, 1, 2},
        "v2": {0, 1},
    }
    constraints = [
        Constraint(scope=frozenset({"v0", "v1"}), relation=lambda a, b: a != b),
    ]
    return CSP(variables, domains, constraints, name="Asymmetric")

class TestSymmetryPartitioner(unittest.TestCase):
    """Pruebas para el particionador por simetría."""

    def test_detect_symmetry_groups(self):
        """Test: _detect_symmetry_groups encuentra un único grupo para el CSP simétrico."""
        partitioner = VariablePartitioner(strategy='symmetry')
        csp = create_symmetric_csp()
        groups = partitioner._detect_symmetry_groups(csp)
        
        # La heurística actual debería agrupar todas las variables en un solo grupo.
        self.assertEqual(len(groups), 1, f"Se esperaba 1 grupo, pero se obtuvieron {len(groups)}")
        expected_group = frozenset({'v0', 'v1', 'v2', 'v3'})
        self.assertCountEqual(groups[0], expected_group)

    def test_partition_with_symmetries(self):
        """Test: El particionador por simetría delega a simple cuando la heurística no es efectiva."""
        partitioner = VariablePartitioner(strategy='symmetry')
        csp = create_symmetric_csp()
        partition = partitioner.partition(csp, k=2)
        
        # Como la heurística de simetría agrupa todo, se delega a _partition_simple.
        expected_partition = [frozenset({"v0", "v1"}), frozenset({"v2", "v3"})]
        self.assertCountEqual([frozenset(p) for p in partition], expected_partition)

    def test_partition_no_symmetries(self):
        """Test: El particionador por simetría recurre a simple cuando no hay simetrías."""
        partitioner = VariablePartitioner(strategy='symmetry')
        csp = create_asymmetric_csp()
        partition = partitioner.partition(csp, k=2)
        
        # Se delega a _partition_simple.
        expected_partition = [frozenset({"v0", "v1"}), frozenset({"v2"})]
        self.assertCountEqual([frozenset(p) for p in partition], expected_partition)

if __name__ == "__main__":
    unittest.main()

