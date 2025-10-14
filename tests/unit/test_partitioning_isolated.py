"""
# tests/unit/test_partitioning_isolated.py

Pruebas aisladas para la lógica de particionamiento y detección de simetrías.
"""

import unittest
from collections import defaultdict
import networkx as nx
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

class TestIsolatedSymmetry(unittest.TestCase):
    """Pruebas aisladas para la detección de simetrías."""

    def test_signature_generation_and_grouping(self):
        """Verifica que la generación de firmas y la agrupación funcionan como se espera."""
        partitioner = VariablePartitioner(strategy='symmetry')
        csp = create_symmetric_csp()
        
        signatures = defaultdict(list)
        constraint_graph = partitioner._build_constraint_graph(csp)
        
        generated_signatures = {}
        for var in sorted(list(csp.variables)):
            domain_size = len(csp.domains[var])
            degree = constraint_graph.degree(var)
            neighbor_signatures = []
            for neighbor in sorted(list(constraint_graph.neighbors(var))):
                neighbor_sig = (len(csp.domains[neighbor]), constraint_graph.degree(neighbor))
                neighbor_signatures.append(neighbor_sig)
            constraints_signature = frozenset(neighbor_signatures)
            signature = (domain_size, degree, constraints_signature)
            generated_signatures[var] = signature
            signatures[signature].append(var)

        # Imprimir las firmas generadas para cada variable
        for var, sig in generated_signatures.items():
            print(f"Variable: {var}, Signature: {sig}")

        # Verificar que todas las firmas son idénticas
        first_signature = next(iter(generated_signatures.values()))
        for sig in generated_signatures.values():
            self.assertEqual(sig, first_signature)

        # Verificar que todas las variables se agrupan en un solo grupo
        groups = [set(group) for group in signatures.values()]
        print(f"Generated groups: {groups}")
        self.assertEqual(len(groups), 1)
        self.assertCountEqual(groups[0], {"v0", "v1", "v2", "v3"})

if __name__ == "__main__":
    unittest.main()

