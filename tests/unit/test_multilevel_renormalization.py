'''
# tests/unit/test_multilevel_renormalization.py

import unittest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.renormalization.core import renormalize_multilevel
from lattice_weaver.renormalization.hierarchy import AbstractionHierarchy

def create_simple_csp():
    variables = {f"v{i}" for i in range(4)}
    domains = {var: {0, 1} for var in variables}
    constraints = [
        Constraint(scope=frozenset({"v0", "v1"}), relation=lambda a, b: a != b),
        Constraint(scope=frozenset({"v1", "v2"}), relation=lambda a, b: a != b),
        Constraint(scope=frozenset({"v2", "v3"}), relation=lambda a, b: a != b),
    ]
    return CSP(variables, domains, constraints, name="Simple4Var")

class TestMultilevelRenormalization(unittest.TestCase):

    def test_hierarchy_creation(self):
        original_csp = create_simple_csp()
        hierarchy = renormalize_multilevel(original_csp, target_level=2, k_function=lambda l: 2)

        self.assertIsInstance(hierarchy, AbstractionHierarchy)
        self.assertEqual(hierarchy.highest_level, 2)
        self.assertIn(0, hierarchy.levels)
        self.assertIn(1, hierarchy.levels)
        self.assertIn(2, hierarchy.levels)

        # Verificar Nivel 0
        level0 = hierarchy.get_level(0)
        self.assertEqual(level0.csp, original_csp)

        # Verificar Nivel 1
        level1 = hierarchy.get_level(1)
        self.assertEqual(level1.level, 1)
        self.assertEqual(len(level1.csp.variables), 2) # 4 variables -> 2 grupos
        self.assertTrue(all(v.startswith("L1_G") for v in level1.csp.variables))

        # Verificar Nivel 2
        level2 = hierarchy.get_level(2)
        self.assertEqual(level2.level, 2)
        self.assertEqual(len(level2.csp.variables), 1) # 2 variables -> 1 grupo
        self.assertTrue(all(v.startswith("L2_G") for v in level2.csp.variables))

if __name__ == "__main__":
    unittest.main()

'''
