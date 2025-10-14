import unittest
import time
from collections import deque
from typing import Any, Callable, Dict, Tuple, List

from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.arc_engine.domains import SetDomain
from lattice_weaver.arc_engine.constraints import Constraint, get_relation, register_relation
from lattice_weaver.arc_engine.ac31 import revise_with_last_support

# Register a sample relation for testing

def not_equal(a, b, metadata=None):
    return a != b

register_relation("not_equal", not_equal)

class TestAC31Optimizations(unittest.TestCase):

    def setUp(self):
        self.engine = ArcEngine()
        self.engine.add_variable("X", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.engine.add_variable("Y", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.engine.add_variable("Z", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.engine.add_constraint("X", "Y", "not_equal", cid="C1")
        self.engine.add_constraint("Y", "Z", "not_equal", cid="C2")
        self.engine.add_constraint("X", "Z", "not_equal", cid="C3")

    def test_revise_with_last_support_correctness(self):
        # Test a simple case where revision should happen
        engine = ArcEngine()
        engine.add_variable("A", [1, 2])
        engine.add_variable("B", [1])
        engine.add_constraint("A", "B", "not_equal", cid="C_AB")

        relation_func = get_relation("not_equal")
        revised, removed = revise_with_last_support(engine, "A", "B", "C_AB", relation_func, engine.constraints["C_AB"].metadata)

        self.assertTrue(revised)
        self.assertEqual(removed, [1])
        self.assertEqual(list(engine.variables["A"].get_values()), [2])
        self.assertIn(("A", "B", 2), engine.last_support)
        self.assertEqual(engine.last_support[("A", "B", 2)], 1)

    def test_enforce_arc_consistency_correctness(self):
        # Test a simple CSP problem
        engine = ArcEngine()
        engine.add_variable("X", [1, 2])
        engine.add_variable("Y", [1, 2])
        engine.add_constraint("X", "Y", "not_equal", cid="XY")

        result = engine.enforce_arc_consistency()
        self.assertTrue(result)
        self.assertEqual(list(engine.variables["X"].get_values()), [1, 2])
        self.assertEqual(list(engine.variables["Y"].get_values()), [1, 2])

        # Add a constraint that forces a reduction
        engine.add_variable("Z", [1])
        engine.add_constraint("Y", "Z", "not_equal", cid="YZ")
        result = engine.enforce_arc_consistency()
        self.assertTrue(result) # Should still be consistent, Y can be 2
        self.assertEqual(list(engine.variables["Y"].get_values()), [2])

    def test_performance_ac31(self):
        # Test performance on a larger problem
        num_vars = 10
        domain_size = 10
        engine = ArcEngine()
        variables = [f"V{i}" for i in range(num_vars)]
        for var in variables:
            engine.add_variable(var, list(range(1, domain_size + 1)))

        # Add constraints (e.g., all_diff-like structure)
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                engine.add_constraint(variables[i], variables[j], "not_equal", cid=f"C_{i}_{j}")

        start_time = time.perf_counter()
        result = engine.enforce_arc_consistency()
        end_time = time.perf_counter()
        
        self.assertTrue(result)
        print(f"\nTime taken for AC-3.1 on {num_vars} vars, {domain_size} domain size: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    unittest.main()

