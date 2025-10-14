import unittest
from lattice_weaver.fibration.constraint_hierarchy import Constraint, ConstraintHierarchy, ConstraintLevel, Hardness

class TestConstraintHierarchy(unittest.TestCase):

    def setUp(self):
        self.hierarchy = ConstraintHierarchy()

    def test_add_level(self):
        self.hierarchy.add_level("CUSTOM_LEVEL")
        self.assertIn("CUSTOM_LEVEL", self.hierarchy.constraints)
        self.assertEqual(self.hierarchy.constraints["CUSTOM_LEVEL"], [])

    def test_add_constraint_to_new_level(self):
        self.hierarchy.add_level("CUSTOM_LEVEL")
        constraint = Constraint(
            level="CUSTOM_LEVEL",
            variables=["x"],
            predicate=lambda assignment: assignment["x"] > 0
        )
        self.hierarchy.add_constraint(constraint)
        self.assertIn(constraint, self.hierarchy.constraints["CUSTOM_LEVEL"])

    def test_add_hard_constraint_to_new_level(self):
        self.hierarchy.add_hard_constraint(("x", lambda x: x > 0), level="CUSTOM_LEVEL")
        self.assertIn("CUSTOM_LEVEL", self.hierarchy.constraints)
        self.assertEqual(len(self.hierarchy.constraints["CUSTOM_LEVEL"]), 1)
        self.assertEqual(self.hierarchy.constraints["CUSTOM_LEVEL"][0].hardness, Hardness.HARD)

    def test_add_soft_constraint_to_new_level(self):
        self.hierarchy.add_soft_constraint(("x", lambda x: x > 0), weight=0.5, level="CUSTOM_LEVEL")
        self.assertIn("CUSTOM_LEVEL", self.hierarchy.constraints)
        self.assertEqual(len(self.hierarchy.constraints["CUSTOM_LEVEL"]), 1)
        self.assertEqual(self.hierarchy.constraints["CUSTOM_LEVEL"][0].hardness, Hardness.SOFT)
        self.assertEqual(self.hierarchy.constraints["CUSTOM_LEVEL"][0].weight, 0.5)

    def test_evaluate_solution_with_new_level(self):
        self.hierarchy.add_level("CUSTOM_LEVEL")
        self.hierarchy.add_hard_constraint(("x", lambda x: x["x"] > 0), level="CUSTOM_LEVEL")
        self.hierarchy.add_soft_constraint(("y", lambda y: y["y"] < 0), weight=0.5, level="CUSTOM_LEVEL")

        solution_valid = {"x": 1, "y": -1}
        all_hard_satisfied, total_energy = self.hierarchy.evaluate_solution(solution_valid)
        self.assertTrue(all_hard_satisfied)
        self.assertEqual(total_energy, 0.0)

        solution_invalid_hard = {"x": -1, "y": -1}
        all_hard_satisfied, total_energy = self.hierarchy.evaluate_solution(solution_invalid_hard)
        self.assertFalse(all_hard_satisfied)

        solution_invalid_soft = {"x": 1, "y": 1}
        all_hard_satisfied, total_energy = self.hierarchy.evaluate_solution(solution_invalid_soft)
        self.assertTrue(all_hard_satisfied)
        self.assertEqual(total_energy, 0.5)

    def test_get_constraints_by_level_with_new_level(self):
        self.hierarchy.add_level("CUSTOM_LEVEL")
        constraint_expr = (["x"], lambda x: x > 0)
        self.hierarchy.add_hard_constraint(constraint_expr, level="CUSTOM_LEVEL")
        constraints = self.hierarchy.get_constraints_by_level("CUSTOM_LEVEL")
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0], constraint_expr)

    def test_get_all_constraints_with_new_level(self):
        self.hierarchy.add_level("CUSTOM_LEVEL")
        self.hierarchy.add_hard_constraint(("x", lambda x: x > 0), level="CUSTOM_LEVEL")
        all_constraints = self.hierarchy.get_all_constraints()
        self.assertIn("CUSTOM_LEVEL", all_constraints)
        self.assertEqual(len(all_constraints["CUSTOM_LEVEL"]), 1)

if __name__ == '__main__':
    unittest.main()

