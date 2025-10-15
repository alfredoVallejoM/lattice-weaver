'''
Pruebas unitarias para el FibrationSearchSolver.
'''
import unittest
from typing import Dict, List, Any

from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Constraint, ConstraintLevel, Hardness
from lattice_weaver.fibration.fibration_search_solver import FibrationSearchSolver

class TestFibrationSearchSolver(unittest.TestCase):

    def test_solve_simple_problem(self):
        '''Test: Resolver un problema simple con una solución obvia.'''
        variables = ['x', 'y']
        domains = {'x': [1, 2, 3], 'y': [1, 2, 3]}
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint('x', 'y', lambda a: a['x'] == a['y'], hardness=Hardness.HARD)

        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve()

        self.assertIsNotNone(solution)
        self.assertEqual(solution['x'], solution['y'])

    def test_solve_no_solution(self):
        '''Test: Resolver un problema sin solución.'''
        variables = ['x', 'y']
        domains = {'x': [1], 'y': [2]}
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint('x', 'y', lambda a: a['x'] == a['y'], hardness=Hardness.HARD)

        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve()

        self.assertIsNone(solution)

    def test_solve_with_soft_constraints(self):
        '''Test: Resolver un problema con restricciones soft.'''
        variables = ['x', 'y']
        domains = {'x': [1, 2], 'y': [1, 2]}
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint('x', 'y', lambda a: a['x'] != a['y'], hardness=Hardness.HARD)
        hierarchy.add_unary_constraint('x', lambda a: a['x'] == 1, weight=1.0, hardness=Hardness.SOFT)

        solver = FibrationSearchSolver(variables, domains, hierarchy)
        solution = solver.solve()

        self.assertIsNotNone(solution)
        self.assertEqual(solution['x'], 1)
        self.assertEqual(solution['y'], 2)

if __name__ == '__main__':
    unittest.main()

