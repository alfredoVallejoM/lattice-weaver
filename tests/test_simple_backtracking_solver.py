import unittest
from collections import defaultdict
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.simple_backtracking_solver import solve_csp_backtracking, generate_solutions_backtracking

class TestSimpleBacktrackingSolver(unittest.TestCase):

    def test_trivial_csp_satisfiable(self):
        # CSP: A in {1}, B in {1}, A == B
        variables = {"A", "B"}
        domains = {"A": frozenset({1}), "B": frozenset({1})}
        constraints = [Constraint(frozenset({"A", "B"}), lambda a, b: a == b, name="A_eq_B")]
        csp = CSP(variables, domains, constraints)
        
        solution = solve_csp_backtracking(csp)
        self.assertIsNotNone(solution)
        self.assertEqual(solution, {"A": 1, "B": 1})

    def test_trivial_csp_unsatisfiable(self):
        # CSP: A in {1}, B in {2}, A == B
        variables = {"A", "B"}
        domains = {"A": frozenset({1}), "B": frozenset({2})}
        constraints = [Constraint(frozenset({"A", "B"}), lambda a, b: a == b, name="A_eq_B")]
        csp = CSP(variables, domains, constraints)
        
        solution = solve_csp_backtracking(csp)
        self.assertIsNone(solution)

    def test_nqueens_n2_unsatisfiable(self):
        # N-Queens N=2 should be unsatisfiable
        variables = {"Q0", "Q1"}
        domains = {"Q0": frozenset({0, 1}), "Q1": frozenset({0, 1})}
        constraints = [
            Constraint(frozenset({"Q0", "Q1"}), lambda q0, q1: q0 != q1, name="row_col_Q0_Q1"),
            Constraint(frozenset({"Q0", "Q1"}), lambda q0, q1: abs(q0 - q1) != abs(0 - 1), name="diag_Q0_Q1")
        ]
        csp = CSP(variables, domains, constraints)
        solution = solve_csp_backtracking(csp)
        self.assertIsNone(solution)

    def test_nqueens_n4_satisfiable(self):
        # N-Queens N=4 should be satisfiable
        # Using a simplified N-Queens CSP for direct testing
        variables = {f"Q{i}" for i in range(4)}
        domains = {f"Q{i}": frozenset(range(4)) for i in range(4)}
        constraints = []
        for i in range(4):
            for j in range(i + 1, 4):
                qi = f"Q{i}"
                qj = f"Q{j}"
                constraints.append(Constraint(frozenset({qi, qj}), lambda val_i, val_j: val_i != val_j, name=f"row_col_{qi}_{qj}"))
                constraints.append(Constraint(frozenset({qi, qj}), lambda val_i, val_j, diff=abs(i - j): abs(val_i - val_j) != diff, name=f"diag_{qi}_{qj}"))
        csp = CSP(variables, domains, constraints)
        solution = solve_csp_backtracking(csp)
        self.assertIsNotNone(solution)

    def test_generate_solutions_backtracking(self):
        # CSP: A in {0,1}, B in {0,1}, A != B
        variables = {"A", "B"}
        domains = {"A": frozenset({0, 1}), "B": frozenset({0, 1})}
        constraints = [Constraint(frozenset({"A", "B"}), lambda a, b: a != b, name="A_neq_B")]
        csp = CSP(variables, domains, constraints)
        
        solutions = generate_solutions_backtracking(csp, num_solutions=2)
        self.assertEqual(len(solutions), 2)
        self.assertIn({"A": 0, "B": 1}, solutions)
        self.assertIn({"A": 1, "B": 0}, solutions)

if __name__ == '__main__':
    unittest.main()
