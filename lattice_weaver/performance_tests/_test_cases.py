from .test_suite_generator import generate_n_queens_problem, generate_random_csp

def get_test_cases():
    test_cases = []

    # --- Problemas Base ---
    test_cases.append(generate_n_queens_problem(n=6, include_soft_constraints=False))
    test_cases.append(generate_n_queens_problem(n=6, include_soft_constraints=True))

    # --- Problemas de Complejidad Media ---
    test_cases.append(generate_random_csp(
        num_vars=10, domain_size=5, num_hard_constraints=3, num_soft_constraints=5,
        soft_constraint_weights=[1, 5, 2, 10, 3], name="Weighted Soft CSP V10D5H3S5"))
    test_cases.append(generate_random_csp(
        num_vars=12, domain_size=6, num_hard_constraints=4, num_soft_constraints=6,
        use_graph_structure=True, name="Graph CSP V12D6H4S6"))
    test_cases.append(generate_random_csp(
        num_vars=10, domain_size=5, num_hard_constraints=3, num_soft_constraints=5,
        use_hierarchical_constraints=True, name="Hierarchical CSP V10D5H3S5"))

    # --- Problemas de Alta Complejidad (Combinados y a Gran Escala - Ajustados) ---
    test_cases.append(generate_random_csp(
        num_vars=15, domain_size=7, num_hard_constraints=5, num_soft_constraints=8,
        soft_constraint_weights=[1, 2, 1, 3, 1, 2, 1, 4],
        use_graph_structure=True, use_hierarchical_constraints=True, name="Combined CSP V15D7H5S8"))
    test_cases.append(generate_random_csp(
        num_vars=18, domain_size=8, num_hard_constraints=6, num_soft_constraints=10,
        soft_constraint_weights=[1, 1, 2, 1, 3, 1, 1, 2, 1, 4],
        use_graph_structure=True, use_hierarchical_constraints=True, name="Combined CSP V18D8H6S10"))

    return test_cases

if __name__ == "__main__":
    # Este bloque asegura que pytest no intente recolectar este archivo como un test
    # y solo se ejecute cuando el script es llamado directamente.
    cases = get_test_cases()
    print(f"Generados {len(cases)} casos de prueba:")
    for case in cases:
        print(f"  - {case['name']}")

