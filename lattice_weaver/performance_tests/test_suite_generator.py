import random
from typing import List, Dict, Any, Callable

from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness, Constraint

def generate_n_queens_problem(n: int, include_soft_constraints: bool = False) -> Dict[str, Any]:
    """
    Genera un problema de N-Reinas con la opción de incluir restricciones SOFT.
    """
    variables = [f"Q{i}" for i in range(n)]
    domains = {var: list(range(n)) for var in variables}
    hierarchy = ConstraintHierarchy()

    # Restricciones HARD: No dos reinas pueden atacarse entre sí
    for i in range(n):
        for j in range(i + 1, n):
            # No en la misma fila (implícito por la definición de variables)
            # No en la misma columna (implícito por la definición de variables)
            
            # No en la misma diagonal (principal)
            def diagonal1_predicate(assignment, q1=f"Q{i}", q2=f"Q{j}"):
                return abs(assignment[q1] - assignment[q2]) != abs(int(q1[1:]) - int(q2[1:]))
            hierarchy.add_constraint(Constraint(
                variables=[f"Q{i}", f"Q{j}"],
                predicate=diagonal1_predicate,
                level=ConstraintLevel.LOCAL,
                hardness=Hardness.HARD,
                metadata={"name": f"diag1_Q{i}_Q{j}"}
            ))

            # No en la misma diagonal (secundaria)
            def diagonal2_predicate(assignment, q1=f"Q{i}", q2=f"Q{j}"):
                return abs(assignment[q1] - assignment[q2]) != abs(int(q1[1:]) - int(q2[1:]))
            hierarchy.add_constraint(Constraint(
                variables=[f"Q{i}", f"Q{j}"],
                predicate=diagonal2_predicate,
                level=ConstraintLevel.LOCAL,
                hardness=Hardness.HARD,
                metadata={"name": f"diag2_Q{i}_Q{j}"}
            ))

    if include_soft_constraints:
        # Restricción SOFT: Minimizar el número de reinas en las filas centrales
        # Por ejemplo, para N=8, las filas centrales son 3 y 4 (índices 2 y 3)
        center_rows = [k for k in range(n) if k >= n//2 - 1 and k <= n//2]
        
        def center_row_predicate(assignment, queen_var, row_value):
            # Penaliza si la reina está en una fila central
            return 0 if assignment[queen_var] not in center_rows else 1

        for i in range(n):
            hierarchy.add_constraint(Constraint(
                variables=[f"Q{i}"],
                predicate=lambda assign, q=f"Q{i}": center_row_predicate(assign, q, center_rows),
                level=ConstraintLevel.LOCAL,
                hardness=Hardness.SOFT,
                metadata={"name": f"soft_center_Q{i}"}
            ))

    return {
        "name": f"N-Queens N={n}" + ("_soft" if include_soft_constraints else ""),
        "variables": variables,
        "domains": domains,
        "hierarchy": hierarchy
    }

def generate_random_csp(
    num_vars: int, 
    domain_size: int, 
    num_hard_constraints: int, 
    num_soft_constraints: int, 
    soft_constraint_weights: List[int] = None, 
    name: str = None,
    use_graph_structure: bool = False,
    use_hierarchical_constraints: bool = False
) -> Dict[str, Any]:
    """
    Genera un problema CSP aleatorio con restricciones HARD y SOFT, y opcionalmente con estructura de grafo y jerarquías.
    """
    variables = [f"v{i}" for i in range(num_vars)]
    domains = {var: list(range(domain_size)) for var in variables}
    hierarchy = ConstraintHierarchy()

    # Generar restricciones HARD aleatorias (binarias)
    for _ in range(num_hard_constraints):
        v1, v2 = random.sample(variables, 2)
        op = random.choice(["!=", "==", ">", "<"])
        val = random.randint(0, domain_size - 1)

        def hard_predicate(assignment, var1=v1, var2=v2, operator=op, value=val):
            if operator == "!=":
                return assignment[var1] != assignment[var2]
            elif operator == "==":
                return assignment[var1] == assignment[var2]
            elif operator == ">":
                return assignment[var1] > value
            elif operator == "<":
                return assignment[var1] < value
            return False # Debería ser inalcanzable

        hierarchy.add_constraint(Constraint(
            variables=[v1, v2],
            predicate=hard_predicate,
            level=ConstraintLevel.LOCAL,
            hardness=Hardness.HARD,
            metadata={"name": f"hard_{v1}{op}{v2}_{val}"}
        ))

    # Generar restricciones SOFT aleatorias (binarias o unarias) con pesos
    if soft_constraint_weights is None:
        soft_constraint_weights = [1] * num_soft_constraints
    
    for i in range(num_soft_constraints):
        if use_hierarchical_constraints and i % 2 == 0 and num_vars >= 3:
            # Restricción jerárquica: se activa si otra variable tiene un valor específico
            v_trigger, v1, v2 = random.sample(variables, 3)
            trigger_val = random.randint(0, domain_size - 1)
            op = random.choice(["!=", "=="])

            def hierarchical_soft_predicate(assignment, trigger_var=v_trigger, trigger_value=trigger_val, var1=v1, var2=v2, operator=op):
                if assignment[trigger_var] == trigger_value:
                    if operator == "!=":
                        return 0 if assignment[var1] != assignment[var2] else soft_constraint_weights[i]
                    else:
                        return 0 if assignment[var1] == assignment[var2] else soft_constraint_weights[i]
                return 0 # No se activa la restricción

            hierarchy.add_constraint(Constraint(
                variables=[v_trigger, v1, v2],
                predicate=hierarchical_soft_predicate,
                level=ConstraintLevel.LOCAL,
                hardness=Hardness.SOFT,
                metadata={"name": f"h_soft_{v_trigger}_{v1}{op}{v2}"}
            ))
        elif use_graph_structure and num_vars >= 2:
            # Restricción basada en grafo: las variables adyacentes no pueden tener el mismo valor
            v1, v2 = random.sample(variables, 2)
            def graph_soft_predicate(assignment, var1=v1, var2=v2):
                return 0 if assignment[var1] != assignment[var2] else soft_constraint_weights[i]

            hierarchy.add_constraint(Constraint(
                variables=[v1, v2],
                predicate=graph_soft_predicate,
                level=ConstraintLevel.LOCAL,
                hardness=Hardness.SOFT,
                metadata={"name": f"g_soft_{v1}_{v2}"}
            ))
        elif random.random() < 0.5 and num_vars >= 2: # Restricción binaria SOFT
            v1, v2 = random.sample(variables, 2)
            op = random.choice(["!=", "==", ">", "<"])
            val = random.randint(0, domain_size - 1)

            def soft_predicate_binary(assignment, var1=v1, var2=v2, operator=op, value=val):
                if operator == "!=":
                    return 0 if assignment[var1] != assignment[var2] else soft_constraint_weights[i]
                elif operator == "==":
                    return 0 if assignment[var1] == assignment[var2] else soft_constraint_weights[i]
                elif operator == ">":
                    return 0 if assignment[var1] > value else soft_constraint_weights[i]
                elif operator == "<":
                    return 0 if assignment[var1] < value else soft_constraint_weights[i]
                return soft_constraint_weights[i] # Por defecto, penalizar si no se puede evaluar

            hierarchy.add_constraint(Constraint(
                variables=[v1, v2],
                predicate=soft_predicate_binary,
                level=ConstraintLevel.LOCAL,
                hardness=Hardness.SOFT,
                metadata={"name": f"soft_{v1}{op}{v2}_{val}"}
            ))
        else: # Restricción unaria SOFT
            v = random.choice(variables)
            op = random.choice(["!=", "==", ">", "<"])
            val = random.randint(0, domain_size - 1)

            def soft_predicate_unary(assignment, var=v, operator=op, value=val):
                if operator == "!=":
                    return 0 if assignment[var] != value else soft_constraint_weights[i]
                elif operator == "==":
                    return 0 if assignment[var] == value else soft_constraint_weights[i]
                elif operator == ">":
                    return 0 if assignment[var] > value else soft_constraint_weights[i]
                elif operator == "<":
                    return 0 if assignment[var] < value else soft_constraint_weights[i]
                return soft_constraint_weights[i]
            
            hierarchy.add_constraint(Constraint(
                variables=[v],
                predicate=soft_predicate_unary,
                level=ConstraintLevel.LOCAL,
                hardness=Hardness.SOFT,
                metadata={"name": f"soft_{v}{op}{val}"}
            ))

    if name is None:
        name = f"Random CSP V{num_vars}D{domain_size}H{num_hard_constraints}S{num_soft_constraints}"
    return {
        "name": name,
        "variables": variables,
        "domains": domains,
        "hierarchy": hierarchy
    }


if __name__ == "__main__":
    # Ejemplo de uso:
    # Problema de N-Reinas
    n_queens_problem = generate_n_queens_problem(4)
    print(f'Generado: {n_queens_problem["name"]}')
    print(f'  Variables: {n_queens_problem["variables"]}')
    print(f'  Dominios: {n_queens_problem["domains"]}')
    print(f'  Número de restricciones HARD: {len(n_queens_problem["hierarchy"].get_constraints_at_level(ConstraintLevel.LOCAL))}')

    n_queens_soft_problem = generate_n_queens_problem(4, include_soft_constraints=True)
    print(f'\nGenerado: {n_queens_soft_problem["name"]}')
    print(f'  Variables: {n_queens_soft_problem["variables"]}')
    print(f'  Dominios: {n_queens_soft_problem["domains"]}')
    print(f'  Número de restricciones HARD: {len([c for c in n_queens_soft_problem["hierarchy"].get_constraints_at_level(ConstraintLevel.LOCAL) if c.hardness == Hardness.HARD])}')
    print(f'  Número de restricciones SOFT: {len([c for c in n_queens_soft_problem["hierarchy"].get_constraints_at_level(ConstraintLevel.LOCAL) if c.hardness == Hardness.SOFT])}')

    # Problema CSP aleatorio
    random_csp_problem = generate_random_csp(num_vars=5, domain_size=3, num_hard_constraints=3, num_soft_constraints=2)
    print(f'\nGenerado: {random_csp_problem["name"]}')
    print(f'  Variables: {random_csp_problem["variables"]}')
    print(f'  Dominios: {random_csp_problem["domains"]}')
    print(f'  Número de restricciones HARD: {len([c for c in random_csp_problem["hierarchy"].get_constraints_at_level(ConstraintLevel.LOCAL) if c.hardness == Hardness.HARD])}')
    print(f'  Número de restricciones SOFT: {len([c for c in random_csp_problem["hierarchy"].get_constraints_at_level(ConstraintLevel.LOCAL) if c.hardness == Hardness.SOFT])}')

    # Problema CSP con estructura de grafo
    graph_csp_problem = generate_random_csp(num_vars=10, domain_size=4, num_hard_constraints=5, num_soft_constraints=5, use_graph_structure=True, name="Graph CSP V10D4H5S5")
    print(f'\nGenerado: {graph_csp_problem["name"]}')
    print(f'  Número de restricciones SOFT: {len([c for c in graph_csp_problem["hierarchy"].get_constraints_at_level(ConstraintLevel.LOCAL) if c.hardness == Hardness.SOFT])}')

    # Problema CSP con restricciones jerárquicas
    hierarchical_csp_problem = generate_random_csp(num_vars=12, domain_size=5, num_hard_constraints=6, num_soft_constraints=6, use_hierarchical_constraints=True, name="Hierarchical CSP V12D5H6S6")
    print(f'\nGenerado: {hierarchical_csp_problem["name"]}')
    print(f'  Número de restricciones SOFT: {len([c for c in hierarchical_csp_problem["hierarchy"].get_constraints_at_level(ConstraintLevel.LOCAL) if c.hardness == Hardness.SOFT])}')

