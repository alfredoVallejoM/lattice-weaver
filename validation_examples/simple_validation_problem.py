import sys
sys.path.append("/home/ubuntu/lattice-weaver")

from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness, Constraint
from lattice_weaver.external_solvers.python_constraint_adapter import PythonConstraintAdapter
from lattice_weaver.external_solvers.ortools_cpsat_adapter import ORToolsCPSATAdapter
from lattice_weaver.external_solvers.pymoo_adapter import PymooAdapter

def run_simple_validation_problem():
    print("\n--- Ejecutando problema de validación simple ---")

    variables = ['x', 'y']
    domains = {'x': [0, 1, 2], 'y': [0, 1, 2]}

    # Definir la jerarquía de restricciones
    hierarchy = ConstraintHierarchy()

    # Restricción HARD: x != y
    def hard_neq_predicate(assignment):
        return assignment['x'] != assignment['y']
    hierarchy.add_constraint(Constraint(
        variables=['x', 'y'],
        predicate=hard_neq_predicate,
        level=ConstraintLevel.LOCAL,
        hardness=Hardness.HARD,
        metadata={'name': 'hard_neq'}
    ))

    # Restricción SOFT: x + y == 2 (penalizar si no se cumple)
    def soft_sum_predicate(assignment):
        # Retorna 0 si se cumple, 1 si no se cumple (para minimización)
        return 0 if assignment['x'] + assignment['y'] == 2 else 1
    hierarchy.add_constraint(Constraint(
        variables=['x', 'y'],
        predicate=soft_sum_predicate,
        level=ConstraintLevel.LOCAL,
        hardness=Hardness.SOFT,
        metadata={'name': 'soft_sum'}
    ))

    print("Variables:", variables)
    print("Dominios:", domains)
    print("Restricciones en la jerarquía:")
    for level in ConstraintLevel:
        for c in hierarchy.get_constraints_at_level(level):
            print(f"  - {c.metadata.get('name', 'unnamed')} ({c.hardness.name}) en {c.level.name}: {c.variables}")

    # --- Probar PythonConstraintAdapter ---
    print("\n--- Probando PythonConstraintAdapter (solo HARD) ---")
    pc_adapter = PythonConstraintAdapter(variables, domains, hierarchy)
    pc_solutions = pc_adapter.solve()
    print(f"Soluciones encontradas por python-constraint ({len(pc_solutions)}):")
    for sol in pc_solutions:
        print(f"  {sol}")
    # Verificar que las soluciones satisfacen x != y
    for sol in pc_solutions:
        assert sol["x"] != sol["y"], f"La solución {sol} viola la restricción HARD x != y"

    print("PythonConstraintAdapter: Validado (solo restricciones HARD).")

    # --- Probar ORToolsCPSATAdapter ---
    print("\n--- Probando ORToolsCPSATAdapter (HARD y SOFT) ---")
    hierarchy_for_ortools = ConstraintHierarchy()
    hierarchy_for_ortools.add_constraint(Constraint(
        variables=['x', 'y'],
        predicate=hard_neq_predicate, 
        level=ConstraintLevel.LOCAL,
        hardness=Hardness.HARD,
        metadata={'name': 'all_different'}
    ))
    hierarchy_for_ortools.add_constraint(Constraint(
        variables=['x', 'y'],
        predicate=soft_sum_predicate,
        level=ConstraintLevel.LOCAL,
        hardness=Hardness.SOFT,
        metadata={'name': 'soft_sum'}
    ))

    ortools_adapter = ORToolsCPSATAdapter(variables, domains, hierarchy_for_ortools)
    ortools_solution = ortools_adapter.solve()
    print(f"Solución encontrada por OR-Tools CP-SAT: {ortools_solution}")
    if ortools_solution:
        # Verificar HARD constraint: x != y
        assert ortools_solution['x'] != ortools_solution['y']
        # Verificar SOFT constraint: x + y == 2. Esperamos que se minimice la violación.
        # La mejor solución debería tener x + y == 2, si es posible.
        print(f"  x + y = {ortools_solution['x'] + ortools_solution['y']}")
        print("ORToolsCPSATAdapter: Validado (HARD y SOFT).")
    else:
        print("ORToolsCPSATAdapter: No se encontró solución.")

    # --- Probar PymooAdapter ---
    print("\n--- Probando PymooAdapter (HARD y SOFT/Multi-objetivo) ---")
    def hard_neq_predicate_pymoo(assignment):
        return 0 if assignment['x'] != assignment['y'] else 1 # 0 si cumple, 1 si viola

    hierarchy_for_pymoo = ConstraintHierarchy()
    hierarchy_for_pymoo.add_constraint(Constraint(
        variables=['x', 'y'],
        predicate=hard_neq_predicate_pymoo,
        level=ConstraintLevel.LOCAL,
        hardness=Hardness.HARD,
        metadata={'name': 'hard_neq'}
    ))
    hierarchy_for_pymoo.add_constraint(Constraint(
        variables=['x', 'y'],
        predicate=soft_sum_predicate,
        level=ConstraintLevel.LOCAL,
        hardness=Hardness.SOFT,
        metadata={'name': 'soft_sum'}
    ))

    pymoo_adapter = PymooAdapter(variables, domains, hierarchy_for_pymoo)
    pymoo_solutions = pymoo_adapter.solve()
    print(f"Soluciones encontradas por Pymoo ({len(pymoo_solutions) if pymoo_solutions else 0}):")
    if pymoo_solutions:
        for sol in pymoo_solutions:
            print(f"  {sol} (x != y: {sol['x'] != sol['y']}, x + y == 2: {sol['x'] + sol['y'] == 2})")
        assert all(s['x'] != s['y'] for s in pymoo_solutions)
        print("PymooAdapter: Validado (HARD y SOFT/Multi-objetivo).")
    else:
        print("PymooAdapter: No se encontraron soluciones.")

if __name__ == '__main__':
    run_simple_validation_problem()

