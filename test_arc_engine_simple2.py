"""
Script de diagnóstico simple para probar el CSPSolver correctamente.
"""

from lattice_weaver.arc_engine.csp_solver import CSPSolver, CSPProblem
from lattice_weaver.arc_engine.constraints import register_relation
from lattice_weaver.benchmarks.generators import generate_nqueens

# Registrar relación de N-Queens
def not_attack_queens(a, b, metadata):
    """Relación para N-Queens: dos reinas no se atacan."""
    i = metadata.get('var1_idx')
    j = metadata.get('var2_idx')
    
    if i is None or j is None:
        return False
    
    # No misma columna
    if a == b:
        return False
    
    # No misma diagonal
    if abs(a - b) == abs(i - j):
        return False
    
    return True

# Intentar registrar (puede fallar si ya está registrada)
try:
    register_relation("not_attack_queens", not_attack_queens)
    print("✓ Relación 'not_attack_queens' registrada")
except ValueError as e:
    print(f"⚠ Relación ya registrada: {e}")

# Generar problema de 4-Queens
print("\n" + "="*60)
print("Generando problema de 4-Queens")
print("="*60)

csp = generate_nqueens(4)
print(f"Variables: {csp.variables}")
print(f"Dominios: {csp.domains}")
print(f"Número de restricciones: {len(csp.constraints)}")

# Convertir a CSPProblem
print("\n" + "="*60)
print("Convirtiendo a CSPProblem")
print("="*60)

variables_list = sorted(list(csp.variables))
domains_dict = {var: list(csp.domains[var]) for var in variables_list}

# Crear lista de restricciones con nombres de relación
constraints_list = []
for i, constraint in enumerate(csp.constraints):
    if len(constraint.scope) == 2:
        scope_list = list(constraint.scope)
        var1, var2 = scope_list[0], scope_list[1]
        
        # Usar el nombre de relación registrado
        constraints_list.append((var1, var2, "not_attack_queens"))

print(f"Variables: {variables_list}")
print(f"Dominios: {domains_dict}")
print(f"Restricciones: {len(constraints_list)}")

problem = CSPProblem(variables_list, domains_dict, constraints_list)

# Crear solver y resolver
print("\n" + "="*60)
print("Creando CSPSolver y resolviendo")
print("="*60)

solver = CSPSolver(use_tms=False, parallel=False)
result = solver.solve(problem, return_all=False, max_solutions=1)

print(f"Soluciones encontradas: {len(result.solutions)}")
print(f"Nodos explorados: {result.nodes_explored}")

if result.solutions:
    print(f"\nPrimera solución:")
    for var, val in sorted(result.solutions[0].assignment.items()):
        print(f"  {var} = {val}")
else:
    print("\n⚠ No se encontró solución")

