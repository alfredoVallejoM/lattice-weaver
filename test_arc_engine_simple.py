"""
Script de diagnóstico simple para probar el ArcEngine.
"""

from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.arc_engine.csp_solver import CSPSolver
from lattice_weaver.arc_engine.constraints import register_relation
from lattice_weaver.benchmarks.generators import generate_nqueens

# Registrar relación de N-Queens
def not_attack_queens(a, b, metadata):
    """Relación para N-Queens: dos reinas no se atacan."""
    i = metadata.get('var1_idx')
    j = metadata.get('var2_idx')
    
    if i is None or j is None:
        print(f"WARNING: metadata missing for constraint: {metadata}")
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

# Crear ArcEngine manualmente
print("\n" + "="*60)
print("Creando ArcEngine manualmente")
print("="*60)

engine = ArcEngine()

# Añadir variables
for var in csp.variables:
    engine.add_variable(var, csp.domains[var])
    print(f"✓ Variable '{var}' añadida con dominio {list(csp.domains[var])}")

# Añadir restricciones
for i, constraint in enumerate(csp.constraints):
    if len(constraint.scope) == 2:
        scope_list = list(constraint.scope)
        var1, var2 = scope_list[0], scope_list[1]
        
        # Extraer índices
        var1_idx = int(var1[1:]) if var1[1:].isdigit() else None
        var2_idx = int(var2[1:]) if var2[1:].isdigit() else None
        
        metadata = {}
        if var1_idx is not None and var2_idx is not None:
            metadata['var1_idx'] = var1_idx
            metadata['var2_idx'] = var2_idx
        
        engine.add_constraint(var1, var2, "not_attack_queens", metadata=metadata, cid=f"c{i}")
        print(f"✓ Restricción {i}: {var1} <-> {var2} (metadata: {metadata})")

# Probar AC-3
print("\n" + "="*60)
print("Ejecutando AC-3")
print("="*60)

consistent = engine.enforce_arc_consistency()
print(f"Resultado de AC-3: {'CONSISTENTE' if consistent else 'INCONSISTENTE'}")

if consistent:
    print("\nDominios después de AC-3:")
    for var in sorted(engine.variables.keys()):
        domain = list(engine.variables[var].get_values())
        print(f"  {var}: {domain}")

# Crear solver y resolver
print("\n" + "="*60)
print("Creando CSPSolver y resolviendo")
print("="*60)

solver = CSPSolver(engine)
result = solver.solve(return_all=False, max_solutions=1)

print(f"Soluciones encontradas: {len(result.solutions)}")
print(f"Nodos explorados: {result.nodes_explored}")

if result.solutions:
    print(f"\nPrimera solución:")
    for var, val in sorted(result.solutions[0].items()):
        print(f"  {var} = {val}")
else:
    print("\n⚠ No se encontró solución")

