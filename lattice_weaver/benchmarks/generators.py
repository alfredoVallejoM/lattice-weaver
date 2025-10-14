"""
Generadores de problemas CSP para benchmarking.

Este módulo proporciona funciones para generar problemas CSP de diferentes
dominios y complejidades para usar en pruebas de rendimiento.
"""

from lattice_weaver.core.csp_problem import CSP, Constraint


def generate_nqueens(n: int) -> CSP:
    """
    Genera un problema de N-Reinas.
    
    Args:
        n: Tamaño del tablero (número de reinas).
        
    Returns:
        Problema CSP de N-Reinas.
    """
    # Crear variables: una por cada fila (valor = columna)
    variables = set([f"Q{i}" for i in range(n)])
    
    # Dominios: columnas disponibles (0 a n-1)
    domains = {var: frozenset(range(n)) for var in variables}
    
    # Restricciones: no dos reinas en la misma columna o diagonal
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            row_diff = j - i
            var_i = f"Q{i}"
            var_j = f"Q{j}"
            # Restricción: columnas diferentes y no en la misma diagonal
            constraints.append(Constraint(
                scope=frozenset([var_i, var_j]),
                relation=lambda col_i, col_j, rd=row_diff: col_i != col_j and abs(col_i - col_j) != rd,
                name=f"NoAttack_Q{i}_Q{j}"
            ))
    
    return CSP(variables, domains, constraints)


def generate_sudoku(size: int = 9) -> CSP:
    """
    Genera un problema de Sudoku.
    
    Args:
        size: Tamaño del tablero (9 o 16).
        
    Returns:
        Problema CSP de Sudoku.
    """
    # Crear variables: una por cada celda
    variables = set([f"cell_{i}_{j}" for i in range(size) for j in range(size)])
    
    # Dominios: 1 a size para cada variable
    domains = {var: frozenset(range(1, size + 1)) for var in variables}
    
    # Restricciones: todos diferentes en filas, columnas y bloques
    constraints = []
    
    # Restricciones de filas
    for i in range(size):
        row_vars = [f"cell_{i}_{j}" for j in range(size)]
        for j1 in range(size):
            for j2 in range(j1 + 1, size):
                constraints.append(Constraint(
                    scope=frozenset([row_vars[j1], row_vars[j2]]),
                    relation=lambda x, y: x != y,
                    name=f"Row{i}_NE_{j1}_{j2}"
                ))
    
    # Restricciones de columnas
    for j in range(size):
        col_vars = [f"cell_{i}_{j}" for i in range(size)]
        for i1 in range(size):
            for i2 in range(i1 + 1, size):
                constraints.append(Constraint(
                    scope=frozenset([col_vars[i1], col_vars[i2]]),
                    relation=lambda x, y: x != y,
                    name=f"Col{j}_NE_{i1}_{i2}"
                ))
    
    # Restricciones de bloques (solo para 9x9 y 16x16)
    if size == 9:
        block_size = 3
    elif size == 16:
        block_size = 4
    else:
        block_size = int(size ** 0.5)
    
    for block_i in range(size // block_size):
        for block_j in range(size // block_size):
            block_vars = []
            for i in range(block_i * block_size, (block_i + 1) * block_size):
                for j in range(block_j * block_size, (block_j + 1) * block_size):
                    block_vars.append(f"cell_{i}_{j}")
            
            for v1_idx in range(len(block_vars)):
                for v2_idx in range(v1_idx + 1, len(block_vars)):
                    constraints.append(Constraint(
                        scope=frozenset([block_vars[v1_idx], block_vars[v2_idx]]),
                        relation=lambda x, y: x != y,
                        name=f"Block{block_i}_{block_j}_NE_{v1_idx}_{v2_idx}"
                    ))
    
    return CSP(variables, domains, constraints)


def generate_graph_coloring(num_nodes: int, edge_probability: float = 0.3, num_colors: int = 3) -> CSP:
    """
    Genera un problema de coloreado de grafos aleatorio.
    
    Args:
        num_nodes: Número de nodos en el grafo.
        edge_probability: Probabilidad de que exista una arista entre dos nodos.
        num_colors: Número de colores disponibles.
        
    Returns:
        Problema CSP de coloreado de grafos.
    """
    import random
    
    # Crear variables: una por cada nodo
    variables = set([f"node_{i}" for i in range(num_nodes)])
    
    # Dominios: colores disponibles
    domains = {var: frozenset(range(num_colors)) for var in variables}
    
    # Generar aristas aleatoriamente
    constraints = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_probability:
                constraints.append(Constraint(
                    scope=frozenset([f"node_{i}", f"node_{j}"]),
                    relation=lambda x, y: x != y,
                    name=f"Edge_{i}_{j}"
                ))
    
    return CSP(variables, domains, constraints)


def generate_job_shop_scheduling(num_jobs: int, num_machines: int) -> CSP:
    """
    Genera un problema simplificado de Job Shop Scheduling.
    
    Args:
        num_jobs: Número de trabajos.
        num_machines: Número de máquinas.
        
    Returns:
        Problema CSP de Job Shop Scheduling.
    """
    import random
    
    # Crear variables: tiempo de inicio de cada operación
    variables = set()
    for job in range(num_jobs):
        for op in range(num_machines):
            variables.add(f"job_{job}_op_{op}")
    
    # Dominios: tiempos de inicio posibles (0 a 100)
    domains = {var: frozenset(range(100)) for var in variables}
    
    # Restricciones de precedencia: operaciones del mismo trabajo deben ser secuenciales
    constraints = []
    for job in range(num_jobs):
        for op in range(num_machines - 1):
            var1 = f"job_{job}_op_{op}"
            var2 = f"job_{job}_op_{op + 1}"
            duration = random.randint(1, 10)
            constraints.append(Constraint(
                scope=frozenset([var1, var2]),
                relation=lambda x, y, d=duration: x + d <= y,
                name=f"Precedence_job{job}_op{op}"
            ))
    
    # Restricciones de máquina: operaciones en la misma máquina no pueden solaparse
    for machine in range(num_machines):
        machine_ops = [f"job_{job}_op_{machine}" for job in range(num_jobs)]
        for i in range(len(machine_ops)):
            for j in range(i + 1, len(machine_ops)):
                duration_i = random.randint(1, 10)
                duration_j = random.randint(1, 10)
                constraints.append(Constraint(
                    scope=frozenset([machine_ops[i], machine_ops[j]]),
                    relation=lambda x, y, di=duration_i, dj=duration_j: x + di <= y or y + dj <= x,
                    name=f"Machine{machine}_NoOverlap_{i}_{j}"
                ))
    
    return CSP(variables, domains, constraints)


def generate_simple_csp(num_variables: int, domain_size: int, constraint_density: float = 0.3) -> CSP:
    """
    Genera un problema CSP simple con restricciones binarias aleatorias.
    
    Args:
        num_variables: Número de variables.
        domain_size: Tamaño del dominio de cada variable.
        constraint_density: Densidad de restricciones (0.0 a 1.0).
        
    Returns:
        Problema CSP simple.
    """
    import random
    
    # Crear variables
    variables = set([f"var_{i}" for i in range(num_variables)])
    
    # Dominios
    domains = {var: frozenset(range(domain_size)) for var in variables}
    
    # Generar restricciones aleatorias
    constraints = []
    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            if random.random() < constraint_density:
                # Restricción aleatoria: x != y o x < y o x + y < domain_size
                constraint_type = random.choice(['ne', 'lt', 'sum'])
                if constraint_type == 'ne':
                    constraints.append(Constraint(
                        scope=frozenset([f"var_{i}", f"var_{j}"]),
                        relation=lambda x, y: x != y,
                        name=f"NE_{i}_{j}"
                    ))
                elif constraint_type == 'lt':
                    constraints.append(Constraint(
                        scope=frozenset([f"var_{i}", f"var_{j}"]),
                        relation=lambda x, y: x < y,
                        name=f"LT_{i}_{j}"
                    ))
                else:  # sum
                    threshold = random.randint(domain_size // 2, domain_size * 2)
                    constraints.append(Constraint(
                        scope=frozenset([f"var_{i}", f"var_{j}"]),
                        relation=lambda x, y, t=threshold: x + y < t,
                        name=f"SUM_{i}_{j}"
                    ))
    
    return CSP(variables, domains, constraints)

