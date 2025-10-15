"""
Fixtures compartidas para tests de fibration flow.

Este módulo proporciona fixtures reutilizables para facilitar
la creación de tests con diferentes problemas CSP y de optimización.
"""

import pytest
from lattice_weaver.fibration import (
    ConstraintHierarchy,
    Constraint,
    ConstraintLevel,
    Hardness
)


@pytest.fixture
def simple_csp_problem():
    """
    Problema CSP simple: 3 variables, dominios [0,1,2], all_different.
    
    Returns:
        Tupla (variables, domains, hierarchy)
    """
    variables = ["x", "y", "z"]
    domains = {"x": [0, 1, 2], "y": [0, 1, 2], "z": [0, 1, 2]}
    hierarchy = ConstraintHierarchy()
    
    # All different constraints (HARD)
    hierarchy.add_local_constraint(
        "x", "y",
        lambda a: a["x"] != a["y"],
        hardness=Hardness.HARD,
        metadata={"name": "x_neq_y"}
    )
    hierarchy.add_local_constraint(
        "x", "z",
        lambda a: a["x"] != a["z"],
        hardness=Hardness.HARD,
        metadata={"name": "x_neq_z"}
    )
    hierarchy.add_local_constraint(
        "y", "z",
        lambda a: a["y"] != a["z"],
        hardness=Hardness.HARD,
        metadata={"name": "y_neq_z"}
    )
    
    return variables, domains, hierarchy


@pytest.fixture
def optimization_problem():
    """
    Problema de optimización: minimizar suma con restricciones.
    
    Variables: a, b, c con dominios [1, 2, 3]
    HARD: a != b
    SOFT: minimizar suma de valores
    
    Returns:
        Tupla (variables, domains, hierarchy)
    """
    variables = ["a", "b", "c"]
    domains = {"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]}
    hierarchy = ConstraintHierarchy()
    
    # HARD: a != b
    hierarchy.add_local_constraint(
        "a", "b",
        lambda x: x["a"] != x["b"],
        hardness=Hardness.HARD,
        metadata={"name": "a_neq_b"}
    )
    
    # SOFT: minimizar suma (normalizado a [0, 1])
    hierarchy.add_global_constraint(
        ["a", "b", "c"],
        lambda x: sum(x.values()) / 9.0,  # Máximo posible: 9
        hardness=Hardness.SOFT,
        metadata={"name": "minimize_sum", "objective": "minimize"}
    )
    
    return variables, domains, hierarchy


@pytest.fixture
def nqueens_4x4():
    """
    Problema N-Queens 4x4.
    
    4 variables (reinas), cada una con dominio [0, 1, 2, 3] (columnas).
    Restricciones: no dos reinas en la misma columna, fila o diagonal.
    
    Returns:
        Tupla (variables, domains, hierarchy)
    """
    variables = ["Q0", "Q1", "Q2", "Q3"]  # Reinas en filas 0, 1, 2, 3
    domains = {var: [0, 1, 2, 3] for var in variables}  # Columnas posibles
    hierarchy = ConstraintHierarchy()
    
    # Restricciones: no dos reinas en la misma columna o diagonal
    for i in range(4):
        for j in range(i + 1, 4):
            var_i = f"Q{i}"
            var_j = f"Q{j}"
            
            def make_constraint(row_i, row_j):
                def constraint(a):
                    col_i = a[f"Q{row_i}"]
                    col_j = a[f"Q{row_j}"]
                    # No misma columna
                    if col_i == col_j:
                        return False
                    # No misma diagonal
                    if abs(col_i - col_j) == abs(row_i - row_j):
                        return False
                    return True
                return constraint
            
            hierarchy.add_local_constraint(
                var_i, var_j,
                make_constraint(i, j),
                hardness=Hardness.HARD,
                metadata={"name": f"queens_{i}_{j}"}
            )
    
    return variables, domains, hierarchy


@pytest.fixture
def graph_coloring_problem():
    """
    Problema de 3-coloreo de un grafo pequeño.
    
    Grafo: 4 nodos, aristas: (0,1), (1,2), (2,3), (3,0), (0,2)
    Colores: [0, 1, 2] (3 colores)
    
    Returns:
        Tupla (variables, domains, hierarchy)
    """
    variables = ["v0", "v1", "v2", "v3"]
    domains = {var: [0, 1, 2] for var in variables}  # 3 colores
    hierarchy = ConstraintHierarchy()
    
    # Aristas del grafo
    edges = [("v0", "v1"), ("v1", "v2"), ("v2", "v3"), ("v3", "v0"), ("v0", "v2")]
    
    for v1, v2 in edges:
        hierarchy.add_local_constraint(
            v1, v2,
            lambda a, var1=v1, var2=v2: a[var1] != a[var2],
            hardness=Hardness.HARD,
            metadata={"name": f"edge_{v1}_{v2}"}
        )
    
    return variables, domains, hierarchy


@pytest.fixture
def trivial_problem():
    """
    Problema trivial: 1 variable, 1 valor, sin restricciones.
    
    Returns:
        Tupla (variables, domains, hierarchy)
    """
    variables = ["x"]
    domains = {"x": [1]}
    hierarchy = ConstraintHierarchy()
    return variables, domains, hierarchy


@pytest.fixture
def unsolvable_problem():
    """
    Problema sin solución: 2 variables, dominios [0], restricción x != y.
    
    Returns:
        Tupla (variables, domains, hierarchy)
    """
    variables = ["x", "y"]
    domains = {"x": [0], "y": [0]}
    hierarchy = ConstraintHierarchy()
    
    hierarchy.add_local_constraint(
        "x", "y",
        lambda a: a["x"] != a["y"],
        hardness=Hardness.HARD,
        metadata={"name": "x_neq_y"}
    )
    
    return variables, domains, hierarchy


@pytest.fixture
def soft_constraints_problem():
    """
    Problema con solo restricciones SOFT (preferencias).
    
    Variables: x, y con dominios [0, 1, 2]
    SOFT: preferir x + y cercano a 3
    
    Returns:
        Tupla (variables, domains, hierarchy)
    """
    variables = ["x", "y"]
    domains = {"x": [0, 1, 2], "y": [0, 1, 2]}
    hierarchy = ConstraintHierarchy()
    
    # SOFT: preferir que x + y esté cerca de 3
    def distance_from_target(a):
        total = a["x"] + a["y"]
        return abs(total - 3) / 4.0  # Normalizado (máxima distancia: 4)
    
    hierarchy.add_global_constraint(
        ["x", "y"],
        distance_from_target,
        hardness=Hardness.SOFT,
        metadata={"name": "sum_near_3", "objective": "minimize"}
    )
    
    return variables, domains, hierarchy

