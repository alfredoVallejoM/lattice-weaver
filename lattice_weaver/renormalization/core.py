# lattice_weaver/renormalization/core.py

"""
Núcleo de la Renormalización Computacional

Este módulo implementa el flujo principal de la renormalización computacional dentro de LatticeWeaver.
Proporciona funciones para realizar un paso de renormalización de un CSP, construir jerarquías
de abstracción multinivel y refinar soluciones desde niveles abstractos al original.

Autor: LatticeWeaver Development Team
Fecha: 14 de Octubre de 2025
"""

from typing import List, Set, Dict, Tuple, Any, Optional, Callable
from collections import defaultdict

from ..core.csp_problem import CSP, Constraint, verify_solution, is_satisfiable
from ..core.simple_backtracking_solver import solve_csp_backtracking as simple_backtracking_solver
from .partition import VariablePartitioner
from .effective_domains import EffectiveDomainDeriver, LazyEffectiveDomain
from .effective_constraints import EffectiveConstraintDeriver, LazyEffectiveConstraint
from .hierarchy import AbstractionHierarchy, AbstractionLevel


def renormalize_csp(
    source_csp: CSP,
    k: int,
    partition_strategy: str = 'metis',
    domain_deriver: Optional[EffectiveDomainDeriver] = None,
    constraint_deriver: Optional[EffectiveConstraintDeriver] = None,
    source_level: int = 0, # Nivel de abstracción del CSP de entrada
    target_level: Optional[int] = None # Si se especifica, construye hasta este nivel
) -> Tuple[Optional[CSP], Optional[List[Set[str]]], Optional[Dict[str, str]]]:
    """
    Realiza un paso de renormalización o construye una jerarquía de abstracción.

    Args:
        source_csp: El CSP original o de nivel inferior a renormalizar.
        k: El factor de renormalización (número de grupos).
        partition_strategy: Estrategia para particionar las variables.
        domain_deriver: Instancia de EffectiveDomainDeriver. Si es None, se crea uno.
        constraint_deriver: Instancia de EffectiveConstraintDeriver. Si es None, se crea uno.
        source_level: El nivel de abstracción del CSP de entrada (0 para el original).
        target_level: Si se especifica, construye una jerarquía hasta este nivel.
                      Si es None, realiza un solo paso de renormalización.

    Returns:
        Returns:
            Tuple[Optional[CSP | AbstractionHierarchy], Optional[List[Set[str]]], Optional[Dict[str, str]]]:
            Si `target_level` es `None` (un solo paso de renormalización):
                - El nuevo CSP renormalizado (nivel L+1).
                - La partición de variables utilizada para crear el CSP renormalizado.
                - Un diccionario que mapea las variables originales a las nuevas variables renormalizadas.
            Si `target_level` se especifica (construcción de jerarquía):
                - Un objeto `AbstractionHierarchy` que contiene todos los niveles de abstracción construidos.
                - `None` (la partición y el mapa de variables están dentro de la jerarquía).
                - `None` (la partición y el mapa de variables están dentro de la jerarquía).
    """
    if domain_deriver is None:
        domain_deriver = EffectiveDomainDeriver()
    if constraint_deriver is None:
        constraint_deriver = EffectiveConstraintDeriver()

    partitioner = VariablePartitioner(strategy=partition_strategy)
    partition_result = partitioner.partition(source_csp, k)

    if partition_result is None:
        return None, None, None

    partition = [group for group in partition_result if group]

    if not partition:
        return None, None, None

    # Crear nuevas variables y el mapa de variables
    variable_map = {}
    renormalized_variables = set()
    for i, group in enumerate(partition):
        new_var_name = f"L{source_level+1}_G{i}"
        renormalized_variables.add(new_var_name)
        for old_var in group:
            variable_map[old_var] = new_var_name

    renormalized_domains_lazy: Dict[str, LazyEffectiveDomain] = {}
    for i, group in enumerate(partition):
        group_name = f"L{source_level+1}_G{i}"
        renormalized_domains_lazy[group_name] = LazyEffectiveDomain(domain_deriver, source_csp, group)

    renormalized_constraints: List[Constraint] = []
    for i in range(len(partition)):
        for j in range(i + 1, len(partition)):
            group1_name = f"L{source_level+1}_G{i}"
            group2_name = f"L{source_level+1}_G{j}"
            group1 = partition[i]
            group2 = partition[j]

            lazy_effective_constraint = LazyEffectiveConstraint(
                constraint_deriver,
                source_csp,
                group1,
                group2,
                renormalized_domains_lazy[group1_name],
                renormalized_domains_lazy[group2_name]
            )
            renormalized_constraints.append(Constraint(
                scope=frozenset({group1_name, group2_name}),
                relation=lazy_effective_constraint,
                name=f"RG_C_{group1_name}_{group2_name}"
            ))

    renormalized_domains_concrete = {}
    for name, lazy_domain in renormalized_domains_lazy.items():
        domain_values = lazy_domain.get()
        if not domain_values:
            return None, None, None
        renormalized_domains_concrete[name] = domain_values

    renormalized_csp = CSP(
        variables=renormalized_variables,
        domains=renormalized_domains_concrete,
        constraints=renormalized_constraints,
        name=f"L{source_level+1}_{source_csp.name or 'CSP'}"
    )

    if target_level is None:
        return renormalized_csp, partition, variable_map
    else:
        # Si se especifica target_level, construimos la jerarquía
        hierarchy = AbstractionHierarchy(source_csp)
        hierarchy.add_level(source_level + 1, renormalized_csp, partition, variable_map)

        current_level_csp = renormalized_csp
        current_level = source_level + 1

        while current_level < target_level:
            next_k = k # Asumimos k constante para la jerarquía, se puede hacer más flexible
            next_strategy = partition_strategy # Asumimos estrategia constante

            next_csp, next_partition, next_var_map = renormalize_csp(
                source_csp=current_level_csp,
                k=next_k,
                partition_strategy=next_strategy,
                domain_deriver=domain_deriver,
                constraint_deriver=constraint_deriver,
                source_level=current_level,
                target_level=None # Un solo paso
            )

            if next_csp is None or next_partition is None or next_var_map is None:
                break
            
            hierarchy.add_level(current_level + 1, next_csp, next_partition, next_var_map)
            current_level_csp = next_csp
            current_level += 1
        
        return hierarchy, None, None # Devolvemos la jerarquía

def refine_solution(
    renormalized_solution: Dict[str, Any],
    original_csp: CSP,
    partition: List[Set[str]],
    domain_deriver: Optional[EffectiveDomainDeriver] = None
) -> Optional[Dict[str, Any]]:
    """
    Refina una solución de un CSP renormalizado a una solución del CSP original.
    
    Args:
        renormalized_solution: La solución del CSP renormalizado.
        original_csp: El CSP original.
        partition: La partición de variables utilizada para la renormalización que generó el CSP abstracto.
                   Es una lista de conjuntos de variables del CSP original que forman cada variable abstracta.
        domain_deriver: Instancia de EffectiveDomainDeriver. Si es None, se crea uno.
        
    Returns:
        Una solución del CSP original, o None si no se puede refinar.
    """
    if domain_deriver is None:
        domain_deriver = EffectiveDomainDeriver()

    refined_solution = {}
    for i, group in enumerate(partition):
        group_name = f"G{i}"
        if group_name not in renormalized_solution:
            return None # La solución renormalizada no es completa
        
        effective_value = renormalized_solution[group_name]
        
        # Reconstruir las asignaciones para las variables originales dentro del grupo
        # que corresponden al effective_value
        possible_original_assignments = domain_deriver.get_original_assignments(
            original_csp, group, effective_value
        )
        
        if not possible_original_assignments:
            return None # No se puede refinar este valor efectivo
        
        # Para simplificar, tomamos la primera asignación posible. 
        # En un sistema real, esto podría requerir un backtracking o una búsqueda más inteligente.
        # TODO: Implementar un mecanismo más robusto para seleccionar la asignación original.
        # Esto podría implicar un backtracking o una búsqueda heurística para asegurar la consistencia
        # con otras variables del mismo nivel, o la integración con un solver de bajo nivel.
        selected_assignment = possible_original_assignments[0]
        refined_solution.update(selected_assignment)
        
    # Verificar que la solución refinada es válida para el CSP original
    if verify_solution(original_csp, refined_solution):
        return refined_solution
    else:
        return None

class RenormalizationSolver:
    """
    Solver de CSP que utiliza el flujo de renormalización multinivel.
    """
    def __init__(self, target_level: int = 3, k: int = 2, partition_strategy: str = 'metis'):
        self.target_level = target_level
        self.k = k
        self.partition_strategy = partition_strategy
        self.domain_deriver = EffectiveDomainDeriver()
        self.constraint_deriver = EffectiveConstraintDeriver()

    def solve(self, csp: CSP) -> Optional[Dict[str, Any]]:
        """
        Resuelve un CSP utilizando el flujo de renormalización multinivel.
        """
        # Usar la función unificada renormalize_csp para construir la jerarquía
        hierarchy, _, _ = renormalize_csp(
            source_csp=csp,
            k=self.k,
            partition_strategy=self.partition_strategy,
            domain_deriver=self.domain_deriver,
            constraint_deriver=self.constraint_deriver,
            source_level=0,
            target_level=self.target_level
        )

        if not hierarchy or not hierarchy.levels:
            return None

        # Intentar resolver el CSP más abstracto
        highest_level_csp = hierarchy.get_level(hierarchy.max_level).csp
        if highest_level_csp is None:
            return None

        abstract_solution = simple_backtracking_solver(highest_level_csp)

        if abstract_solution is None:
            return None

        # Refinar la solución de vuelta al nivel original
        current_solution = abstract_solution
        for level_idx in range(hierarchy.max_level, 0, -1):
            level_info = hierarchy.get_level(level_idx)
            if level_info.parent_level_info is None:
                return None
            
            lower_level_solution = {}
            parent_csp = level_info.parent_level_info.csp
            if parent_csp is None:
                return None

            for original_var in parent_csp.variables:
                renormalized_var = level_info.variable_map.get(original_var)
                if renormalized_var is None:
                    return None
                
                effective_value = current_solution.get(renormalized_var)
                if effective_value is None:
                    return None
                
                group_for_var = next((g for g in level_info.partition if original_var in g), None)
                if group_for_var is None:
                    return None

                possible_original_assignments = self.domain_deriver.get_original_assignments(
                    parent_csp, group_for_var, effective_value
                )
                
                if not possible_original_assignments:
                    return None
                
                lower_level_solution[original_var] = possible_original_assignments[0].get(original_var)
                if lower_level_solution[original_var] is None:
                    return None
            
            current_solution = lower_level_solution
            
            if not verify_solution(parent_csp, current_solution):
                return None

        return current_solution

