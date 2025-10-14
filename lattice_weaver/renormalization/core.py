'''
# lattice_weaver/renormalization/core.py

"""
Núcleo de la Renormalización Computacional

Este módulo implementa el flujo principal de la renormalización computacional,
incluyendo la capacidad de construir jerarquías de abstracción multinivel.
"""

from typing import List, Set, Dict, Tuple, Any, Optional, Callable
from collections import defaultdict

from ..core.csp_problem import CSP, Constraint, verify_solution
from ..core.simple_backtracking_solver import simple_backtracking_solver
from .partition import VariablePartitioner
from .effective_domains import EffectiveDomainDeriver, LazyEffectiveDomain
from .effective_constraints import EffectiveConstraintDeriver, LazyEffectiveConstraint
from .hierarchy import AbstractionHierarchy, AbstractionLevel


def renormalize_single_level(
    source_csp: CSP,
    source_level: int,
    k: int,
    partition_strategy: str = 'metis',
    domain_deriver: Optional[EffectiveDomainDeriver] = None,
    constraint_deriver: Optional[EffectiveConstraintDeriver] = None,
) -> Tuple[Optional[CSP], Optional[List[Set[str]]], Optional[Dict[str, str]]]:
    """
    Realiza un único paso de renormalización desde el nivel `source_level` al `source_level + 1`.

    Returns:
        - El nuevo CSP renormalizado (nivel L+1)
        - La partición utilizada
        - El mapa de variables de L a L+1
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

    return renormalized_csp, partition, variable_map

def renormalize_multilevel(
    original_csp: CSP,
    target_level: int = 6,
    k_function: Callable[[int], int] = lambda level: 2,
    strategy_function: Callable[[int], str] = lambda level: 'metis'
) -> AbstractionHierarchy:
    """
    Construye una jerarquía de abstracción hasta el nivel objetivo.
    """
    hierarchy = AbstractionHierarchy(original_csp)
    
    for level in range(target_level):
        current_csp = hierarchy.get_level(level).csp
        k = k_function(level)
        strategy = strategy_function(level)
        
        new_csp, partition, var_map = renormalize_single_level(
            source_csp=current_csp,
            source_level=level,
            k=k,
            partition_strategy=strategy
        )
        
        if new_csp is None or partition is None or var_map is None:
            break
            
        hierarchy.add_level(level + 1, new_csp, partition, var_map)
        
    return hierarchy

def refine_single_level(
    higher_level_solution: Dict[str, Any],
    level_info: AbstractionLevel
) -> Dict[str, Any]:
    """Refina una solución de un nivel superior (L+1) a un nivel inferior (L)."""
    lower_level_solution = {}
    partition = level_info.partition
    
    for group_idx, group_vars in enumerate(partition):
        group_name = f"L{level_info.level}_G{group_idx}"
        if group_name not in higher_level_solution:
            raise ValueError(f"El grupo {group_name} no se encontró en la solución del nivel superior.")
            
        group_assignment = higher_level_solution[group_name]
        sorted_group_vars = sorted(list(group_vars))
        
        if len(group_assignment) != len(sorted_group_vars):
            raise ValueError("La asignación del grupo no coincide con el número de variables.")

        for var_name, value in zip(sorted_group_vars, group_assignment):
            lower_level_solution[var_name] = value
            
    return lower_level_solution

class RenormalizationSolver:
    """
    Solver de CSP que utiliza el flujo de renormalización multinivel.
    """
    def __init__(self, target_level: int = 3, k: int = 2, partition_strategy: str = 'metis'):
        self.target_level = target_level
        self.k = k
        self.partition_strategy = partition_strategy

    def solve(self, csp: CSP) -> Optional[Dict[str, Any]]:
        """
        Resuelve un CSP utilizando el flujo de renormalización multinivel.
        """
        # 1. Renormalizar hasta el nivel objetivo
        hierarchy = renormalize_multilevel(
            csp, 
            self.target_level,
            k_function=lambda l: self.k,
            strategy_function=lambda l: self.partition_strategy
        )
        
        # 2. Resolver el CSP del nivel más alto
        top_csp = hierarchy.get_highest_csp()
        top_level_solution = simple_backtracking_solver(top_csp)
        
        if not top_level_solution:
            return None
            
        # 3. Refinar la solución hacia abajo a través de los niveles
        solution = self._refine_multilevel(hierarchy, top_level_solution)
        
        # 4. Verificar la solución final
        if not verify_solution(csp, solution):
            raise ValueError("La solución refinada no satisface el CSP original.")

        return solution

    def _refine_multilevel(self, hierarchy: AbstractionHierarchy, top_solution: Dict) -> Dict:
        """Refina la solución desde el nivel más alto hasta el nivel 0."""
        current_solution = top_solution
        for level in range(hierarchy.highest_level, 0, -1):
            level_info = hierarchy.get_level(level)
            # La solución del nivel `L` se refina al nivel `L-1`
            current_solution = refine_single_level(
                higher_level_solution=current_solution,
                level_info=level_info
            )
        return current_solution
'''
