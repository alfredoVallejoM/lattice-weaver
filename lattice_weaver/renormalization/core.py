# lattice_weaver/renormalization/core.py

"""
Núcleo de la Renormalización Computacional

Este módulo implementa el flujo principal de la renormalización computacional,
incluyendo la capacidad de construir jerarquías de abstracción multinivel,
así como funciones para renormalizar un CSP y refinar soluciones.
"""

from typing import List, Set, Dict, Tuple, Any, Optional, Callable
from collections import defaultdict

from ..core.csp_problem import CSP, Constraint, verify_solution, is_satisfiable
from ..core.simple_backtracking_solver import solve_csp_backtracking as simple_backtracking_solver
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

    Args:
        source_csp: El CSP original o de nivel inferior a renormalizar.
        source_level: El nivel de abstracción actual (0 para el CSP original).
        k: El factor de renormalización (número de grupos).
        partition_strategy: Estrategia para particionar las variables.
        domain_deriver: Instancia de EffectiveDomainDeriver. Si es None, se crea uno.
        constraint_deriver: Instancia de EffectiveConstraintDeriver. Si es None, se crea uno.

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

def renormalize_csp(
    original_csp: CSP,
    k: int,
    partition_strategy: str = 'metis',
    domain_deriver: Optional[EffectiveDomainDeriver] = None,
    constraint_deriver: Optional[EffectiveConstraintDeriver] = None,
) -> Tuple[Optional[CSP], Optional[List[Set[str]]]]:
    """
    Renormaliza un CSP dado, transformándolo en un CSP con menos variables
    pero con dominios y restricciones efectivas. Esta es una versión de un solo paso.
    
    Args:
        original_csp: El CSP original a renormalizar.
        k: El factor de renormalización (número de grupos).
        partition_strategy: Estrategia para particionar las variables.
        domain_deriver: Instancia de EffectiveDomainDeriver. Si es None, se crea uno.
        constraint_deriver: Instancia de EffectiveConstraintDeriver. Si es None, se crea uno.
    
    Returns:
        Una tupla que contiene:
        - El CSP renormalizado.
        - La partición de variables original utilizada.
    """
    if domain_deriver is None:
        domain_deriver = EffectiveDomainDeriver()
    if constraint_deriver is None:
        constraint_deriver = EffectiveConstraintDeriver()

    partitioner = VariablePartitioner(strategy=partition_strategy)
    partition_result = partitioner.partition(original_csp, k)

    if partition_result is None:
        return None, None

    partition = [group for group in partition_result if group]

    if not partition:
        return None, None
    
    renormalized_variables = {f"G{i}" for i in range(len(partition))}

    renormalized_domains_lazy: Dict[str, LazyEffectiveDomain] = {}
    for i, group in enumerate(partition):
        group_name = f"G{i}"
        renormalized_domains_lazy[group_name] = LazyEffectiveDomain(domain_deriver, original_csp, group)

    renormalized_constraints: List[Constraint] = []
    for i in range(len(partition)):
        for j in range(i + 1, len(partition)):
            group1_name = f"G{i}"
            group2_name = f"G{j}"
            group1 = partition[i]
            group2 = partition[j]

            lazy_effective_constraint = LazyEffectiveConstraint(
                constraint_deriver,
                original_csp,
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
            return None, None
        renormalized_domains_concrete[name] = domain_values

    renormalized_csp = CSP(
        variables=renormalized_variables,
        domains=renormalized_domains_concrete,
        constraints=renormalized_constraints,
        name=f"RG_{original_csp.name or 'CSP'}"
    )

    return renormalized_csp, partition

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
        partition: La partición de variables utilizada para la renormalización.
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
        hierarchy = renormalize_multilevel(
            original_csp=csp,
            target_level=self.target_level,
            k_function=lambda level: self.k,
            strategy_function=lambda level: self.partition_strategy
        )

        if not hierarchy.levels:
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
                # Esto no debería pasar si la jerarquía está bien construida
                return None
            
            # Mapear la solución abstracta a la partición del nivel inferior
            # La función refine_solution opera de un CSP renormalizado a su original.
            # Aquí necesitamos ir de la solución del CSP de nivel L+1 a la solución del CSP de nivel L.
            # Esto implica usar el `variable_map` y la `partition` del nivel_info.
            
            # Reconstruir la solución para el CSP del nivel inferior (parent_level_info.csp)
            # a partir de la solución del CSP actual (level_info.csp)
            
            # La `refine_solution` que tenemos está diseñada para ir de un CSP renormalizado a su original.
            # Necesitamos adaptar esto para ir de un nivel L+1 a L.
            # La `variable_map` en `level_info` mapea variables originales a variables renormalizadas.
            # Necesitamos el inverso: variables renormalizadas a variables originales.
            
            # Para simplificar, vamos a usar la lógica de `refine_solution` pero adaptada.
            # La `renormalized_solution` sería `current_solution`.
            # El `original_csp` sería `level_info.parent_level_info.csp`.
            # La `partition` sería `level_info.partition`.
            
            # Esto es un poco más complejo de lo que `refine_solution` maneja directamente.
            # `refine_solution` asume que `partition` es la partición del `original_csp`.
            # Aquí, `level_info.partition` es la partición del `parent_level_info.csp`.
            
            # Vamos a usar una lógica de refinamiento más directa basada en el `variable_map`
            # y los dominios efectivos.
            
            lower_level_solution = {}
            parent_csp = level_info.parent_level_info.csp
            if parent_csp is None:
                return None

            # Iterar sobre las variables del nivel inferior
            for original_var in parent_csp.variables:
                # Encontrar a qué variable renormalizada pertenece
                renormalized_var = level_info.variable_map.get(original_var)
                if renormalized_var is None:
                    # Esto no debería pasar si el mapa es completo
                    return None
                
                # Obtener el valor efectivo de la solución abstracta
                effective_value = current_solution.get(renormalized_var)
                if effective_value is None:
                    return None
                
                # Derivar los valores originales posibles para esta variable
                # que son consistentes con el effective_value
                group_for_var = next((g for g in level_info.partition if original_var in g), None)
                if group_for_var is None:
                    return None

                possible_original_assignments = self.domain_deriver.get_original_assignments(
                    parent_csp, group_for_var, effective_value
                )
                
                if not possible_original_assignments:
                    return None
                
                # Tomar la primera asignación posible para esta variable
                # TODO: Esto es una simplificación. Necesita un mecanismo más robusto.
                lower_level_solution[original_var] = possible_original_assignments[0].get(original_var)
                if lower_level_solution[original_var] is None:
                    return None
            
            current_solution = lower_level_solution
            
            # Verificar la solución en cada paso de refinamiento (opcional, pero buena práctica)
            if not verify_solution(parent_csp, current_solution):
                return None

        return current_solution



