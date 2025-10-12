"""
Paralelización Topológica para AC-3

Este módulo implementa la paralelización del algoritmo AC-3 explotando
la estructura topológica del grafo de restricciones. Utiliza los grupos
de restricciones independientes identificados por HomotopyRules para
procesarlos en paralelo usando multiprocessing.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from multiprocessing import Pool, Manager, cpu_count
from typing import Dict, List, Set, Tuple, Any, Optional
import logging
from copy import deepcopy
from collections import deque

logger = logging.getLogger(__name__)

def _worker_init():
    """Función de inicialización para cada worker del pool."""
    from .constraints import register_relation, nqueens_not_equal, nqueens_not_diagonal, RELATION_REGISTRY
    
    # Asegurar que las relaciones se registren en cada proceso hijo
    if "nqueens_not_equal" not in RELATION_REGISTRY:
        register_relation("nqueens_not_equal", nqueens_not_equal)
    if "nqueens_not_diagonal" not in RELATION_REGISTRY:
        register_relation("nqueens_not_diagonal", nqueens_not_diagonal)

def _process_independent_group_worker(args: Tuple) -> Tuple[bool, Dict[str, Set[Any]]]:
    """
    Worker function para procesar un grupo independiente de restricciones.
    
    Esta función se ejecuta en un proceso separado y procesa un subconjunto
    de restricciones que son independientes entre sí.
    
    Args:
        args: Tupla con (group_constraints_data, variables_state, last_support_state)
    
    Returns:
        Tupla (consistente, dominios_modificados)
    """
    from .ac31 import revise_with_last_support
    from .domains import SetDomain
    from .constraints import get_relation

    group_constraints_data, variables_state, last_support_state = args
    
    # Reconstruir el estado local del ArcEngine (solo lo necesario para revise_with_last_support)
    class LocalArcEngine:
        def __init__(self, vars_state, ls_state):
            self.variables = {}
            self.last_support = ls_state
            self.constraints = {}
            
            # Reconstruir dominios desde sets
            for var_name, values_set in vars_state.items():
                self.variables[var_name] = SetDomain(values_set)

    local_engine = LocalArcEngine(variables_state, last_support_state)
    
    # Procesar restricciones del grupo
    modified_domains = {}
    
    for constraint_data in group_constraints_data:
        cid = constraint_data["id"]
        var1 = constraint_data["var1"]
        var2 = constraint_data["var2"]
        relation_name = constraint_data["relation_name"]
        
        # Obtener la función de relación registrada
        relation_func = get_relation(relation_name)

        # Revisar ambos arcos
        for xi, xj in [(var1, var2), (var2, var1)]:
            # Solo revisar si ambas variables existen en el motor local
            if xi in local_engine.variables and xj in local_engine.variables:
                revised, removed = revise_with_last_support(local_engine, xi, xj, cid, relation_func=relation_func)
                
                if revised:
                    # Verificar inconsistencia
                    if not local_engine.variables[xi]:
                        return False, {}
                    
                    # Registrar dominio modificado
                    modified_domains[xi] = set(local_engine.variables[xi].get_values())
    
    # Retornar dominios modificados
    return True, modified_domains


class TopologicalParallelAC3:
    """
    AC-3 paralelizado usando estructura topológica.
    
    Esta clase explota los grupos de restricciones independientes identificados
    por HomotopyRules para procesarlos en paralelo usando multiprocessing,
    evitando así el Global Interpreter Lock (GIL) de Python.
    
    Attributes:
        arc_engine: Referencia al ArcEngine
        num_workers: Número de procesos paralelos
        homotopy_rules: Reglas de homotopía precomputadas
    """
    
    def __init__(self, arc_engine, num_workers: int = None):
        """
        Inicializa el motor de paralelización topológica.
        
        Args:
            arc_engine: Instancia de ArcEngine
            num_workers: Número de workers (default: CPU count)
        """
        self.arc_engine = arc_engine
        self.num_workers = num_workers or min(cpu_count(), 4)
        self.homotopy_rules = None
        
        logger.info(f"TopologicalParallelAC3 inicializado con {self.num_workers} workers")
    
    def _precompute_homotopy_rules(self):
        """
        Precomputa las reglas de homotopía si no existen.
        """
        if self.homotopy_rules is None:
            from ..homotopy.rules import HomotopyRules
            self.homotopy_rules = HomotopyRules()
            # Pasar solo los IDs de las restricciones, no los objetos completos
            self.homotopy_rules.precompute_from_engine(self.arc_engine)

            logger.info(f"Reglas de homotopía precomputadas: "
                       f"{len(self.homotopy_rules.commutative_pairs)} pares conmutativos")
    
    def enforce_arc_consistency_topological(self) -> bool:
        """
        Ejecuta AC-3 con optimización topológica y paralelización.
        
        Returns:
            False si se encuentra una inconsistencia, True si es consistente
        """
        logger.info("Iniciando enforce_arc_consistency_topological...")
        self._precompute_homotopy_rules()
        
        independent_groups = self.homotopy_rules.get_independent_groups()
        if not independent_groups:
            logger.warning("No se encontraron grupos independientes. Ejecutando AC-3 secuencial.")
            return self._execute_optimized_sequential()

        # Preparar datos para los workers
        groups_data = self._prepare_groups_data(independent_groups)
        
        # Ejecutar en paralelo
        with Pool(processes=self.num_workers, initializer=_worker_init) as pool:
            results = pool.map(_process_independent_group_worker, groups_data)
        
        # Fusionar resultados
        inconsistent = False
        for consistent, modified_domains in results:
            if not consistent:
                inconsistent = True
                break
            for var_name, new_domain_values in modified_domains.items():
                # Intersección de dominios para fusionar los resultados de los workers
                current_domain = self.arc_engine.variables[var_name]
                current_domain.intersect(new_domain_values)
                if not current_domain:
                    inconsistent = True
                    break
            if inconsistent:
                break

        if inconsistent:
            logger.debug("Inconsistencia detectada durante la fusión de resultados paralelos.")
            return False

        # Una pasada final secuencial para propagar cualquier cambio residual
        # que pueda haber surgido de la fusión de dominios.
        return self._final_propagation()
    
    def _execute_optimized_sequential(self) -> bool:
        """
        Ejecuta AC-3 secuencial optimizado con reglas de homotopía.
        
        Returns:
            False si se encuentra inconsistencia, True si es consistente
        """
        from .ac31 import revise_with_last_support
        from .constraints import get_relation
        
        queue = deque()
        for cid, c in self.arc_engine.constraints.items():
            queue.append((c.var1, c.var2, cid))
            queue.append((c.var2, c.var1, cid))
        
        while queue:
            xi, xj, constraint_id = queue.popleft()
            
            constraint = self.arc_engine.constraints[constraint_id]
            relation_func = get_relation(constraint.relation_name)

            revised, removed_values = revise_with_last_support(
                self.arc_engine, xi, xj, constraint_id, relation_func=relation_func
            )
            
            if revised:
                if not self.arc_engine.variables[xi]:
                    return False
                
                # Añadir arcos afectados
                for neighbor in self.arc_engine.graph.neighbors(xi):
                    if neighbor != xj:
                        c_id = self.arc_engine.graph.get_edge_data(neighbor, xi)["cid"]
                        if (neighbor, xi, c_id) not in queue:
                            queue.append((neighbor, xi, c_id))
        
        return True
    
    def _prepare_groups_data(self, independent_groups: List[Set[str]]) -> List[Tuple]:
        """
        Prepara los datos para enviar a los workers.
        
        Args:
            independent_groups: Lista de grupos de constraint IDs independientes
        
        Returns:
            Lista de tuplas (group_constraints_data, variables_state, last_support_state)
        """
        groups_data = []
        
        # Serializar estado de variables (dominios)
        variables_state = {}
        for var_name, domain in self.arc_engine.variables.items():
            variables_state[var_name] = set(domain.get_values())
        
        # Serializar last_support
        last_support_state = dict(self.arc_engine.last_support)
        
        for group in independent_groups:
            # Recolectar datos de constraints del grupo
            group_constraints_data = []
            for cid in group:
                if cid in self.arc_engine.constraints:
                    constraint = self.arc_engine.constraints[cid]
                    group_constraints_data.append({
                        "id": cid,
                        "var1": constraint.var1,
                        "var2": constraint.var2,
                        "relation_name": constraint.relation_name
                    })
            
            if group_constraints_data:
                groups_data.append((
                    group_constraints_data,
                    deepcopy(variables_state),
                    deepcopy(last_support_state)
                ))
        
        return groups_data
    
    def _final_propagation(self) -> bool:
        """
        Ejecuta una pasada final de AC-3 secuencial para propagar cambios.
        
        Returns:
            False si se encuentra inconsistencia, True si es consistente
        """
        logger.info("Ejecutando propagación final secuencial")
        
        # Ejecutar AC-3 secuencial estándar
        return self._execute_optimized_sequential()
    
    def get_stats(self) -> dict:
        """
        Retorna estadísticas del procesamiento paralelo.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "num_workers": self.num_workers,
            "total_constraints": len(self.arc_engine.constraints),
            "total_variables": len(self.arc_engine.variables)
        }
        
        if self.homotopy_rules:
            independent_groups = self.homotopy_rules.get_independent_groups()
            stats["independent_groups"] = len(independent_groups)
            stats["avg_group_size"] = (
                sum(len(g) for g in independent_groups) / len(independent_groups)
                if independent_groups else 0
            )
        
        return stats
    
    def __repr__(self) -> str:
        return (f"TopologicalParallelAC3(workers={self.num_workers}, "
                f"constraints={len(self.arc_engine.constraints)})")

