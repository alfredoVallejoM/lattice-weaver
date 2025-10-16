"""
AC-3 con Multiprocessing Real

Implementa AC-3 usando multiprocessing para eludir el GIL de Python
y lograr paralelización real.

Requisitos:
- Restricciones serializables (no lambdas)
- Dominios serializables

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from typing import Dict, List, Tuple, Set, Any
from multiprocessing import Pool, Manager, cpu_count
import logging

logger = logging.getLogger(__name__)


def revise_arc_worker(args: Tuple) -> Tuple[str, str, str, bool, List[Any]]:
    """
    Worker function para revisar un arco en paralelo.
    
    Args:
        args: (xi, xj, constraint_id, domain_xi, domain_xj, constraint)
    
    Returns:
        (xi, xj, constraint_id, revised, removed_values)
    """
    xi, xj, constraint_id, domain_xi, domain_xj, constraint = args
    
    revised = False
    removed_values = []
    
    # Convertir a sets para eficiencia
    domain_xi_set = set(domain_xi)
    domain_xj_set = set(domain_xj)
    
    for val_i in list(domain_xi_set):
        # Buscar soporte en domain_xj
        has_support = False
        
        for val_j in domain_xj_set:
            if constraint.check(val_i, val_j):
                has_support = True
                break
        
        if not has_support:
            domain_xi_set.remove(val_i)
            removed_values.append(val_i)
            revised = True
    
    return (xi, xj, constraint_id, revised, removed_values, list(domain_xi_set))


class MultiprocessAC3:
    """
    AC-3 con multiprocessing real.
    
    Paraleliza la revisión de arcos usando multiprocessing.Pool.
    """
    
    def __init__(self, arc_engine, num_workers: int = None):
        """
        Inicializa MultiprocessAC3.
        
        Args:
            arc_engine: Instancia de ArcEngine
            num_workers: Número de workers (default: cpu_count())
        """
        self.engine = arc_engine
        self.num_workers = num_workers or cpu_count()
        
        logger.info(f"MultiprocessAC3 inicializado con {self.num_workers} workers")
    
    def enforce_arc_consistency_multiprocess(self) -> bool:
        """
        Ejecuta AC-3 con multiprocessing.
        
        Returns:
            True si el CSP es consistente, False si es inconsistente
        """
        # Verificar que todas las restricciones sean serializables
        if not self._check_serializability():
            logger.warning("Restricciones no serializables detectadas, usando AC-3 secuencial")
            return self.engine.enforce_arc_consistency()
        
        # Crear cola de arcos
        queue = []
        for cid, c in self.engine.constraints.items():
            queue.append((c.var1, c.var2, cid))
            queue.append((c.var2, c.var1, cid))
        
        iteration = 0
        
        while queue:
            iteration += 1
            logger.debug(f"Iteración {iteration}, arcos en cola: {len(queue)}")
            
            # Preparar trabajos para workers
            jobs = []
            for xi, xj, cid in queue:
                constraint = self.engine.constraints[cid]
                
                # Obtener dominios actuales
                domain_xi = list(self.engine.variables[xi].get_values())
                domain_xj = list(self.engine.variables[xj].get_values())
                
                # Determinar qué restricción usar
                if constraint.var1 == xi:
                    relation = constraint.relation
                else:
                    # Invertir restricción
                    relation = self._invert_constraint(constraint.relation)
                
                jobs.append((xi, xj, cid, domain_xi, domain_xj, relation))
            
            # Ejecutar en paralelo
            with Pool(processes=self.num_workers) as pool:
                results = pool.map(revise_arc_worker, jobs)
            
            # Procesar resultados
            queue = []
            
            for xi, xj, cid, revised, removed_values, new_domain in results:
                if revised:
                    # Actualizar dominio
                    self.engine.variables[xi]._values = set(new_domain)
                    
                    # Registrar en TMS si está habilitado
                    if self.engine.use_tms and self.engine.tms and removed_values:
                        for removed_val in removed_values:
                            self.engine.tms.record_removal(
                                variable=xi,
                                value=removed_val,
                                constraint_id=cid,
                                supporting_values={xj: list(self.engine.variables[xj].get_values())}
                            )
                    
                    # Verificar inconsistencia
                    if not self.engine.variables[xi]:
                        if self.engine.use_tms and self.engine.tms:
                            explanations = self.engine.tms.explain_inconsistency(xi)
                            suggested = self.engine.tms.suggest_constraint_to_relax(xi)
                            if suggested:
                                logger.warning(f"Sugerencia: relajar restricción '{suggested}'")
                        
                        return False
                    
                    # Agregar arcos afectados
                    for neighbor in self.engine.graph.neighbors(xi):
                        if neighbor != xj:
                            c_id = self.engine.graph.get_edge_data(neighbor, xi)['cid']
                            queue.append((neighbor, xi, c_id))
        
        logger.info(f"AC-3 multiprocess completado en {iteration} iteraciones")
        return True
    
    def _check_serializability(self) -> bool:
        """
        Verifica que todas las restricciones sean serializables.
        
        Returns:
            True si todas son serializables
        """
        from .serializable_constraints import SerializableConstraint
        
        for cid, constraint in self.engine.constraints.items():
            if not isinstance(constraint.relation, SerializableConstraint):
                logger.warning(f"Restricción {cid} no es serializable")
                return False
        
        return True
    
    def _invert_constraint(self, constraint):
        """
        Invierte una restricción (swap argumentos).
        
        Args:
            constraint: Restricción original
        
        Returns:
            Restricción invertida
        """
        from .serializable_constraints import SerializableConstraint
        
        if isinstance(constraint, SerializableConstraint):
            class InvertedConstraint(SerializableConstraint):
                def __init__(self, original):
                    self.original = original
                
                def check(self, val1: Any, val2: Any) -> bool:
                    return self.original.check(val2, val1)
                
                def __repr__(self) -> str:
                    return f"Inverted({self.original})"
            
            return InvertedConstraint(constraint)
        else:
            # Fallback para lambdas
            return lambda x, y: constraint(y, x)


class GroupParallelAC3:
    """
    AC-3 con paralelización por grupos independientes.
    
    Identifica grupos de restricciones independientes y los procesa
    en paralelo usando multiprocessing.
    """
    
    def __init__(self, arc_engine, num_workers: int = None):
        """
        Inicializa GroupParallelAC3.
        
        Args:
            arc_engine: Instancia de ArcEngine
            num_workers: Número de workers (default: cpu_count())
        """
        self.engine = arc_engine
        self.num_workers = num_workers or cpu_count()
        
        logger.info(f"GroupParallelAC3 inicializado con {self.num_workers} workers")
    
    def enforce_arc_consistency_groups(self) -> bool:
        """
        Ejecuta AC-3 paralelizando grupos independientes.
        
        Returns:
            True si el CSP es consistente
        """
        # Identificar grupos independientes
        groups = self._identify_independent_groups()
        
        logger.info(f"Identificados {len(groups)} grupos independientes")
        
        if len(groups) <= 1:
            # No hay paralelización posible
            logger.warning("Solo 1 grupo, usando AC-3 secuencial")
            return self.engine.enforce_arc_consistency()
        
        # Procesar grupos en paralelo
        with Pool(processes=min(self.num_workers, len(groups))) as pool:
            results = pool.map(self._process_group, groups)
        
        # Verificar resultados
        return all(results)
    
    def _identify_independent_groups(self) -> List[Set[str]]:
        """
        Identifica grupos de variables independientes.
        
        Returns:
            Lista de grupos (cada grupo es un set de variables)
        """
        import networkx as nx
        
        # Obtener componentes conexas del grafo de restricciones
        components = list(nx.connected_components(self.engine.graph))
        
        return components
    
    def _process_group(self, group: Set[str]) -> bool:
        """
        Procesa un grupo de variables independiente.
        
        Args:
            group: Set de variables del grupo
        
        Returns:
            True si el grupo es consistente
        """
        # Crear sub-engine para el grupo
        from .core import ArcEngine
        
        sub_engine = ArcEngine()
        
        # Agregar variables del grupo
        for var in group:
            domain = list(self.engine.variables[var].get_values())
            sub_engine.add_variable(var, domain)
        
        # Agregar restricciones del grupo
        for cid, constraint in self.engine.constraints.items():
            if constraint.var1 in group and constraint.var2 in group:
                sub_engine.add_constraint(
                    constraint.var1,
                    constraint.var2,
                    constraint.relation,
                    cid=cid
                )
        
        # Ejecutar AC-3 en el sub-engine
        consistent = sub_engine.enforce_arc_consistency()
        
        if consistent:
            # Actualizar dominios en el engine principal
            for var in group:
                self.engine.variables[var]._values = sub_engine.variables[var]._values
        
        return consistent


def create_multiprocess_ac3(arc_engine, num_workers: int = None) -> MultiprocessAC3:
    """
    Crea una instancia de MultiprocessAC3.
    
    Args:
        arc_engine: Instancia de ArcEngine
        num_workers: Número de workers
    
    Returns:
        Instancia de MultiprocessAC3
    """
    return MultiprocessAC3(arc_engine, num_workers)


def create_group_parallel_ac3(arc_engine, num_workers: int = None) -> GroupParallelAC3:
    """
    Crea una instancia de GroupParallelAC3.
    
    Args:
        arc_engine: Instancia de ArcEngine
        num_workers: Número de workers
    
    Returns:
        Instancia de GroupParallelAC3
    """
    return GroupParallelAC3(arc_engine, num_workers)

