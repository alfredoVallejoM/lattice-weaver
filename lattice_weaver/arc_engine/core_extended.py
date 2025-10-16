# lattice_weaver/arc_engine/core_extended.py

"""
ArcEngine Extendido con Reglas de Homotopía

Esta es una versión extendida del ArcEngine original que integra
las reglas de homotopía precomputadas para optimizar la propagación
de restricciones.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
Versión: 1.0
"""

from typing import Iterable, Callable, Any, Optional, Dict, Tuple, Set, List
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class ArcEngineExtended:
    """
    Motor de consistencia de arcos de alto rendimiento basado en AC-3.1
    con optimizaciones de homotopía.
    
    Esta versión extiende el ArcEngine original con:
    - Precomputación de reglas de homotopía
    - Orden de propagación optimizado
    - Identificación de grupos independientes para paralelización
    """

    def __init__(self, parallel: bool = False, parallel_mode: str = 'thread', 
                 use_homotopy_rules: bool = True):
        """
        Inicializa el ArcEngine extendido.

        Args:
            parallel: Si True, habilita ejecución paralela
            parallel_mode: Tipo de paralelización ('thread', 'topological')
            use_homotopy_rules: Si True, usa reglas de homotopía precomputadas
        """
        # Importar dependencias necesarias
        from .domains import Domain
        from .constraints import Constraint
        
        self.variables: Dict[str, Any] = {}  # Domain objects
        self.constraints: Dict[str, Any] = {}  # Constraint objects
        self.graph = nx.Graph()  # Grafo de restricciones
        self.parallel = parallel
        self.parallel_mode = parallel_mode
        self.use_homotopy_rules = use_homotopy_rules
        
        # Estructura de datos para optimización AC-3.1
        self.last_support: Dict[Tuple[str, str, Any], Any] = {}
        
        # Reglas de homotopía
        self.homotopy_rules = None
        if use_homotopy_rules:
            from ..homotopy.rules import HomotopyRules
            self.homotopy_rules = HomotopyRules()
            logger.info("HomotopyRules habilitadas")

    def add_variable(self, name: str, domain: Iterable[Any]):
        """
        Añade una variable con su dominio inicial.

        Args:
            name: Nombre de la variable
            domain: Iterable de valores posibles para la variable
        
        Raises:
            ValueError: Si la variable ya existe
        """
        from .domains import create_optimal_domain
        
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists.")
        
        self.variables[name] = create_optimal_domain(domain)
        self.graph.add_node(name)
        logger.debug(f"Variable '{name}' añadida con dominio de tamaño {len(list(domain))}")

    def add_constraint(self, var1: str, var2: str, relation: Callable[[Any, Any], bool], 
                      cid: Optional[str] = None):
        """
        Añade una restricción binaria entre dos variables.

        Args:
            var1: Nombre de la primera variable
            var2: Nombre de la segunda variable
            relation: Función que retorna True si dos valores son consistentes
            cid: ID opcional para la restricción
        
        Raises:
            ValueError: Si el ID de restricción ya existe
        """
        from .constraints import Constraint
        
        if cid is None:
            cid = f"{var1}_{var2}"
        
        if cid in self.constraints:
            raise ValueError(f"Constraint ID '{cid}' already exists.")
        
        self.constraints[cid] = Constraint(var1, var2, relation)
        self.graph.add_edge(var1, var2, cid=cid)
        logger.debug(f"Restricción '{cid}' añadida entre {var1} y {var2}")

    def enforce_arc_consistency(self) -> bool:
        """
        Ejecuta consistencia de arcos usando AC-3.1 optimizado.
        
        Si use_homotopy_rules está habilitado, precomputa las reglas
        y usa un orden de propagación optimizado.

        Returns:
            False si se encuentra una inconsistencia (dominio vacío), True en caso contrario
        """
        # Precomputar reglas de homotopía si están habilitadas
        if self.use_homotopy_rules and self.homotopy_rules and self.constraints:
            logger.info("Precomputando reglas de homotopía...")
            self.homotopy_rules.precompute_from_engine(self)
            
            # Obtener estadísticas
            stats = self.homotopy_rules.get_statistics()
            logger.info(f"Reglas precomputadas: {stats}")
            
            # Usar orden optimizado
            optimal_order = self.homotopy_rules.get_optimal_propagation_order()
            logger.info(f"Usando orden optimizado de propagación con {len(optimal_order)} restricciones")
            
            return self._enforce_with_order(optimal_order)
        
        # Fallback al AC-3 estándar
        logger.info("Ejecutando AC-3 estándar (sin optimización de homotopía)")
        return self._enforce_standard()
    
    def _enforce_with_order(self, order: List[str]) -> bool:
        """
        Ejecuta AC-3 siguiendo un orden específico de restricciones.
        
        Este método usa el orden topológico calculado por HomotopyRules
        para minimizar el número de iteraciones.
        
        Args:
            order: Lista ordenada de IDs de restricciones
        
        Returns:
            False si se encuentra inconsistencia, True en caso contrario
        """
        from .ac31 import revise_with_last_support
        
        # Inicializar cola con el orden optimizado
        queue: List[Tuple[str, str, str]] = []
        
        for cid in order:
            if cid in self.constraints:
                c = self.constraints[cid]
                queue.append((c.var1, c.var2, cid))
                queue.append((c.var2, c.var1, cid))
        
        # Procesar cola
        iterations = 0
        while queue:
            xi, xj, constraint_id = queue.pop(0)
            iterations += 1

            # Núcleo del algoritmo AC-3.1
            revised, removed_values = revise_with_last_support(self, xi, xj, constraint_id)

            if revised:
                if not self.variables[xi]:
                    logger.warning(f"Inconsistencia encontrada: dominio de '{xi}' está vacío")
                    return False

                # Añadir arcos afectados de vuelta a la cola
                for neighbor in self.graph.neighbors(xi):
                    if neighbor != xj:
                        c_id = self.graph.get_edge_data(neighbor, xi)['cid']
                        queue.append((neighbor, xi, c_id))
        
        logger.info(f"AC-3 completado en {iterations} iteraciones")
        return True
    
    def _enforce_standard(self) -> bool:
        """
        Ejecuta el algoritmo AC-3.1 estándar sin optimizaciones.
        
        Returns:
            False si se encuentra inconsistencia, True en caso contrario
        """
        from .ac31 import revise_with_last_support
        
        # Usar versión paralela si está habilitada
        if self.parallel:
            if self.parallel_mode == 'topological':
                from .topological_parallel import TopologicalParallelAC3
                topological_ac3 = TopologicalParallelAC3(self)
                return topological_ac3.enforce_arc_consistency_topological()
            else:  # 'thread' mode
                from .parallel_ac3 import ParallelAC3
                parallel_ac3 = ParallelAC3(self)
                return parallel_ac3.enforce_arc_consistency_parallel()
        
        # Algoritmo AC-3.1 secuencial
        queue: List[Tuple[str, str, str]] = []
        for cid, c in self.constraints.items():
            queue.append((c.var1, c.var2, cid))
            queue.append((c.var2, c.var1, cid))

        iterations = 0
        while queue:
            xi, xj, constraint_id = queue.pop(0)
            iterations += 1

            revised, removed_values = revise_with_last_support(self, xi, xj, constraint_id)

            if revised:
                if not self.variables[xi]:
                    logger.warning(f"Inconsistencia encontrada: dominio de '{xi}' está vacío")
                    return False

                for neighbor in self.graph.neighbors(xi):
                    if neighbor != xj:
                        c_id = self.graph.get_edge_data(neighbor, xi)['cid']
                        queue.append((neighbor, xi, c_id))
        
        logger.info(f"AC-3 estándar completado en {iterations} iteraciones")
        return True

    def get_independent_groups(self) -> List[Set[str]]:
        """
        Retorna grupos de restricciones independientes.
        
        Estos grupos pueden procesarse en paralelo sin interferencia.
        
        Returns:
            Lista de conjuntos de IDs de restricciones
        
        Raises:
            RuntimeError: Si las reglas de homotopía no están habilitadas o precomputadas
        """
        if not self.use_homotopy_rules or not self.homotopy_rules:
            raise RuntimeError("Homotopy rules not enabled")
        
        return self.homotopy_rules.get_independent_groups()
    
    def get_homotopy_statistics(self) -> Dict[str, Any]:
        """
        Retorna estadísticas sobre las reglas de homotopía.
        
        Returns:
            Diccionario con métricas
        
        Raises:
            RuntimeError: Si las reglas de homotopía no están habilitadas
        """
        if not self.use_homotopy_rules or not self.homotopy_rules:
            raise RuntimeError("Homotopy rules not enabled")
        
        return self.homotopy_rules.get_statistics()
    
    def build_consistency_graph(self) -> nx.Graph:
        """
        Construye el grafo de consistencia (micro-estructura) del CSP.
        
        Los nodos son pares (variable, valor), las aristas conectan asignaciones consistentes.
        
        (Pendiente de implementación en Fase 3)
        """
        raise NotImplementedError("build_consistency_graph will be implemented in Phase 3.")

    def analyze_simplicial_topology(self, concept_lattice: Optional[Any] = None) -> Dict[str, int]:
        """
        Realiza análisis topológico en el grafo de consistencia.
        
        (Pendiente de implementación en Fase 3)
        """
        raise NotImplementedError("analyze_simplicial_topology will be implemented in Phase 3.")

    def remove_constraint(self, constraint_id: str):
        """
        Elimina una restricción y restaura eficientemente la consistencia usando TMS.
        
        (Pendiente de implementación en Fase 5)
        """
        raise NotImplementedError("remove_constraint requires a Truth Maintenance System (Phase 5).")

    def __repr__(self):
        homotopy_status = "enabled" if self.use_homotopy_rules else "disabled"
        return (f"ArcEngineExtended(variables={len(self.variables)}, "
                f"constraints={len(self.constraints)}, "
                f"homotopy={homotopy_status})")

