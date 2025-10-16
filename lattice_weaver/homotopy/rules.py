# lattice_weaver/homotopy/rules.py

"""
Reglas de Homotopía Precomputadas

Este módulo implementa la precomputación de reglas de conmutatividad entre
restricciones para optimizar la detección de homotopías y el orden de propagación
en el motor de consistencia de arcos.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
Versión: 1.0
"""

from itertools import combinations
from typing import Dict, Tuple, Set, List, Optional, Any
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class HomotopyRules:
    """
    Reglas precomputadas para detección rápida de homotopías.
    
    Esta clase analiza el grafo de restricciones de un CSP y precomputa:
    1. Pares de restricciones que conmutan (independientes)
    2. Grupos de restricciones independientes (componentes)
    3. Grafo de dependencias para ordenamiento topológico
    
    Attributes:
        commutative_pairs: Conjunto de pares de restricciones que conmutan
        independent_groups: Lista de grupos de restricciones independientes
        dependency_graph: Grafo dirigido de dependencias entre restricciones
        _precomputed: Flag que indica si las reglas han sido precomputadas
    """
    
    def __init__(self):
        """Inicializa las estructuras de datos para las reglas de homotopía."""
        self.commutative_pairs: Set[Tuple[str, str]] = set()
        self.independent_groups: List[Set[str]] = []
        self.dependency_graph: nx.DiGraph = nx.DiGraph()
        self._precomputed = False
        self._constraint_to_vars: Dict[str, Set[str]] = {}
    
    def precompute_from_engine(self, arc_engine) -> None:
        """
        Analiza todas las restricciones del ArcEngine y precomputa reglas.
        
        Este método realiza un análisis O(k²) una sola vez al inicio para
        permitir consultas O(1) posteriores.
        
        Args:
            arc_engine: Instancia de ArcEngine con restricciones definidas
        
        Raises:
            ValueError: Si el engine no tiene restricciones
        """
        if not arc_engine.constraints:
            raise ValueError("ArcEngine no tiene restricciones definidas")
        
        logger.info(f"Iniciando precomputación de reglas de homotopía para {len(arc_engine.constraints)} restricciones")
        
        constraints = list(arc_engine.constraints.values())
        
        # Construir mapeo de restricciones a variables
        for cid, constraint in arc_engine.constraints.items():
            self._constraint_to_vars[cid] = {constraint.var1, constraint.var2}
        
        # 1. Identificar pares conmutativos
        self._compute_commutative_pairs(constraints, arc_engine)
        
        # 2. Construir grafo de dependencias
        self._build_dependency_graph(arc_engine)
        
        # 3. Identificar grupos independientes (componentes conexos)
        self._identify_independent_groups()
        
        self._precomputed = True
        
        logger.info(f"Precomputación completada: {len(self.commutative_pairs)} pares conmutativos, "
                   f"{len(self.independent_groups)} grupos independientes")
    
    def _compute_commutative_pairs(self, constraints: List[Any], arc_engine) -> None:
        """
        Identifica pares de restricciones que conmutan.
        
        Dos restricciones conmutan si son independientes, es decir, si el orden
        de su aplicación no afecta el resultado final.
        
        Args:
            constraints: Lista de objetos Constraint
            arc_engine: Instancia de ArcEngine
        """
        for c1, c2 in combinations(constraints, 2):
            if self._are_independent(c1, c2, arc_engine):
                # Usar IDs de restricciones del engine
                cid1 = self._get_constraint_id(c1, arc_engine)
                cid2 = self._get_constraint_id(c2, arc_engine)
                
                if cid1 and cid2:
                    pair = tuple(sorted([cid1, cid2]))
                    self.commutative_pairs.add(pair)
    
    def _get_constraint_id(self, constraint, arc_engine) -> Optional[str]:
        """
        Obtiene el ID de una restricción en el engine.
        
        Args:
            constraint: Objeto Constraint
            arc_engine: Instancia de ArcEngine
        
        Returns:
            ID de la restricción o None si no se encuentra
        """
        for cid, c in arc_engine.constraints.items():
            if c is constraint:
                return cid
        return None
    
    def _are_independent(self, c1, c2, engine) -> bool:
        """
        Verifica si dos restricciones son independientes.
        
        Dos restricciones son independientes si:
        1. Operan sobre variables completamente disjuntas
        2. No existe un camino de propagación directo entre ellas
        
        Args:
            c1: Primera restricción
            c2: Segunda restricción
            engine: Instancia de ArcEngine
        
        Returns:
            True si las restricciones son independientes
        """
        vars1 = {c1.var1, c1.var2}
        vars2 = {c2.var1, c2.var2}
        
        # Caso simple: variables disjuntas
        if vars1.isdisjoint(vars2):
            return True
        
        # Caso complejo: comparten variables pero pueden ser independientes
        # si no hay propagación cruzada significativa
        # Por ahora, consideramos que si comparten variables, no son independientes
        # Esta heurística puede refinarse en futuras versiones
        
        return False
    
    def _build_dependency_graph(self, arc_engine) -> None:
        """
        Construye el grafo de dependencias entre restricciones.
        
        Una arista (c1, c2) indica que c1 puede afectar a c2 durante la propagación.
        
        Args:
            arc_engine: Instancia de ArcEngine
        """
        # Añadir nodos para cada restricción
        for cid in arc_engine.constraints.keys():
            self.dependency_graph.add_node(cid)
        
        # Añadir aristas basadas en variables compartidas
        for cid1, c1 in arc_engine.constraints.items():
            for cid2, c2 in arc_engine.constraints.items():
                if cid1 != cid2:
                    if self._share_variables(c1, c2):
                        # Existe dependencia: c1 puede afectar a c2
                        self.dependency_graph.add_edge(cid1, cid2)
    
    def _share_variables(self, c1, c2) -> bool:
        """
        Verifica si dos restricciones comparten al menos una variable.
        
        Args:
            c1: Primera restricción
            c2: Segunda restricción
        
        Returns:
            True si comparten variables
        """
        vars1 = {c1.var1, c1.var2}
        vars2 = {c2.var1, c2.var2}
        return not vars1.isdisjoint(vars2)
    
    def _identify_independent_groups(self) -> None:
        """
        Identifica grupos de restricciones independientes.
        
        Usa componentes conexos del grafo no dirigido para encontrar grupos
        de restricciones que pueden procesarse en paralelo.
        """
        # Convertir a grafo no dirigido para encontrar componentes
        undirected = self.dependency_graph.to_undirected()
        
        # Obtener componentes conexos
        self.independent_groups = [
            set(component) 
            for component in nx.connected_components(undirected)
        ]
    
    def is_commutative(self, cid1: str, cid2: str) -> bool:
        """
        Verifica si dos restricciones conmutan (consulta O(1)).
        
        Args:
            cid1: ID de la primera restricción
            cid2: ID de la segunda restricción
        
        Returns:
            True si las restricciones conmutan
        
        Raises:
            RuntimeError: Si las reglas no han sido precomputadas
        """
        if not self._precomputed:
            raise RuntimeError("Rules not precomputed. Call precompute_from_engine first.")
        
        pair = tuple(sorted([cid1, cid2]))
        return pair in self.commutative_pairs
    
    def get_independent_groups(self) -> List[Set[str]]:
        """
        Retorna grupos de restricciones independientes.
        
        Estos grupos pueden procesarse en paralelo sin interferencia mutua.
        
        Returns:
            Lista de conjuntos de IDs de restricciones
        
        Raises:
            RuntimeError: Si las reglas no han sido precomputadas
        """
        if not self._precomputed:
            raise RuntimeError("Rules not precomputed. Call precompute_from_engine first.")
        
        return self.independent_groups
    
    def get_optimal_propagation_order(self) -> List[str]:
        """
        Calcula el orden óptimo de propagación basado en el grafo de dependencias.
        
        Usa ordenamiento topológico para minimizar el reprocesamiento de restricciones.
        Si el grafo contiene ciclos, usa una heurística basada en el grado de entrada.
        
        Returns:
            Lista ordenada de IDs de restricciones
        
        Raises:
            RuntimeError: Si las reglas no han sido precomputadas
        """
        if not self._precomputed:
            raise RuntimeError("Rules not precomputed. Call precompute_from_engine first.")
        
        # Verificar si el grafo es acíclico
        if nx.is_directed_acyclic_graph(self.dependency_graph):
            # Usar ordenamiento topológico
            return list(nx.topological_sort(self.dependency_graph))
        else:
            # El grafo tiene ciclos, usar heurística alternativa
            # Ordenar por grado de entrada (restricciones con menos dependencias primero)
            logger.warning("Grafo de dependencias contiene ciclos, usando heurística de grado de entrada")
            
            in_degrees = dict(self.dependency_graph.in_degree())
            sorted_nodes = sorted(in_degrees.items(), key=lambda x: x[1])
            return [node for node, _ in sorted_nodes]
    
    def get_constraint_variables(self, cid: str) -> Set[str]:
        """
        Retorna las variables involucradas en una restricción.
        
        Args:
            cid: ID de la restricción
        
        Returns:
            Conjunto de nombres de variables
        
        Raises:
            KeyError: Si la restricción no existe
        """
        if cid not in self._constraint_to_vars:
            raise KeyError(f"Constraint '{cid}' not found")
        
        return self._constraint_to_vars[cid]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estadísticas sobre las reglas precomputadas.
        
        Returns:
            Diccionario con métricas de las reglas
        
        Raises:
            RuntimeError: Si las reglas no han sido precomputadas
        """
        if not self._precomputed:
            raise RuntimeError("Rules not precomputed. Call precompute_from_engine first.")
        
        return {
            'total_constraints': len(self._constraint_to_vars),
            'commutative_pairs': len(self.commutative_pairs),
            'independent_groups': len(self.independent_groups),
            'largest_group_size': max(len(g) for g in self.independent_groups) if self.independent_groups else 0,
            'has_cycles': not nx.is_directed_acyclic_graph(self.dependency_graph),
            'graph_density': nx.density(self.dependency_graph)
        }
    
    def __repr__(self) -> str:
        """Representación en string de las reglas."""
        if not self._precomputed:
            return "HomotopyRules(not precomputed)"
        
        stats = self.get_statistics()
        return (f"HomotopyRules(constraints={stats['total_constraints']}, "
                f"commutative_pairs={stats['commutative_pairs']}, "
                f"groups={stats['independent_groups']})")

