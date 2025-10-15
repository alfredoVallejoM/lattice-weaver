"""
topology_guided.py: Estrategias de selección guiadas por análisis topológico.

Este módulo implementa estrategias que usan análisis topológico del grafo de
consistencia para guiar la selección de variables.

Autor: Manus AI
Fecha: 15 de Octubre de 2025
"""

from typing import Dict, List, Any, Optional
from .base import VariableSelector
from lattice_weaver.core.csp_problem import CSP
from ..topology_adapter import CSPTopologyAdapter


class TopologyGuidedSelector(VariableSelector):
    """
    Selector que usa análisis topológico para guiar la selección de variables.
    
    Esta estrategia:
    1. Construye el grafo de consistencia del CSP
    2. Identifica nodos críticos (alta centralidad de intermediación)
    3. Prioriza variables que aparecen en nodos críticos
    4. Usa MRV como desempate
    
    La idea es que variables en nodos críticos tienen mayor impacto en la
    estructura del espacio de soluciones.
    """
    
    def __init__(self):
        """Inicializa el selector topológico."""
        self._topology_cache = {}
        self._critical_vars_cache = {}
    
    def select(self, csp: CSP, assignment: Dict[str, Any], current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """
        Selecciona variable usando análisis topológico + MRV.
        
        Args:
            csp: Problema CSP
            assignment: Asignación parcial actual
            current_domains: Dominios actuales de variables
        
        Returns:
            Variable a asignar, o None si todas están asignadas
        """
        unassigned_vars = [v for v in csp.variables if v not in assignment]
        if not unassigned_vars:
            return None
        
        # Si solo queda una variable, retornarla directamente
        if len(unassigned_vars) == 1:
            return unassigned_vars[0]
        
        # Obtener prioridades topológicas (con caché)
        csp_id = id(csp)
        if csp_id not in self._critical_vars_cache:
            self._compute_critical_variables(csp)
        
        critical_vars = self._critical_vars_cache.get(csp_id, {})
        
        # Calcular score para cada variable no asignada
        # Score = (prioridad_topológica, -tamaño_dominio)
        # Mayor prioridad topológica primero, luego menor dominio (MRV)
        scores = []
        for var in unassigned_vars:
            topology_priority = critical_vars.get(var, 0.0)
            domain_size = len(current_domains[var])
            scores.append((var, (topology_priority, -domain_size)))
        
        # Ordenar por score descendente y retornar la mejor
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]
    
    def _compute_critical_variables(self, csp: CSP):
        """
        Calcula prioridades de variables basadas en análisis topológico.
        
        Args:
            csp: Problema CSP
        """
        csp_id = id(csp)
        
        # Construir adaptador y grafo de consistencia
        adapter = CSPTopologyAdapter(csp)
        adapter.build_consistency_graph()
        
        # Encontrar nodos críticos
        critical_nodes = adapter.find_critical_nodes(top_k=20)
        
        # Agregar centralidad por variable
        var_priorities = {}
        for (var, _val), centrality in critical_nodes:
            if var not in var_priorities:
                var_priorities[var] = 0.0
            var_priorities[var] += centrality
        
        # Normalizar prioridades
        if var_priorities:
            max_priority = max(var_priorities.values())
            if max_priority > 0:
                var_priorities = {v: p / max_priority for v, p in var_priorities.items()}
        
        # Cachear resultados
        self._topology_cache[csp_id] = adapter
        self._critical_vars_cache[csp_id] = var_priorities
    
    def reset_cache(self):
        """Limpia el caché de análisis topológico."""
        self._topology_cache.clear()
        self._critical_vars_cache.clear()


class ComponentBasedSelector(VariableSelector):
    """
    Selector que procesa componentes conexas del grafo de consistencia.
    
    Esta estrategia:
    1. Identifica componentes conexas del grafo de consistencia
    2. Procesa una componente a la vez (la más pequeña primero)
    3. Dentro de cada componente, usa MRV
    
    La idea es resolver componentes independientes por separado, reduciendo
    la complejidad del problema.
    """
    
    def __init__(self):
        """Inicializa el selector basado en componentes."""
        self._topology_cache = {}
        self._components_cache = {}
        self._current_component_idx = {}
    
    def select(self, csp: CSP, assignment: Dict[str, Any], current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """
        Selecciona variable procesando componentes conexas.
        
        Args:
            csp: Problema CSP
            assignment: Asignación parcial actual
            current_domains: Dominios actuales de variables
        
        Returns:
            Variable a asignar, o None si todas están asignadas
        """
        unassigned_vars = [v for v in csp.variables if v not in assignment]
        if not unassigned_vars:
            return None
        
        # Si solo queda una variable, retornarla directamente
        if len(unassigned_vars) == 1:
            return unassigned_vars[0]
        
        # Obtener componentes (con caché)
        csp_id = id(csp)
        if csp_id not in self._components_cache:
            self._compute_components(csp)
        
        components = self._components_cache.get(csp_id, [])
        
        # Agrupar variables no asignadas por componente
        var_to_component = {}
        for comp_idx, component in enumerate(components):
            for (var, _val) in component:
                if var in unassigned_vars:
                    var_to_component[var] = comp_idx
        
        # Si no hay componentes o todas las variables están en la misma,
        # usar MRV simple
        if not var_to_component or len(set(var_to_component.values())) == 1:
            return min(unassigned_vars, key=lambda v: len(current_domains[v]))
        
        # Seleccionar la componente más pequeña con variables no asignadas
        component_sizes = {}
        for var, comp_idx in var_to_component.items():
            if comp_idx not in component_sizes:
                component_sizes[comp_idx] = 0
            component_sizes[comp_idx] += 1
        
        smallest_comp = min(component_sizes.keys(), key=lambda c: component_sizes[c])
        
        # Dentro de la componente más pequeña, usar MRV
        vars_in_smallest = [v for v, c in var_to_component.items() if c == smallest_comp]
        return min(vars_in_smallest, key=lambda v: len(current_domains[v]))
    
    def _compute_components(self, csp: CSP):
        """
        Calcula componentes conexas del grafo de consistencia.
        
        Args:
            csp: Problema CSP
        """
        csp_id = id(csp)
        
        # Construir adaptador y grafo de consistencia
        adapter = CSPTopologyAdapter(csp)
        adapter.build_consistency_graph()
        
        # Encontrar componentes conexas
        components = adapter.find_connected_components()
        
        # Ordenar componentes por tamaño (más pequeñas primero)
        components = sorted(components, key=len)
        
        # Cachear resultados
        self._topology_cache[csp_id] = adapter
        self._components_cache[csp_id] = components
        self._current_component_idx[csp_id] = 0
    
    def reset_cache(self):
        """Limpia el caché de componentes."""
        self._topology_cache.clear()
        self._components_cache.clear()
        self._current_component_idx.clear()

