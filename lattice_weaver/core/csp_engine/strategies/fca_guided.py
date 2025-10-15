"""
Estrategia FCA-Guided: Selección de Variables Guiada por Análisis FCA

Esta estrategia usa Formal Concept Analysis para guiar la selección de variables,
aprovechando el análisis estructural del CSP.

Autor: Manus AI
Fecha: 15 de Octubre, 2025
"""

from typing import Dict, List, Any, Optional
from .base import VariableSelector
from ...csp_problem import CSP
from ..fca_analyzer import FCAAnalyzer


class FCAGuidedSelector(VariableSelector):
    """
    Selector de variables guiado por análisis FCA.
    
    Esta estrategia combina:
    1. Análisis FCA para detectar estructura del problema
    2. MRV (Minimum Remaining Values) como heurística base
    3. Priorización basada en propiedades estructurales
    
    El análisis FCA se realiza una vez al inicio y se cachea para eficiencia.
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Inicializa el selector FCA-guided.
        
        Args:
            use_cache: Si True, cachea el análisis FCA para reutilizarlo
        """
        self.use_cache = use_cache
        self._analyzer_cache: Optional[FCAAnalyzer] = None
        self._priority_cache: Optional[Dict[str, float]] = None
    
    def select(self, 
               csp: CSP, 
               assignment: Dict[str, Any], 
               current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """
        Selecciona la siguiente variable a asignar usando análisis FCA.
        
        El proceso es:
        1. Filtrar variables no asignadas
        2. Si es la primera llamada, realizar análisis FCA
        3. Combinar MRV con prioridades FCA
        4. Seleccionar la variable con mayor score
        
        Args:
            csp: Problema CSP
            assignment: Asignación actual
            current_domains: Dominios actuales de las variables
        
        Returns:
            Nombre de la variable a asignar, o None si todas están asignadas
        """
        # Filtrar variables no asignadas
        unassigned = [var for var in csp.variables if var not in assignment]
        
        if not unassigned:
            return None
        
        # Si solo queda una variable, retornarla directamente
        if len(unassigned) == 1:
            return unassigned[0]
        
        # Realizar análisis FCA si no está cacheado
        if self._analyzer_cache is None or not self.use_cache:
            self._analyzer_cache = FCAAnalyzer(csp)
            self._analyzer_cache.analyze()
            self._priority_cache = self._analyzer_cache._analysis_cache['variable_priorities']
        
        # Calcular scores para cada variable no asignada
        scores = {}
        for var in unassigned:
            # Componente 1: MRV (menor dominio = mayor score)
            domain_size = len(current_domains[var])
            mrv_score = 1000.0 / max(domain_size, 1)
            
            # Componente 2: Prioridad FCA (de análisis estructural)
            fca_priority = self._priority_cache.get(var, 0.0)
            
            # Combinar ambos componentes (70% MRV, 30% FCA)
            # MRV es más importante para eficiencia inmediata
            # FCA aporta conocimiento estructural a largo plazo
            total_score = 0.7 * mrv_score + 0.3 * fca_priority
            
            scores[var] = total_score
        
        # Seleccionar variable con mayor score
        best_var = max(scores.items(), key=lambda x: x[1])[0]
        
        return best_var
    
    def reset_cache(self):
        """Limpia el cache del análisis FCA."""
        self._analyzer_cache = None
        self._priority_cache = None


class FCAOnlySelector(VariableSelector):
    """
    Selector que usa solo las prioridades FCA (sin MRV).
    
    Útil para comparar el impacto puro del análisis FCA.
    """
    
    def __init__(self):
        """Inicializa el selector FCA-only."""
        self._analyzer_cache: Optional[FCAAnalyzer] = None
        self._priority_cache: Optional[Dict[str, float]] = None
    
    def select(self, 
               csp: CSP, 
               assignment: Dict[str, Any], 
               current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """
        Selecciona la siguiente variable usando solo prioridades FCA.
        
        Args:
            csp: Problema CSP
            assignment: Asignación actual
            current_domains: Dominios actuales de las variables
        
        Returns:
            Nombre de la variable a asignar, o None si todas están asignadas
        """
        # Filtrar variables no asignadas
        unassigned = [var for var in csp.variables if var not in assignment]
        
        if not unassigned:
            return None
        
        if len(unassigned) == 1:
            return unassigned[0]
        
        # Realizar análisis FCA si no está cacheado
        if self._analyzer_cache is None:
            self._analyzer_cache = FCAAnalyzer(csp)
            self._analyzer_cache.analyze()
            self._priority_cache = self._analyzer_cache._analysis_cache['variable_priorities']
        
        # Seleccionar variable con mayor prioridad FCA
        best_var = max(unassigned, key=lambda v: self._priority_cache.get(v, 0.0))
        
        return best_var


class FCAClusterSelector(VariableSelector):
    """
    Selector que usa clusters FCA para agrupar variables similares.
    
    Selecciona variables de un cluster antes de pasar al siguiente,
    aprovechando la similaridad estructural.
    """
    
    def __init__(self):
        """Inicializa el selector FCA-cluster."""
        self._analyzer_cache: Optional[FCAAnalyzer] = None
        self._clusters: Optional[List] = None
        self._current_cluster_idx: int = 0
    
    def select(self, 
               csp: CSP, 
               assignment: Dict[str, Any], 
               current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """
        Selecciona la siguiente variable usando clusters FCA.
        
        Args:
            csp: Problema CSP
            assignment: Asignación actual
            current_domains: Dominios actuales de las variables
        
        Returns:
            Nombre de la variable a asignar, o None si todas están asignadas
        """
        # Filtrar variables no asignadas
        unassigned = [var for var in csp.variables if var not in assignment]
        
        if not unassigned:
            return None
        
        if len(unassigned) == 1:
            return unassigned[0]
        
        # Realizar análisis FCA si no está cacheado
        if self._analyzer_cache is None:
            self._analyzer_cache = FCAAnalyzer(csp)
            self._analyzer_cache.analyze()
            self._clusters = self._analyzer_cache.get_variable_clusters()
        
        # Intentar seleccionar del cluster actual
        if self._clusters and self._current_cluster_idx < len(self._clusters):
            current_cluster = self._clusters[self._current_cluster_idx]
            cluster_unassigned = [v for v in current_cluster if v in unassigned]
            
            if cluster_unassigned:
                # Usar MRV dentro del cluster
                best_var = min(cluster_unassigned, 
                             key=lambda v: len(current_domains[v]))
                return best_var
            else:
                # Cluster completado, pasar al siguiente
                self._current_cluster_idx += 1
                return self.select(csp, assignment, current_domains)
        
        # Si no hay más clusters, usar MRV simple
        return min(unassigned, key=lambda v: len(current_domains[v]))

