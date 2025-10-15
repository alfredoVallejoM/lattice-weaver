"""
hybrid_multiescala.py: Estrategias híbridas que combinan FCA + Topología.

Este módulo implementa estrategias de análisis multiescala que combinan
Formal Concept Analysis (FCA) con análisis topológico para guiar la selección
de variables de manera más sofisticada.

Autor: Manus AI
Fecha: 15 de Octubre de 2025
"""

from typing import Dict, List, Any, Optional
from .base import VariableSelector
from lattice_weaver.core.csp_problem import CSP
from ..fca_adapter import CSPToFCAAdapter
from ..fca_analyzer import FCAAnalyzer
from ..topology_adapter import CSPTopologyAdapter


class HybridFCATopologySelector(VariableSelector):
    """
    Selector híbrido que combina FCA y análisis topológico.
    
    Esta estrategia multiescala:
    1. Usa FCA para identificar clusters y prioridades estructurales
    2. Usa topología para identificar nodos críticos
    3. Combina ambos análisis para seleccionar la variable óptima
    4. Usa MRV como desempate final
    
    La combinación permite aprovechar:
    - FCA: Estructura conceptual y agrupamiento lógico
    - Topología: Conectividad y puntos críticos del espacio
    - MRV: Heurística clásica probada
    """
    
    def __init__(self, fca_weight: float = 0.5, topology_weight: float = 0.5):
        """
        Inicializa el selector híbrido.
        
        Args:
            fca_weight: Peso del análisis FCA (0-1)
            topology_weight: Peso del análisis topológico (0-1)
        """
        self.fca_weight = fca_weight
        self.topology_weight = topology_weight
        
        # Normalizar pesos
        total = fca_weight + topology_weight
        if total > 0:
            self.fca_weight /= total
            self.topology_weight /= total
        
        # Cachés
        self._fca_cache = {}
        self._topology_cache = {}
        self._fca_priorities = {}
        self._topology_priorities = {}
    
    def select(self, csp: CSP, assignment: Dict[str, Any], current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """
        Selecciona variable usando análisis híbrido FCA + Topología + MRV.
        
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
        
        # Obtener prioridades FCA y topológicas (con caché)
        csp_id = id(csp)
        if csp_id not in self._fca_priorities:
            self._compute_fca_priorities(csp)
        if csp_id not in self._topology_priorities:
            self._compute_topology_priorities(csp)
        
        fca_prio = self._fca_priorities.get(csp_id, {})
        topo_prio = self._topology_priorities.get(csp_id, {})
        
        # Calcular score híbrido para cada variable no asignada
        # Score = (score_híbrido, -tamaño_dominio)
        # Mayor score híbrido primero, luego menor dominio (MRV)
        scores = []
        for var in unassigned_vars:
            fca_score = fca_prio.get(var, 0.0)
            topo_score = topo_prio.get(var, 0.0)
            
            # Combinar scores con pesos
            hybrid_score = (self.fca_weight * fca_score + 
                           self.topology_weight * topo_score)
            
            domain_size = len(current_domains[var])
            scores.append((var, (hybrid_score, -domain_size)))
        
        # Ordenar por score descendente y retornar la mejor
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]
    
    def _compute_fca_priorities(self, csp: CSP):
        """
        Calcula prioridades de variables basadas en FCA.
        
        Args:
            csp: Problema CSP
        """
        csp_id = id(csp)
        
        try:
            # Construir adaptador FCA y analizar
            fca_adapter = CSPToFCAAdapter(csp)
            analyzer = FCAAnalyzer(fca_adapter)
            analysis = analyzer.analyze()
            
            # Obtener prioridades de variables
            priorities = analysis.get('priorities', {})
            
            # Normalizar prioridades
            if priorities:
                max_prio = max(priorities.values())
                if max_prio > 0:
                    priorities = {v: p / max_prio for v, p in priorities.items()}
            
            # Cachear resultados
            self._fca_cache[csp_id] = analyzer
            self._fca_priorities[csp_id] = priorities
        except Exception as e:
            # Si FCA falla, usar prioridades uniformes
            self._fca_priorities[csp_id] = {v: 0.5 for v in csp.variables}
    
    def _compute_topology_priorities(self, csp: CSP):
        """
        Calcula prioridades de variables basadas en análisis topológico.
        
        Args:
            csp: Problema CSP
        """
        csp_id = id(csp)
        
        try:
            # Construir adaptador topológico
            topo_adapter = CSPTopologyAdapter(csp)
            topo_adapter.build_consistency_graph()
            
            # Encontrar nodos críticos
            critical_nodes = topo_adapter.find_critical_nodes(top_k=20)
            
            # Agregar centralidad por variable
            var_priorities = {}
            for (var, _val), centrality in critical_nodes:
                if var not in var_priorities:
                    var_priorities[var] = 0.0
                var_priorities[var] += centrality
            
            # Normalizar prioridades
            if var_priorities:
                max_prio = max(var_priorities.values())
                if max_prio > 0:
                    var_priorities = {v: p / max_prio for v, p in var_priorities.items()}
            
            # Cachear resultados
            self._topology_cache[csp_id] = topo_adapter
            self._topology_priorities[csp_id] = var_priorities
        except Exception as e:
            # Si topología falla, usar prioridades uniformes
            self._topology_priorities[csp_id] = {v: 0.5 for v in csp.variables}
    
    def reset_cache(self):
        """Limpia todos los cachés."""
        self._fca_cache.clear()
        self._topology_cache.clear()
        self._fca_priorities.clear()
        self._topology_priorities.clear()


class AdaptiveMultiscaleSelector(VariableSelector):
    """
    Selector adaptativo que ajusta dinámicamente los pesos de FCA y Topología.
    
    Esta estrategia:
    1. Comienza con pesos equilibrados (50% FCA, 50% Topología)
    2. Monitorea el progreso de la búsqueda
    3. Ajusta pesos dinámicamente según efectividad
    4. Converge a la estrategia más efectiva para el problema
    
    La adaptación permite que el solver aprenda qué análisis es más útil
    para el problema específico que está resolviendo.
    """
    
    def __init__(self, initial_fca_weight: float = 0.5):
        """
        Inicializa el selector adaptativo.
        
        Args:
            initial_fca_weight: Peso inicial de FCA (0-1)
        """
        self.fca_weight = initial_fca_weight
        self.topology_weight = 1.0 - initial_fca_weight
        
        # Métricas de efectividad
        self._fca_successes = 0
        self._topology_successes = 0
        self._total_selections = 0
        
        # Selector híbrido interno
        self._hybrid_selector = HybridFCATopologySelector(
            fca_weight=self.fca_weight,
            topology_weight=self.topology_weight
        )
    
    def select(self, csp: CSP, assignment: Dict[str, Any], current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """
        Selecciona variable usando análisis adaptativo.
        
        Args:
            csp: Problema CSP
            assignment: Asignación parcial actual
            current_domains: Dominios actuales de variables
        
        Returns:
            Variable a asignar, o None si todas están asignadas
        """
        # Actualizar pesos del selector híbrido
        self._hybrid_selector.fca_weight = self.fca_weight
        self._hybrid_selector.topology_weight = self.topology_weight
        
        # Seleccionar usando híbrido
        selected = self._hybrid_selector.select(csp, assignment, current_domains)
        
        # Incrementar contador
        self._total_selections += 1
        
        # Cada N selecciones, ajustar pesos
        if self._total_selections % 10 == 0:
            self._adjust_weights()
        
        return selected
    
    def _adjust_weights(self):
        """
        Ajusta pesos basándose en efectividad observada.
        
        Esta es una implementación simple que podría mejorarse con
        técnicas más sofisticadas (e.g., bandits multi-armed).
        """
        total_successes = self._fca_successes + self._topology_successes
        
        if total_successes > 0:
            # Ajustar pesos proporcionalmente a éxitos
            self.fca_weight = self._fca_successes / total_successes
            self.topology_weight = self._topology_successes / total_successes
            
            # Aplicar suavizado para evitar convergencia prematura
            smoothing = 0.1
            self.fca_weight = (1 - smoothing) * self.fca_weight + smoothing * 0.5
            self.topology_weight = (1 - smoothing) * self.topology_weight + smoothing * 0.5
            
            # Normalizar
            total = self.fca_weight + self.topology_weight
            if total > 0:
                self.fca_weight /= total
                self.topology_weight /= total
    
    def record_success(self, strategy: str):
        """
        Registra un éxito de una estrategia específica.
        
        Args:
            strategy: 'fca' o 'topology'
        """
        if strategy == 'fca':
            self._fca_successes += 1
        elif strategy == 'topology':
            self._topology_successes += 1
    
    def reset_cache(self):
        """Limpia caché del selector híbrido."""
        self._hybrid_selector.reset_cache()
    
    def reset_adaptation(self):
        """Reinicia métricas de adaptación."""
        self._fca_successes = 0
        self._topology_successes = 0
        self._total_selections = 0
        self.fca_weight = 0.5
        self.topology_weight = 0.5

