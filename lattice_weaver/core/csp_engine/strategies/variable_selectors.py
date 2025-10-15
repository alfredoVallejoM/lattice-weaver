"""
Implementaciones concretas de estrategias de selección de variables.

Este módulo contiene las implementaciones de las heurísticas clásicas y modernas
para seleccionar la siguiente variable a asignar durante la búsqueda de backtracking.
"""

from typing import Dict, List, Any, Optional
from .base import VariableSelector
from ...csp_problem import CSP


class FirstUnassignedSelector(VariableSelector):
    """
    Selecciona la primera variable no asignada en el orden original.
    
    Esta es la estrategia más simple, equivalente al backtracking básico sin heurísticas.
    Útil como baseline para comparaciones.
    """
    
    def select(self,
               csp: CSP,
               assignment: Dict[str, Any],
               current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """Selecciona la primera variable no asignada."""
        for var in csp.variables:
            if var not in assignment:
                return var
        return None


class MRVSelector(VariableSelector):
    """
    Minimum Remaining Values (MRV) heuristic.
    
    Selecciona la variable con el menor número de valores legales restantes.
    También conocida como "most constrained variable" o "fail-first" heuristic.
    
    Ventajas:
    - Detecta fallos temprano (si dominio vacío)
    - Reduce el factor de ramificación
    - Muy efectiva en problemas fuertemente restringidos
    
    Referencias:
    - Haralick & Elliot (1980), "Increasing Tree Search Efficiency for CSPs"
    """
    
    def select(self,
               csp: CSP,
               assignment: Dict[str, Any],
               current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """Selecciona la variable con menor dominio restante."""
        unassigned_vars = [v for v in csp.variables if v not in assignment]
        if not unassigned_vars:
            return None
        
        return min(unassigned_vars, key=lambda var: len(current_domains[var]))


class DegreeSelector(VariableSelector):
    """
    Degree heuristic.
    
    Selecciona la variable involucrada en el mayor número de restricciones
    con variables no asignadas.
    
    Ventajas:
    - Reduce el espacio de búsqueda futuro
    - Efectiva cuando hay muchas variables con dominios similares
    - Buen desempate para MRV
    
    Referencias:
    - Dechter & Pearl (1988), "Network-Based Heuristics for CSPs"
    """
    
    def select(self,
               csp: CSP,
               assignment: Dict[str, Any],
               current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """Selecciona la variable con mayor degree."""
        unassigned_vars = [v for v in csp.variables if v not in assignment]
        if not unassigned_vars:
            return None
        
        # Calcular degree para cada variable no asignada
        degrees = {}
        for var in unassigned_vars:
            degree = 0
            for constraint in csp.constraints:
                if var in constraint.scope:
                    # Contar cuántas otras variables no asignadas están en la misma restricción
                    for other_var in constraint.scope:
                        if other_var != var and other_var in unassigned_vars:
                            degree += 1
            degrees[var] = degree
        
        # Seleccionar variable con mayor degree
        return max(unassigned_vars, key=lambda var: degrees[var])


class MRVDegreeSelector(VariableSelector):
    """
    Combinación de MRV y Degree heuristics.
    
    Selecciona la variable con menor dominio restante (MRV).
    En caso de empate, usa Degree como desempate (mayor degree primero).
    
    Esta es la estrategia implementada en la Fase 1 del proyecto.
    
    Ventajas:
    - Combina las ventajas de MRV y Degree
    - MRV detecta fallos temprano
    - Degree resuelve empates efectivamente
    - Estado del arte para backtracking determinístico
    
    Referencias:
    - Russell & Norvig (2020), "Artificial Intelligence: A Modern Approach"
    - Capítulo 6.3.1: Variable and Value Ordering
    """
    
    def select(self,
               csp: CSP,
               assignment: Dict[str, Any],
               current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """Selecciona variable con MRV, usando Degree como desempate."""
        unassigned_vars = [v for v in csp.variables if v not in assignment]
        if not unassigned_vars:
            return None
        
        # Calcular degree para cada variable no asignada
        degrees = {}
        for var in unassigned_vars:
            degree = 0
            for constraint in csp.constraints:
                if var in constraint.scope:
                    # Contar cuántas otras variables no asignadas están en la misma restricción
                    for other_var in constraint.scope:
                        if other_var != var and other_var in unassigned_vars:
                            degree += 1
            degrees[var] = degree
        
        # Combinar MRV y Degree: priorizar MRV, luego Degree (mayor degree primero)
        return min(unassigned_vars, key=lambda var: (len(current_domains[var]), -degrees[var]))

