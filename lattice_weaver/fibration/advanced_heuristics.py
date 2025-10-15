"""
Advanced Heuristics - Heurísticas Avanzadas de Variable Ordering

Implementa heurísticas sofisticadas para ordenamiento de variables:
- Weighted Degree Heuristic (WDeg)
- Impact-Based Search (IBS)
- Conflict-Directed Variable Ordering (CDVO)

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

from typing import Dict, Any, List, Set, Optional, Tuple
from collections import defaultdict
import math

from lattice_weaver.fibration.general_constraint import GeneralConstraint as Constraint


class WeightedDegreeHeuristic:
    """
    Weighted Degree Heuristic (WDeg).
    
    Mantiene pesos dinámicos para cada restricción basados en conflictos.
    Selecciona variable con mayor weighted degree (suma de pesos de restricciones).
    """
    
    def __init__(self, constraints: List[Constraint]):
        """
        Inicializa Weighted Degree Heuristic.
        
        Args:
            constraints: Lista de restricciones
        """
        self.constraints = constraints
        
        # Pesos de restricciones (inicialmente 1)
        self.weights: Dict[int, float] = {
            id(c): 1.0 for c in constraints
        }
        
        # Índice: variable -> restricciones que la involucran
        self.var_to_constraints: Dict[str, Set[int]] = defaultdict(set)
        for constraint in constraints:
            for var in constraint.variables:
                self.var_to_constraints[var].add(id(constraint))
        
        # Estadísticas
        self.conflict_count = 0
    
    def record_conflict(self, constraint: Constraint):
        """
        Registra un conflicto en una restricción.
        
        Incrementa el peso de la restricción.
        
        Args:
            constraint: Restricción que causó conflicto
        """
        constraint_id = id(constraint)
        self.weights[constraint_id] += 1.0
        self.conflict_count += 1
    
    def get_weighted_degree(self, variable: str) -> float:
        """
        Calcula weighted degree de una variable.
        
        Args:
            variable: Variable
        
        Returns:
            Weighted degree (suma de pesos de restricciones)
        """
        constraint_ids = self.var_to_constraints.get(variable, set())
        return sum(self.weights.get(cid, 1.0) for cid in constraint_ids)
    
    def select_variable(
        self,
        unassigned: List[str],
        domains: Dict[str, List[Any]]
    ) -> str:
        """
        Selecciona variable con MRV + WDeg.
        
        Args:
            unassigned: Variables no asignadas
            domains: Dominios de variables
        
        Returns:
            Variable seleccionada
        """
        if not unassigned:
            return None
        
        # MRV + WDeg: domain_size / weighted_degree
        def score(var):
            domain_size = len(domains.get(var, []))
            wdeg = self.get_weighted_degree(var)
            
            # Evitar división por cero
            if wdeg == 0:
                wdeg = 0.1
            
            # Menor es mejor (MRV / WDeg)
            return domain_size / wdeg
        
        return min(unassigned, key=score)


class ImpactBasedSearch:
    """
    Impact-Based Search (IBS).
    
    Mide el "impacto" de asignar valores a variables y usa esto para
    ordenamiento de variables y valores.
    """
    
    def __init__(
        self,
        variables: List[str],
        domains: Dict[str, List[Any]],
        constraints: List[Constraint]
    ):
        """
        Inicializa Impact-Based Search.
        
        Args:
            variables: Lista de variables
            domains: Dominios de variables
            constraints: Restricciones
        """
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        
        # Impactos: variable -> valor -> impacto
        self.impacts: Dict[str, Dict[Any, float]] = defaultdict(lambda: defaultdict(float))
        
        # Inicializar impactos
        self._initialize_impacts()
    
    def _initialize_impacts(self):
        """Inicializa impactos mediante probing."""
        for var in self.variables:
            for value in self.domains.get(var, []):
                # Medir impacto de asignar var=value
                impact = self._measure_impact(var, value)
                self.impacts[var][value] = impact
    
    def _measure_impact(self, variable: str, value: Any) -> float:
        """
        Mide el impacto de asignar variable=value.
        
        Impacto = reducción en espacio de búsqueda.
        
        Args:
            variable: Variable
            value: Valor
        
        Returns:
            Impacto (0-1)
        """
        # Crear asignación temporal
        assignment = {variable: value}
        
        # Contar valores eliminados en otras variables
        eliminated = 0
        total = 0
        
        for other_var in self.variables:
            if other_var == variable:
                continue
            
            domain = self.domains.get(other_var, [])
            total += len(domain)
            
            # Verificar qué valores son inconsistentes
            for other_value in domain:
                assignment[other_var] = other_value
                
                # Verificar restricciones
                inconsistent = False
                for constraint in self.constraints:
                    if variable in constraint.variables and other_var in constraint.variables:
                        if not constraint.predicate(assignment):
                            inconsistent = True
                            break
                
                if inconsistent:
                    eliminated += 1
                
                del assignment[other_var]
        
        # Impacto = fracción de valores eliminados
        return eliminated / total if total > 0 else 0.0
    
    def update_impact(self, variable: str, value: Any, actual_reduction: float):
        """
        Actualiza impacto basado en reducción real.
        
        Args:
            variable: Variable
            value: Valor
            actual_reduction: Reducción real observada
        """
        # Promedio móvil exponencial
        alpha = 0.1
        old_impact = self.impacts[variable][value]
        new_impact = alpha * actual_reduction + (1 - alpha) * old_impact
        self.impacts[variable][value] = new_impact
    
    def get_variable_impact(self, variable: str) -> float:
        """
        Calcula impacto promedio de una variable.
        
        Args:
            variable: Variable
        
        Returns:
            Impacto promedio
        """
        impacts = list(self.impacts[variable].values())
        return sum(impacts) / len(impacts) if impacts else 0.0
    
    def select_variable(self, unassigned: List[str]) -> str:
        """
        Selecciona variable con mayor impacto.
        
        Args:
            unassigned: Variables no asignadas
        
        Returns:
            Variable seleccionada
        """
        if not unassigned:
            return None
        
        return max(unassigned, key=self.get_variable_impact)
    
    def order_values(self, variable: str) -> List[Any]:
        """
        Ordena valores por impacto (menor primero).
        
        Args:
            variable: Variable
        
        Returns:
            Valores ordenados
        """
        values = list(self.impacts[variable].keys())
        return sorted(values, key=lambda v: self.impacts[variable][v])


class ConflictDirectedVariableOrdering:
    """
    Conflict-Directed Variable Ordering (CDVO).
    
    Combina WDeg + IBS + información de conflictos del TMS.
    """
    
    def __init__(
        self,
        variables: List[str],
        domains: Dict[str, List[Any]],
        constraints: List[Constraint]
    ):
        """
        Inicializa CDVO.
        
        Args:
            variables: Lista de variables
            domains: Dominios de variables
            constraints: Restricciones
        """
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        
        # Componentes
        self.wdeg = WeightedDegreeHeuristic(constraints)
        self.ibs = ImpactBasedSearch(variables, domains, constraints)
        
        # Pesos para combinación
        self.alpha_mrv = 0.4
        self.alpha_wdeg = 0.3
        self.alpha_ibs = 0.3
    
    def record_conflict(self, constraint: Constraint):
        """Registra conflicto."""
        self.wdeg.record_conflict(constraint)
    
    def select_variable(
        self,
        unassigned: List[str],
        domains: Dict[str, List[Any]]
    ) -> str:
        """
        Selecciona variable con scoring combinado.
        
        Score = alpha_mrv * MRV + alpha_wdeg * WDeg + alpha_ibs * IBS
        
        Args:
            unassigned: Variables no asignadas
            domains: Dominios actuales
        
        Returns:
            Variable seleccionada
        """
        if not unassigned:
            return None
        
        def score(var):
            # MRV (normalizado)
            domain_size = len(domains.get(var, []))
            max_domain = max(len(domains.get(v, [])) for v in unassigned)
            mrv_score = 1 - (domain_size / max_domain) if max_domain > 0 else 0
            
            # WDeg (normalizado)
            wdeg = self.wdeg.get_weighted_degree(var)
            max_wdeg = max(self.wdeg.get_weighted_degree(v) for v in unassigned)
            wdeg_score = wdeg / max_wdeg if max_wdeg > 0 else 0
            
            # IBS (normalizado)
            impact = self.ibs.get_variable_impact(var)
            max_impact = max(self.ibs.get_variable_impact(v) for v in unassigned)
            ibs_score = impact / max_impact if max_impact > 0 else 0
            
            # Combinación ponderada
            return (
                self.alpha_mrv * mrv_score +
                self.alpha_wdeg * wdeg_score +
                self.alpha_ibs * ibs_score
            )
        
        return max(unassigned, key=score)
    
    def order_values(
        self,
        variable: str,
        domains: Dict[str, List[Any]]
    ) -> List[Any]:
        """
        Ordena valores usando IBS.
        
        Args:
            variable: Variable
            domains: Dominios actuales
        
        Returns:
            Valores ordenados
        """
        return self.ibs.order_values(variable)
    
    def update_weights(self, alpha_mrv: float, alpha_wdeg: float, alpha_ibs: float):
        """
        Actualiza pesos de combinación.
        
        Args:
            alpha_mrv: Peso de MRV
            alpha_wdeg: Peso de WDeg
            alpha_ibs: Peso de IBS
        """
        total = alpha_mrv + alpha_wdeg + alpha_ibs
        self.alpha_mrv = alpha_mrv / total
        self.alpha_wdeg = alpha_wdeg / total
        self.alpha_ibs = alpha_ibs / total

