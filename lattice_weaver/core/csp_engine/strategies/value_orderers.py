"""
Implementaciones concretas de estrategias de ordenamiento de valores.

Este módulo contiene las implementaciones de las heurísticas clásicas y modernas
para ordenar los valores del dominio de una variable durante la búsqueda.
"""

import random
from typing import Dict, List, Any
from .base import ValueOrderer
from ...csp_problem import CSP


class NaturalOrderer(ValueOrderer):
    """
    Mantiene el orden natural del dominio.
    
    No realiza ningún reordenamiento, simplemente retorna los valores
    en el orden en que aparecen en el dominio actual.
    
    Útil como baseline para comparaciones y cuando no hay información
    adicional para guiar el ordenamiento.
    """
    
    def order(self,
              var: str,
              csp: CSP,
              assignment: Dict[str, Any],
              current_domains: Dict[str, List[Any]]) -> List[Any]:
        """Retorna valores en orden natural."""
        return list(current_domains[var])


class LCVOrderer(ValueOrderer):
    """
    Least Constraining Value (LCV) heuristic.
    
    Ordena los valores para probar primero aquellos que eliminan menos
    opciones de las variables vecinas no asignadas.
    
    La idea es dejar la máxima flexibilidad para asignaciones futuras,
    reduciendo la probabilidad de backtracking.
    
    Ventajas:
    - Reduce backtracking significativamente
    - Especialmente efectiva en problemas con alta conectividad
    - Complementa bien a MRV (MRV para variables, LCV para valores)
    
    Desventajas:
    - Más costosa computacionalmente que orden natural
    - Requiere revisar restricciones con variables vecinas
    
    Referencias:
    - Haralick & Elliot (1980), "Increasing Tree Search Efficiency for CSPs"
    - Russell & Norvig (2020), "Artificial Intelligence: A Modern Approach"
    """
    
    def order(self,
              var: str,
              csp: CSP,
              assignment: Dict[str, Any],
              current_domains: Dict[str, List[Any]]) -> List[Any]:
        """Ordena valores por LCV (menos restrictivos primero)."""
        domain = current_domains[var]
        
        # Para cada valor, contar cuántos valores elimina de variables vecinas
        value_constraints = []
        for value in domain:
            eliminated_count = 0
            
            # Revisar cada restricción que involucra a var
            for constraint in csp.constraints:
                if var in constraint.scope and len(constraint.scope) == 2:
                    other_var = next((v for v in constraint.scope if v != var), None)
                    if other_var and other_var not in assignment:
                        # Contar cuántos valores de other_var son incompatibles con value
                        for other_value in current_domains[other_var]:
                            if var == list(constraint.scope)[0]:
                                if not constraint.relation(value, other_value):
                                    eliminated_count += 1
                            else:
                                if not constraint.relation(other_value, value):
                                    eliminated_count += 1
            
            value_constraints.append((value, eliminated_count))
        
        # Ordenar por número de valores eliminados (menos eliminados primero)
        value_constraints.sort(key=lambda x: x[1])
        return [value for value, _ in value_constraints]


class RandomOrderer(ValueOrderer):
    """
    Ordena valores aleatoriamente.
    
    Útil para:
    - Experimentación y comparación con otras estrategias
    - Evitar sesgos en el orden del dominio
    - Algoritmos probabilísticos o de muestreo
    
    Nota: Para reproducibilidad, se puede fijar la semilla de random
    antes de usar esta estrategia.
    """
    
    def __init__(self, seed: int = None):
        """
        Inicializa el ordenador aleatorio.
        
        Args:
            seed: Semilla para el generador de números aleatorios.
                  Si es None, usa semilla aleatoria del sistema.
        """
        self.seed = seed
        self._rng = random.Random(seed)
    
    def order(self,
              var: str,
              csp: CSP,
              assignment: Dict[str, Any],
              current_domains: Dict[str, List[Any]]) -> List[Any]:
        """Ordena valores aleatoriamente."""
        values = list(current_domains[var])
        self._rng.shuffle(values)
        return values
    
    def __repr__(self) -> str:
        return f"RandomOrderer(seed={self.seed})"

