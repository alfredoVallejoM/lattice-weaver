"""
General Constraint - Restricción General para N Variables

Clase de restricción más general que soporta N variables (no solo pares).

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

from typing import List, Callable, Dict, Any, Set
from dataclasses import dataclass


@dataclass
class GeneralConstraint:
    """
    Restricción general sobre N variables.
    
    Attributes:
        variables: Lista de variables involucradas
        predicate: Función que evalúa la restricción
        name: Nombre opcional de la restricción
    """
    
    variables: List[str]
    predicate: Callable[[Dict[str, Any]], bool]
    name: str = ""
    
    def __post_init__(self):
        """Validación post-inicialización."""
        if not self.variables:
            raise ValueError("Constraint must have at least one variable")
        
        if not self.name:
            self.name = f"constraint_{id(self)}"
    
    def evaluate(self, assignment: Dict[str, Any]) -> bool:
        """
        Evalúa la restricción con una asignación.
        
        Args:
            assignment: Asignación de variables
        
        Returns:
            True si la restricción se satisface
        """
        return self.predicate(assignment)
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        """Alias de evaluate."""
        return self.evaluate(assignment)
    
    def involves_variable(self, variable: str) -> bool:
        """
        Verifica si la restricción involucra una variable.
        
        Args:
            variable: Variable a verificar
        
        Returns:
            True si la restricción involucra la variable
        """
        return variable in self.variables
    
    def get_variables(self) -> Set[str]:
        """Retorna conjunto de variables."""
        return set(self.variables)
    
    def arity(self) -> int:
        """Retorna aridad (número de variables)."""
        return len(self.variables)
    
    def __repr__(self) -> str:
        """Representación string."""
        vars_str = ", ".join(self.variables)
        return f"GeneralConstraint({self.name}: [{vars_str}])"
    
    def __hash__(self) -> int:
        """Hash basado en ID."""
        return id(self)
    
    def __eq__(self, other) -> bool:
        """Igualdad basada en identidad."""
        return self is other

