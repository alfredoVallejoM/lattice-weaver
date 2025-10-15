"""
Watched Literals - Técnica de SAT Solvers para CSP

Evita revisar restricciones innecesariamente mediante "watched literals".

Idea: Para cada restricción, mantener 2 "watched variables" que tienen valores
válidos. Solo revisar la restricción cuando una de las watched variables cambia.

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass, field

from lattice_weaver.fibration.general_constraint import GeneralConstraint as Constraint


@dataclass
class WatchedConstraint:
    """Restricción con watched literals."""
    constraint: Constraint
    watched_vars: List[str] = field(default_factory=list)
    is_satisfied: bool = False
    last_check_assignment: Dict[str, Any] = field(default_factory=dict)


class WatchedLiteralsManager:
    """
    Gestor de Watched Literals para restricciones.
    
    Mantiene watched variables para cada restricción y solo revisa
    restricciones cuando watched variables cambian.
    """
    
    def __init__(self, constraints: List[Constraint]):
        """
        Inicializa Watched Literals Manager.
        
        Args:
            constraints: Lista de restricciones
        """
        self.constraints = constraints
        
        # Watched constraints
        self.watched: Dict[str, WatchedConstraint] = {}
        
        # Índice: variable -> restricciones que la watch
        self.var_to_constraints: Dict[str, Set[str]] = {}
        
        # Inicializar watched variables
        self._initialize_watched()
        
        # Estadísticas
        self.stats = {
            'constraints_checked': 0,
            'constraints_skipped': 0,
            'watched_updates': 0
        }
    
    def _initialize_watched(self):
        """Inicializa watched variables para todas las restricciones."""
        for constraint in self.constraints:
            constraint_id = id(constraint)
            
            # Seleccionar 2 variables para watch (o todas si hay menos de 2)
            watched_vars = list(constraint.variables)[:2]
            
            watched_constraint = WatchedConstraint(
                constraint=constraint,
                watched_vars=watched_vars
            )
            
            self.watched[constraint_id] = watched_constraint
            
            # Actualizar índice
            for var in watched_vars:
                if var not in self.var_to_constraints:
                    self.var_to_constraints[var] = set()
                self.var_to_constraints[var].add(constraint_id)
    
    def notify_assignment(
        self,
        variable: str,
        value: Any,
        assignment: Dict[str, Any]
    ) -> List[Constraint]:
        """
        Notifica que una variable fue asignada.
        
        Retorna restricciones que necesitan ser revisadas.
        
        Args:
            variable: Variable asignada
            value: Valor asignado
            assignment: Asignación completa actual
        
        Returns:
            Lista de restricciones a revisar
        """
        to_check = []
        
        # Obtener restricciones que watch esta variable
        constraint_ids = self.var_to_constraints.get(variable, set()).copy()
        
        for constraint_id in constraint_ids:
            watched_constraint = self.watched[constraint_id]
            
            # Si la restricción ya está satisfecha, skip
            if watched_constraint.is_satisfied:
                self.stats['constraints_skipped'] += 1
                continue
            
            # Revisar si necesitamos actualizar watched variables
            needs_check = self._update_watched(watched_constraint, assignment)
            
            if needs_check:
                to_check.append(watched_constraint.constraint)
                self.stats['constraints_checked'] += 1
            else:
                self.stats['constraints_skipped'] += 1
        
        return to_check
    
    def _update_watched(
        self,
        watched_constraint: WatchedConstraint,
        assignment: Dict[str, Any]
    ) -> bool:
        """
        Actualiza watched variables de una restricción.
        
        Args:
            watched_constraint: Restricción watched
            assignment: Asignación actual
        
        Returns:
            True si la restricción necesita ser revisada
        """
        constraint = watched_constraint.constraint
        watched_vars = watched_constraint.watched_vars
        
        # Verificar si alguna watched variable fue asignada
        assigned_watched = [
            var for var in watched_vars
            if var in assignment
        ]
        
        if not assigned_watched:
            # Ninguna watched variable asignada, no revisar
            return False
        
        # Intentar encontrar nuevas watched variables
        for var in constraint.variables:
            if var not in watched_vars and var not in assignment:
                # Encontramos variable no asignada, reemplazar watched
                # Reemplazar la primera watched asignada
                old_var = assigned_watched[0]
                watched_vars.remove(old_var)
                watched_vars.append(var)
                
                # Actualizar índice
                self.var_to_constraints[old_var].discard(id(constraint))
                if var not in self.var_to_constraints:
                    self.var_to_constraints[var] = set()
                self.var_to_constraints[var].add(id(constraint))
                
                self.stats['watched_updates'] += 1
                
                # No necesitamos revisar aún
                return False
        
        # No encontramos variable no asignada, necesitamos revisar
        return True
    
    def mark_satisfied(self, constraint: Constraint):
        """
        Marca una restricción como satisfecha.
        
        Args:
            constraint: Restricción satisfecha
        """
        constraint_id = id(constraint)
        if constraint_id in self.watched:
            self.watched[constraint_id].is_satisfied = True
    
    def mark_unsatisfied(self, constraint: Constraint):
        """
        Marca una restricción como no satisfecha.
        
        Args:
            constraint: Restricción no satisfecha
        """
        constraint_id = id(constraint)
        if constraint_id in self.watched:
            self.watched[constraint_id].is_satisfied = False
    
    def reset(self):
        """Resetea estado de todas las restricciones."""
        for watched_constraint in self.watched.values():
            watched_constraint.is_satisfied = False
            watched_constraint.last_check_assignment.clear()
    
    def get_stats(self) -> dict:
        """Retorna estadísticas."""
        total = self.stats['constraints_checked'] + self.stats['constraints_skipped']
        skip_rate = self.stats['constraints_skipped'] / total if total > 0 else 0
        
        return {
            **self.stats,
            'total_notifications': total,
            'skip_rate': skip_rate
        }
    
    def reset_stats(self):
        """Resetea estadísticas."""
        self.stats = {
            'constraints_checked': 0,
            'constraints_skipped': 0,
            'watched_updates': 0
        }


class WatchedLiteralsConstraintChecker:
    """
    Checker de restricciones con Watched Literals.
    
    Wrapper que usa Watched Literals para optimizar checking de restricciones.
    """
    
    def __init__(self, constraints: List[Constraint]):
        """
        Inicializa checker.
        
        Args:
            constraints: Lista de restricciones
        """
        self.constraints = constraints
        self.manager = WatchedLiteralsManager(constraints)
    
    def check_constraints(
        self,
        assignment: Dict[str, Any],
        changed_variable: Optional[str] = None
    ) -> Tuple[bool, List[Constraint]]:
        """
        Verifica restricciones con Watched Literals.
        
        Args:
            assignment: Asignación actual
            changed_variable: Variable que cambió (opcional)
        
        Returns:
            (all_satisfied, violated_constraints)
        """
        violated = []
        
        if changed_variable is not None:
            # Solo revisar restricciones afectadas
            to_check = self.manager.notify_assignment(
                changed_variable,
                assignment[changed_variable],
                assignment
            )
        else:
            # Revisar todas las restricciones
            to_check = self.constraints
        
        # Verificar restricciones
        for constraint in to_check:
            # Solo verificar si todas las variables están asignadas
            if all(var in assignment for var in constraint.variables):
                satisfied = constraint.predicate(assignment)
                
                if satisfied:
                    self.manager.mark_satisfied(constraint)
                else:
                    self.manager.mark_unsatisfied(constraint)
                    violated.append(constraint)
        
        return len(violated) == 0, violated
    
    def reset(self):
        """Resetea estado."""
        self.manager.reset()
    
    def get_stats(self) -> dict:
        """Retorna estadísticas."""
        return self.manager.get_stats()

