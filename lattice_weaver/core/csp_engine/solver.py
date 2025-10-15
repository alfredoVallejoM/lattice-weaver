from typing import Dict, Any, List, Optional, Tuple, Callable
from lattice_weaver.core.csp_engine.tracing import ExecutionTracer, TraceEvent
from dataclasses import dataclass, field
import time
import itertools

from ..csp_problem import CSP, Constraint
from .strategies import VariableSelector, ValueOrderer
from .strategies import FirstUnassignedSelector, NaturalOrderer

@dataclass
class CSPSolution:
    """
    Representa una solución encontrada para un CSP.
    """
    assignment: Dict[str, Any]
    is_consistent: bool = True

@dataclass
class CSPSolutionStats:
    """
    Estadísticas de la ejecución de un solver CSP.
    """
    solutions: List[CSPSolution] = field(default_factory=list)
    nodes_explored: int = 0
    backtracks: int = 0
    constraints_checked: int = 0
    time_elapsed: float = 0.0

class CSPSolver:
    """
    Un solver básico para Problemas de Satisfacción de Restricciones (CSP).
    Implementa un algoritmo de backtracking con forward checking.
    """


    def __init__(self, 
                 csp: CSP, 
                 tracer: Optional[ExecutionTracer] = None,
                 variable_selector: Optional[VariableSelector] = None,
                 value_orderer: Optional[ValueOrderer] = None):
        """
        Inicializa el CSPSolver.
        
        Args:
            csp: El problema CSP a resolver
            tracer: Tracer opcional para debugging/análisis
            variable_selector: Estrategia para seleccionar variables (default: FirstUnassignedSelector)
            value_orderer: Estrategia para ordenar valores (default: NaturalOrderer)
        """
        self.csp = csp
        self.assignment: Dict[str, Any] = {}
        self.stats = CSPSolutionStats()
        self.tracer = tracer
        
        # Estrategias modulares (usar defaults si no se especifican)
        self.variable_selector = variable_selector or FirstUnassignedSelector()
        self.value_orderer = value_orderer or NaturalOrderer()

        if self.tracer and self.tracer.enabled:
            pass # No hay un método generico record_event. Se puede registrar un evento de inicialización si es necesario, o simplemente omitirlo.

    def _is_consistent(self, var: str, value: Any) -> bool:
        for constraint in self.csp.constraints:
            self.stats.constraints_checked += 1
            if var in constraint.scope:
                # Si la restricción es unaria y no se satisface
                if len(constraint.scope) == 1:
                    is_satisfied = constraint.relation(value)
                    if self.tracer and self.tracer.enabled:
                        self.tracer.record_constraint_check(constraint=constraint, result=is_satisfied, variable=var, value=value)
                    if not is_satisfied:
                        return False
                # Si la restricción es binaria y ambas variables están asignadas
                elif len(constraint.scope) == 2:
                    other_var = next(v for v in constraint.scope if v != var)
                    if other_var in self.assignment:
                        is_satisfied = False
                        if var == list(constraint.scope)[0]: # Asegurar el orden de los argumentos
                            is_satisfied = constraint.relation(value, self.assignment[other_var])
                        else:
                            is_satisfied = constraint.relation(self.assignment[other_var], value)
                        
                        if self.tracer and self.tracer.enabled:
                            self.tracer.record_constraint_check(constraint=constraint, result=is_satisfied, variable=var, value=value, other_var=other_var, other_value=self.assignment[other_var])
                        
                        if not is_satisfied:
                            return False
        return True

    def _select_unassigned_variable(self, current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """
        Selecciona la siguiente variable a asignar usando la estrategia configurada.
        
        Args:
            current_domains: Dominios actuales de todas las variables
        
        Returns:
            Nombre de la variable a asignar, o None si todas están asignadas
        """
        return self.variable_selector.select(self.csp, self.assignment, current_domains)
    
    def _order_domain_values(self, var: str, current_domains: Dict[str, List[Any]]) -> List[Any]:
        """
        Ordena los valores del dominio de una variable usando la estrategia configurada.
        
        Args:
            var: Variable cuyo dominio ordenar
            current_domains: Dominios actuales de todas las variables
        
        Returns:
            Lista de valores ordenados según la estrategia
        """
        return self.value_orderer.order(var, self.csp, self.assignment, current_domains)

    def _backtrack(self, current_domains: Dict[str, List[Any]], all_solutions: bool, max_solutions: int) -> bool:
        self.stats.nodes_explored += 1
        if self.tracer and self.tracer.enabled:
            pass # No hay un método generico record_event para node_explored.

        if len(self.assignment) == len(self.csp.variables):
            solution = CSPSolution(assignment=self.assignment.copy())
            self.stats.solutions.append(solution)
            if self.tracer and self.tracer.enabled:
                pass # No hay un método generico record_event para solution_found.
            # Si no se buscan todas las soluciones, terminar (retornar True para propagar hacia arriba)
            # Si se buscan todas, continuar (retornar False para seguir explorando)
            return not all_solutions

        var = self._select_unassigned_variable(current_domains)
        if var is None:
            return True

        # Ordenar valores del dominio según la estrategia configurada
        ordered_values = self._order_domain_values(var, current_domains)
        
        for value in ordered_values:
            if self._is_consistent(var, value):
                self.assignment[var] = value
                if self.tracer and self.tracer.enabled:
                    self.tracer.record_assignment(variable=var, value=value, depth=len(self.assignment))

                # Forward checking: reducir dominios de variables no asignadas
                new_domains = {v: list(d) for v, d in current_domains.items()}
                pruned_values = self._forward_check(var, value, new_domains)

                if pruned_values is not None: # Si no hay inconsistencia
                    if self._backtrack(new_domains, all_solutions, max_solutions):
                        if not all_solutions or len(self.stats.solutions) >= max_solutions:
                            return True
                else:
                    self.stats.backtracks += 1
                    if self.tracer and self.tracer.enabled:
                        self.tracer.record_backtrack(variable=var, value=value, depth=len(self.assignment))

            del self.assignment[var] # Deshacer asignación
            if self.tracer and self.tracer.enabled:
                pass # No hay un método especifico para unassign.

        return False

    def _forward_check(self, var: str, value: Any, domains: Dict[str, List[Any]]) -> Optional[Dict[str, List[Any]]]:
        # Para cada restricción que involucra a 'var' y otra variable no asignada 'other_var'
        for constraint in self.csp.constraints:
            if var in constraint.scope and len(constraint.scope) == 2:
                other_var = next((v for v in constraint.scope if v != var), None)
                if other_var and other_var not in self.assignment:
                    # Eliminar valores inconsistentes del dominio de 'other_var'
                    original_other_domain = list(domains[other_var])
                    domains[other_var] = [ 
                        other_value for other_value in original_other_domain
                        if (var == list(constraint.scope)[0] and constraint.relation(value, other_value)) or
                           (var == list(constraint.scope)[1] and constraint.relation(other_value, value))
                    ]
                    if not domains[other_var]: # Si el dominio de other_var se vacía
                        if self.tracer and self.tracer.enabled:
                            pass # No hay un método especifico para domain_wipeout.
                        return None # Inconsistencia detectada
        return domains

    def solve(self, all_solutions: bool = False, max_solutions: int = 1) -> CSPSolutionStats:
        """
        Resuelve el CSP usando backtracking con forward checking.
        
        Args:
            all_solutions: Si True, busca todas las soluciones; si False, se detiene en la primera
            max_solutions: Número máximo de soluciones a buscar (solo relevante si all_solutions=True)
        
        Returns:
            CSPSolutionStats con las soluciones encontradas y estadísticas de ejecución
        """
        start_time = time.time()
        
        # Inicializar dominios
        current_domains = {var: list(self.csp.domains[var]) for var in self.csp.variables}
        
        # Ejecutar backtracking
        self._backtrack(current_domains, all_solutions, max_solutions)
        
        # Registrar tiempo de ejecución
        self.stats.time_elapsed = time.time() - start_time
        
        if self.tracer and self.tracer.enabled:
            pass # No hay un método generico para search_completed.
        
        return self.stats

