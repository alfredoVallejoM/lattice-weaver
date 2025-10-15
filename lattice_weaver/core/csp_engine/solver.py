from typing import Dict, Any, List, Optional, Tuple, Callable
from lattice_weaver.core.csp_engine.tracing import ExecutionTracer, TraceEvent
from dataclasses import dataclass, field
import time
import itertools

from ..csp_problem import CSP, Constraint
from ...fibration.csp_adapter import CSPToConstraintHierarchyAdapter
from ...fibration.fibration_search_solver import FibrationSearchSolver

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


    def __init__(self, csp: CSP, tracer: Optional[ExecutionTracer] = None):
        self.csp = csp
        self.assignment: Dict[str, Any] = {}
        self.stats = CSPSolutionStats()
        self.tracer = tracer

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
        # Implementación simple: seleccionar la primera variable no asignada
        for var in self.csp.variables:
            if var not in self.assignment:
                return var
        return None

    def _backtrack(self, current_domains: Dict[str, List[Any]], all_solutions: bool, max_solutions: int) -> bool:
        self.stats.nodes_explored += 1
        if self.tracer and self.tracer.enabled:
            pass # No hay un método generico record_event para node_explored.

        if len(self.assignment) == len(self.csp.variables):
            solution = CSPSolution(assignment=self.assignment.copy())
            self.stats.solutions.append(solution)
            if self.tracer and self.tracer.enabled:
                pass # No hay un método generico record_event para solution_found.
            return all_solutions # Si se buscan todas las soluciones, continuar; si no, terminar

        var = self._select_unassigned_variable(current_domains)
        if var is None:
            return True

        original_domain = list(current_domains[var]) # Copia del dominio original
        for value in original_domain:
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
                            # No hay un método especifico para domain_wipeout. Se puede omitir o crear uno en ExecutionTracer.
                            # self.tracer.record_event(event_type="domain_wipeout", variable=other_var, metadata={"pruning_var": var, "pruning_val": value})
                            return None # Inconsistencia detectada
                    elif len(domains[other_var]) < len(original_other_domain):
                        if self.tracer and self.tracer.enabled:
                            pruned = set(original_other_domain) - set(domains[other_var])
                            self.tracer.record_propagation(variable=other_var, domain_before=original_other_domain, domain_after=domains[other_var], pruned_values=list(pruned), pruning_var=var, pruning_val=value)
        return domains

    def solve(self, all_solutions: bool = False, max_solutions: int = 1) -> CSPSolutionStats:
        start_time = time.perf_counter()
        initial_domains = {var: list(self.csp.domains[var]) for var in self.csp.variables}
        self._backtrack(initial_domains, all_solutions, max_solutions)
        end_time = time.perf_counter()
        self.stats.time_elapsed = end_time - start_time
        return self.stats

    def solve_with_fibration_flow(self) -> CSPSolutionStats:
        """
        Resuelve el CSP utilizando Fibration Flow como motor de búsqueda.
        """
        start_time = time.perf_counter()

        adapter = CSPToConstraintHierarchyAdapter()
        hierarchy, fibration_domains, metadata = adapter.convert_csp_to_hierarchy(self.csp)

        # Usar el mock si está disponible, de lo contrario, instanciar uno nuevo
        fibration_solver = getattr(self, 'fibration_solver', None)
        if fibration_solver is None:
            fibration_solver = FibrationSearchSolver(
                variables=list(self.csp.variables),
                domains=fibration_domains,
                hierarchy=hierarchy
            )
        fibration_solution = fibration_solver.solve()

        if fibration_solution:
            csp_solution = adapter.convert_hierarchy_solution_to_csp_solution(fibration_solution, metadata)
            self.stats.solutions.append(CSPSolution(assignment=csp_solution))
        
        end_time = time.perf_counter()
        self.stats.time_elapsed = end_time - start_time
        return self.stats

    def enforce_arc_consistency(self) -> bool:
        """
        Implementa el algoritmo AC3 para hacer el CSP arco-consistente.
        Retorna True si el CSP es consistente, False si se detecta una inconsistencia.
        """
        queue = []
        # Inicializar la cola con todos los arcos binarios (restricciones)
        for constraint in self.csp.constraints:
            if len(constraint.scope) == 2:
                var_i, var_j = list(constraint.scope)
                queue.append((var_i, var_j, constraint))
                queue.append((var_j, var_i, constraint)) # El arco inverso

        while queue:
            var_i, var_j, constraint = queue.pop(0)
            if self._revise(var_i, var_j, constraint):
                if not self.csp.domains[var_i]:
                    return False # Dominio vacío, inconsistencia
                # Añadir arcos vecinos a la cola
                for neighbor_constraint in self.csp.constraints:
                    if len(neighbor_constraint.scope) == 2:
                        n_var1, n_var2 = list(neighbor_constraint.scope)
                        if n_var2 == var_i and n_var1 != var_j:
                            queue.append((n_var1, n_var2, neighbor_constraint))
                        elif n_var1 == var_i and n_var2 != var_j:
                            queue.append((n_var2, n_var1, neighbor_constraint))
        return True

    def _revise(self, var_i: str, var_j: str, constraint: Constraint) -> bool:
        """
        Revisa el dominio de var_i con respecto a var_j y la restricción.
        Retorna True si el dominio de var_i fue revisado (valores eliminados).
        """
        revised = False
        new_domain_i = []
        original_domain_i = list(self.csp.domains[var_i])
        original_domain_j = list(self.csp.domains[var_j])

        for x in original_domain_i:
            # Buscar si existe algún valor 'y' en el dominio de var_j que satisfaga la restricción
            satisfiable = False
            for y in original_domain_j:
                if var_i == list(constraint.scope)[0]:
                    if constraint.relation(x, y):
                        satisfiable = True
                        break
                else:
                    if constraint.relation(y, x):
                        satisfiable = True
                        break
            if satisfiable:
                new_domain_i.append(x)
            else:
                revised = True
        
        # Actualizar el dominio de var_i si fue revisado
        if revised:
            self.csp.domains[var_i] = frozenset(new_domain_i)
        return revised

def solve_csp(csp: CSP, all_solutions: bool = False, max_solutions: int = 1) -> CSPSolutionStats:
    """
    Función de conveniencia para resolver un CSP.
    """
    solver = CSPSolver(csp)
    return solver.solve(all_solutions=all_solutions, max_solutions=max_solutions)

