import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

from .core import ArcEngine
from .domains import Domain
from .constraints import Constraint
# from ..core.csp_engine.tms import TruthMaintenanceSystem # Importar TMS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

class CSPProblem:
    """
    Representa un Problema de Satisfacción de Restricciones (CSP).
    """
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], constraints: List[Tuple[str, str, Any]]):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints

    def __repr__(self):
        return f"CSPProblem(variables={len(self.variables)}, constraints={len(self.constraints)})"

class CSPSolution:
    """
    Representa una solución a un Problema de Satisfacción de Restricciones.
    """
    def __init__(self, assignment: Dict[str, Any]):
        self.assignment = assignment

    def __repr__(self):
        return f"CSPSolution({self.assignment})"

    def __eq__(self, other):
        if not isinstance(other, CSPSolution):
            return NotImplemented
        return self.assignment == other.assignment

    def __hash__(self,):
        return hash(frozenset(self.assignment.items()))


class CSPSolver:
    """
    Un solver de CSP que utiliza ArcEngine para la propagación de restricciones
    y un algoritmo de backtracking para encontrar soluciones.
    """
    def __init__(self, use_tms: bool = False, parallel: bool = False, parallel_mode: str = 'thread'):
        self.arc_engine = ArcEngine(use_tms=use_tms, parallel=parallel, parallel_mode=parallel_mode)
        self.problem: Optional[CSPProblem] = None
        self.backtracks = 0
        self.constraints_checked = 0

    def _setup_arc_engine(self, problem: CSPProblem):
        """
        Configura el ArcEngine con las variables y restricciones del problema.
        """
        # Asegurarse de que el ArcEngine esté limpio antes de configurar un nuevo problema
        if self.arc_engine.tms:
            self.arc_engine.tms.clear()
        
        # Re-inicializar ArcEngine para asegurar un estado limpio y aplicar use_tms, parallel, etc.
        self.arc_engine = ArcEngine(use_tms=self.arc_engine.use_tms, 
                                    parallel=self.arc_engine.parallel, 
                                    parallel_mode=self.arc_engine.parallel_mode)

        for var_name in problem.variables:
            self.arc_engine.add_variable(var_name, problem.domains[var_name])
        for var1, var2, relation in problem.constraints:
            cid = f"{var1}_{var2}_{len(self.arc_engine.constraints)}"
            self.arc_engine.add_constraint(var1, var2, relation, cid=cid)


    def solve(self, problem: CSPProblem, return_all: bool = False, max_solutions: Optional[int] = None) -> List[CSPSolution]:
        """
        Resuelve un problema CSP utilizando backtracking con propagación de consistencia de arcos.

        :param problem: El problema CSP a resolver.
        :param return_all: Si es True, devuelve todas las soluciones; de lo contrario, devuelve la primera.
        :param max_solutions: Número máximo de soluciones a encontrar.
        :return: Una lista de soluciones encontradas.
        """
        self.problem = problem
        self._setup_arc_engine(problem)

        solutions = []
        initial_assignment = {var: None for var in problem.variables}
        
        # Realizar una primera pasada de consistencia de arcos para reducir dominios iniciales
        if not self.arc_engine.enforce_arc_consistency():
            return CSPSolverResult([], 0) # No hay soluciones si el problema es inconsistente desde el principio

        self.nodes_explored = 0
        self.backtracks = 0
        self.constraints_checked = 0
        
        # Iniciar el backtracking con el ArcEngine principal
        self._backtrack(initial_assignment, solutions, return_all, max_solutions)
        
        # Limpiar el TMS después de la búsqueda (si se usó)
        if self.arc_engine.tms:
            self.arc_engine.tms.clear()

        return CSPSolverResult(solutions, self.nodes_explored)



    def _backtrack(self, assignment: Dict[str, Any], solutions: List[CSPSolution], 
                    return_all: bool, max_solutions: Optional[int]):
        self.nodes_explored += 1
        """
        Algoritmo de backtracking recursivo.
        """
        if max_solutions is not None and len(solutions) >= max_solutions:
            logger.debug(f"Max solutions reached ({len(solutions)}/{max_solutions}). Returning.")
            return

        # Si todas las variables están asignadas, hemos encontrado una solución
        if all(assignment[var] is not None for var in self.problem.variables):
            solution = CSPSolution(assignment.copy())
            solutions.append(solution)
            logger.info(f"✅ Solución encontrada: {solution.assignment}")
            return

        # Seleccionar la próxima variable no asignada (usando MRV si es posible)
        unassigned_var = self._select_unassigned_variable(assignment)
        if unassigned_var is None:
            logger.warning("No unassigned variable found, but not all variables are assigned. This should not happen.")
            return
        logger.debug(f"Seleccionando variable no asignada: {unassigned_var}")

        # Guardar el estado actual de los dominios del ArcEngine antes de probar valores para unassigned_var
        # Esto es crucial para restaurar el estado si un valor no lleva a una solución
        saved_domains = {var_name: domain_obj.__class__(list(domain_obj.get_values())) 
                         for var_name, domain_obj in self.arc_engine.variables.items()}

        # Obtener los valores del dominio de la variable seleccionada del ArcEngine actual
        domain_values = list(self.arc_engine.variables[unassigned_var].get_values())
        logger.debug(f"Dominio de {unassigned_var}: {domain_values}")

        for value in domain_values:
            logger.debug(f"Intentando {unassigned_var}={value}")
            new_assignment = assignment.copy()
            new_assignment[unassigned_var] = value

            # Registrar la decisión en el TMS (si está habilitado)
            if self.arc_engine.tms:
                self.arc_engine.tms.record_decision(unassigned_var, value)

            # Reducir el dominio de la variable asignada a solo el valor elegido en el ArcEngine actual
            original_domain_values_of_unassigned_var = list(self.arc_engine.variables[unassigned_var].get_values())
            for val_to_remove in original_domain_values_of_unassigned_var:
                if val_to_remove != value:
                    self.arc_engine.variables[unassigned_var].remove(val_to_remove)
                    # Registrar la eliminación en el TMS si está habilitado
                    if self.arc_engine.tms:
                        self.arc_engine.tms.record_removal(
                            variable=unassigned_var,
                            value=val_to_remove,
                            constraint_id=f"ASSIGN_{unassigned_var}", # Usar un ID de restricción especial para asignaciones
                            supporting_values={unassigned_var: [value]} # Simplificado
                        )

            logger.debug(f"Antes de enforce_arc_consistency para {unassigned_var}={value}. Dominios: {[f'{v}:{list(d.get_values())}' for v, d in self.arc_engine.variables.items()]}")
            # Propagar la consistencia de arcos en el ArcEngine principal
            if self.arc_engine.enforce_arc_consistency():
                logger.debug(f"Después de enforce_arc_consistency para {unassigned_var}={value}. Consistente. Dominios: {[f'{v}:{list(d.get_values())}' for v, d in self.arc_engine.variables.items()]}")
                # Si es consistente, continuar con el backtracking
                self._backtrack(new_assignment, solutions, return_all, max_solutions)
                if not return_all and len(solutions) > 0:
                    logger.debug(f"Primera solución encontrada, retornando (return_all={return_all}).")
                    return # Encontró una solución y no se piden todas
            else:
                logger.debug(f"Después de enforce_arc_consistency para {unassigned_var}={value}. Inconsistente. Backtracking.")
                self.backtracks += 1
            
            # Retroceder: restaurar el estado de los dominios del ArcEngine al estado guardado
            for var_name, domain_obj in saved_domains.items():
                self.arc_engine.variables[var_name] = domain_obj.__class__(list(domain_obj.get_values()))
            
            # Limpiar las justificaciones de la decisión actual del TMS si está habilitado
            if self.arc_engine.tms:
                self.arc_engine.tms.backtrack_to_decision(self.arc_engine.tms.get_current_decision_level() - 1)

            if max_solutions is not None and len(solutions) >= max_solutions:
                logger.debug(f"Max solutions reached ({len(solutions)}/{max_solutions}). Returning.")
                return

    def _select_unassigned_variable(self, assignment: Dict[str, Any]) -> Optional[str]:
        """
        Selecciona la próxima variable no asignada utilizando la heurística MRV (Minimum Remaining Values).
        """
        min_domain_size = float('inf')
        selected_var = None

        for var in self.problem.variables:
            if assignment[var] is None:
                # Usar el tamaño del dominio actual del ArcEngine
                current_domain_size = len(self.arc_engine.variables[var])
                if current_domain_size < min_domain_size:
                    min_domain_size = current_domain_size
                    selected_var = var
        return selected_var


class CSPSolverResult:
    """
    Resultado de la ejecución del CSPSolver.
    """
    def __init__(self, solutions: List[CSPSolution], nodes_explored: int, backtracks: int = 0, constraints_checked: int = 0):
        self.solutions = solutions
        self.nodes_explored = nodes_explored
        self.backtracks = backtracks
        self.constraints_checked = constraints_checked

    def __repr__(self):
        return f"CSPSolverResult(solutions={len(self.solutions)}, nodes_explored={self.nodes_explored})"


# La función de conveniencia solve_csp ahora usa el CSPSolver con TMS por defecto
def solve_csp(problem: CSPProblem, return_all: bool = False, max_solutions: Optional[int] = None) -> CSPSolverResult:
    """
    Función de conveniencia para resolver un CSP.
    """
    solver = CSPSolver(use_tms=True) # Por defecto, usar TMS
    return solver.solve(problem, return_all, max_solutions)


# Se añade un método solve a ArcEngine que delega en CSPSolver
def _solve_wrapper(self, problem: CSPProblem, return_all: bool = False, max_solutions: Optional[int] = None) -> CSPSolverResult:
    solver = CSPSolver(use_tms=self.use_tms, parallel=self.parallel, parallel_mode=self.parallel_mode)
    return solver.solve(problem, return_all, max_solutions)

ArcEngine.solve = _solve_wrapper

