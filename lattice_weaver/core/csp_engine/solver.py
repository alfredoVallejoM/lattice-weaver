from typing import Dict, Any, List, Optional, Tuple, Callable
from lattice_weaver.core.csp_engine.tracing import ExecutionTracer, TraceEvent
from dataclasses import dataclass, field
import time
import itertools
import warnings

from ..csp_problem import CSP, Constraint
from .strategies import VariableSelector, ValueOrderer
from .strategies import FirstUnassignedSelector, NaturalOrderer

try:
    from ...arc_engine.core import ArcEngine
    from ...arc_engine.domains import create_optimal_domain # Necesario para actualizar dominios
    ARC_ENGINE_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"ArcEngine o sus dependencias no disponibles: {e}. Las funcionalidades de ArcEngine no estarán activas.", ImportWarning)
    ARC_ENGINE_AVAILABLE = False

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
    
    Puede usar opcionalmente ArcEngine para propagación AC-3.1 optimizada.
    """

    def __init__(self, 
                 csp: CSP, 
                 tracer: Optional[ExecutionTracer] = None,
                 variable_selector: Optional[VariableSelector] = None,
                 value_orderer: Optional[ValueOrderer] = None,
                 use_arc_engine: bool = False,  # NUEVO parámetro
                 parallel: bool = False):        # NUEVO parámetro
        """
        Inicializa el CSPSolver.
        
        Args:
            csp: El problema CSP a resolver
            tracer: Tracer opcional para debugging/análisis
            variable_selector: Estrategia para seleccionar variables (default: FirstUnassignedSelector)
            value_orderer: Estrategia para ordenar valores (default: NaturalOrderer)
            use_arc_engine: Si True, usa ArcEngine para AC-3.1 optimizado
            parallel: Si True y use_arc_engine=True, habilita paralelización
        """
        self.csp = csp
        self.assignment: Dict[str, Any] = {}
        self.stats = CSPSolutionStats()
        self.tracer = tracer
        
        # Estrategias modulares (usar defaults si no se especifican)
        self.variable_selector = variable_selector or FirstUnassignedSelector()
        self.value_orderer = value_orderer or NaturalOrderer()
        
        # NUEVO: Configurar ArcEngine si está disponible y solicitado
        self.arc_engine = None
        if use_arc_engine:
            if not ARC_ENGINE_AVAILABLE:
                warnings.warn(
                    "ArcEngine no disponible. Usando AC-3 básico.",
                    RuntimeWarning
                )
            else:
                self.arc_engine = ArcEngine(parallel=parallel, use_tms=False)
                self._setup_arc_engine()
        
        if self.tracer and self.tracer.enabled:
            pass # No hay un método generico record_event. Se puede registrar un evento de inicialización si es necesario, o simplemente omitirlo.

    def _setup_arc_engine(self):
        """
        Configura ArcEngine con el CSP actual.
        Convierte la representación CSP a la API incremental de ArcEngine.
        """
        if self.arc_engine is None:
            return
        
        # Añadir variables
        for var in self.csp.variables:
            self.arc_engine.add_variable(var, self.csp.domains[var])
        
        # Registrar y añadir restricciones
        for idx, constraint in enumerate(self.csp.constraints):
            if len(constraint.scope) == 2:
                var1, var2 = list(constraint.scope)
                
                # ArcEngine requiere un 'relation_name' para la función y un 'cid' para la instancia de la restricción.
                # Ambos deben ser únicos en sus respectivos contextos dentro de ArcEngine.

                # Generar un nombre único para la función de relación si no se proporciona uno en la restricción original
                relation_func_name_base = constraint.name if constraint.name else f"rel_func_{idx}_{var1}_{var2}"
                
                # Asegurarse de que el nombre de la función de relación sea único para el registro de ArcEngine
                unique_relation_func_name = relation_func_name_base
                rel_func_counter = 0
                while unique_relation_func_name in self.arc_engine._relation_registry:
                    unique_relation_func_name = f"{relation_func_name_base}_{rel_func_counter}"
                    rel_func_counter += 1

                # Wrapper que adapta la signatura de la función de relación del CSP a la de ArcEngine
                def make_relation_wrapper(original_relation):
                    def wrapper(val1, val2, metadata):
                        return original_relation(val1, val2)
                    return wrapper
                
                # Solo registrar la relación si no ha sido registrada ya (por su nombre único)
                if unique_relation_func_name not in self.arc_engine._relation_registry:
                    self.arc_engine.register_relation(
                        unique_relation_func_name, 
                        make_relation_wrapper(constraint.relation)
                    )

                # Generar un ID único para la instancia de la restricción (cid)
                # Usamos el nombre de la restricción del CSP si existe, o un ID basado en el índice y un hash
                constraint_instance_id_base = constraint.name if constraint.name else f"constraint_instance_{idx}_{var1}_{var2}_{hash(constraint.relation)}"
                
                # Asegurarse de que el cid sea único para la instancia de restricción en ArcEngine
                unique_cid = constraint_instance_id_base
                cid_counter = 0
                while unique_cid in self.arc_engine.constraints:
                    unique_cid = f"{constraint_instance_id_base}_{cid_counter}"
                    cid_counter += 1

                self.arc_engine.add_constraint(var1, var2, unique_relation_func_name, cid=unique_cid)
            elif len(constraint.scope) == 1:
                # ArcEngine no maneja restricciones unarias directamente en add_constraint
                # Se asume que el dominio inicial ya las satisface o se manejarán en _is_consistent
                pass

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
            # Si no se buscan todas las soluciones, o si ya se alcanzó el límite de max_solutions, terminar.
            # De lo contrario, continuar explorando para encontrar más soluciones.
            if not all_solutions or len(self.stats.solutions) >= max_solutions:
                return True  # Indica que se debe detener la rama actual de búsqueda
            else:
                return False # Indica que se debe continuar explorando en esta rama

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
                        return True # Propagar la señal de detención si se encontró una solución o se alcanzó el límite
                else:
                    self.stats.backtracks += 1
                    if self.tracer and self.tracer.enabled:
                        self.tracer.record_backtrack(variable=var, value=value, depth=len(self.assignment))

            del self.assignment[var] # Deshacer asignación
            if self.tracer and self.tracer.enabled:
                pass # No hay un método especifico para unassign.

        return False

    def enforce_arc_consistency(self) -> bool:
        """
        Implementa el algoritmo AC-3 (o AC-3.1 si ArcEngine está activo).
        Retorna True si el CSP es consistente, False si se detecta inconsistencia.
        """
        if self.arc_engine is not None:
            # Usar AC-3.1 optimizado de ArcEngine
            # Necesitamos actualizar los dominios del ArcEngine con el estado actual del CSP
            # y luego obtener los dominios reducidos de vuelta.
            # Esto es una simplificación, en un caso real, ArcEngine debería operar sobre
            # una copia de los dominios o tener un mecanismo de rollback.
            for var in self.csp.variables:
                # ArcEngine no tiene set_domain. La forma correcta de actualizar su dominio
                # es usar el método `intersect` si el dominio del CSP se ha reducido.
                # Si el dominio del CSP se ha expandido o es completamente diferente, la forma
                # más segura (aunque potencialmente ineficiente) es recrear la variable en ArcEngine.
                # Para esta integración inicial, asumimos que los dominios solo se reducen.
                current_csp_domain_list = list(self.csp.domains[var])
                self.arc_engine.variables[var].intersect(current_csp_domain_list)

            is_consistent = self.arc_engine.enforce_arc_consistency()
            if is_consistent:
                # Actualizar los dominios del CSP con los dominios reducidos por ArcEngine
                for var in self.csp.variables:
                    self.csp.domains[var] = self.arc_engine.variables[var].get_values()
            return is_consistent
        else:
            # Usar AC-3 básico (implementación actual)
            return self._enforce_ac3_basic()

    def _enforce_ac3_basic(self) -> bool:
        """
        Implementación AC-3 básica (código actual).
        Se mantiene como fallback.
        """
        queue = []
        for constraint in self.csp.constraints:
            if len(constraint.scope) == 2:
                var_i, var_j = list(constraint.scope)
                queue.append((var_i, var_j, constraint))
                queue.append((var_j, var_i, constraint))

        while queue:
            var_i, var_j, constraint = queue.pop(0)
            if self._revise(var_i, var_j, constraint):
                if not self.csp.domains[var_i]:
                    return False
                for neighbor_constraint in self.csp.constraints:
                    if len(neighbor_constraint.scope) == 2:
                        n_var1, n_var2 = list(neighbor_constraint.scope)
                        if n_var2 == var_i and n_var1 != var_j:
                            queue.append((n_var1, n_var2, neighbor_constraint))
                        elif n_var1 == var_i and n_var2 != var_j:
                            queue.append((n_var2, n_var1, neighbor_constraint))
        return True

    def _revise(self, var_i: str, var_j: str, constraint: Constraint) -> bool:
        revised = False
        new_domain_i = []
        for x in self.csp.domains[var_i]:
            # Check if there is any value y in domain of var_j that satisfies the constraint
            satisfies = False
            for y in self.csp.domains[var_j]:
                if (var_i == list(constraint.scope)[0] and constraint.relation(x, y)) or \
                   (var_i == list(constraint.scope)[1] and constraint.relation(y, x)):
                    satisfies = True
                    break
            if satisfies:
                new_domain_i.append(x)
            else:
                revised = True
        self.csp.domains[var_i] = new_domain_i
        return revised

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
                        if (var == list(constraint.scope)[0] and constraint.relation(value, other_value)) or\
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
        # Se hace una copia profunda para que los cambios en los dominios no afecten al CSP original
        initial_domains = {var: list(self.csp.domains[var]) for var in self.csp.variables}
        
        # Aplicar consistencia de arcos inicial si ArcEngine está activo
        if self.arc_engine is not None:
            # Necesitamos una copia del CSP para el ArcEngine para no modificar el original
            # antes de la llamada a solve, y para que el ArcEngine pueda operar sobre él.
            # Esto es una simplificación, idealmente ArcEngine debería trabajar con su propia
            # representación interna de dominios y variables.
            # Para la llamada inicial a enforce_arc_consistency, necesitamos que ArcEngine
            # opere sobre los dominios iniciales del CSP. Ya hemos configurado ArcEngine
            # con estos dominios en _setup_arc_engine().
            # No necesitamos un CSP temporal ni un solver temporal aquí.
            # Simplemente llamamos a enforce_arc_consistency en el self.arc_engine.
            if not self.arc_engine.enforce_arc_consistency():
                self.stats.time_elapsed = time.time() - start_time
                return self.stats # No hay soluciones si es inconsistente desde el principio
            
            # Actualizar los dominios iniciales con los reducidos por ArcEngine
            initial_domains = {var: list(self.arc_engine.variables[var].get_values()) for var in self.csp.variables}

        # Ejecutar backtracking
        self._backtrack(initial_domains, all_solutions, max_solutions)
        
        # Registrar tiempo de ejecución
        self.stats.time_elapsed = time.time() - start_time
        
        if self.tracer and self.tracer.enabled:
            pass # No hay un método generico para search_completed.
        
        return self.stats

