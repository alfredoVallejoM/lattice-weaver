from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from ..fibration.constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness

class PymooProblemAdapter(Problem):
    """
    Adaptador para `pymoo`.
    Convierte un problema definido con `ConstraintHierarchy` a un formato compatible con `pymoo`.
    Las restricciones HARD se tratan como restricciones del problema (g) y las SOFT como objetivos (f).
    """
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], hierarchy: ConstraintHierarchy):
        self.variables = variables
        self.domains = domains
        self.hierarchy = hierarchy

        # Determinar el número de objetivos (restricciones SOFT)
        self.num_objectives = sum(1 for level in ConstraintLevel for c in hierarchy.get_constraints_at_level(level) if c.hardness == Hardness.SOFT)
        if self.num_objectives == 0:
            self.num_objectives = 1 # pymoo requiere al menos un objetivo
        
        # Determinar el número de restricciones (restricciones HARD)
        self.num_constraints = sum(1 for level in ConstraintLevel for c in hierarchy.get_constraints_at_level(level) if c.hardness == Hardness.HARD)

        # Obtener los límites de las variables (asumiendo dominios enteros)
        xl = np.array([min(domains[var]) for var in variables])
        xu = np.array([max(domains[var]) for var in variables])

        super().__init__(
            n_var=len(variables),
            n_obj=self.num_objectives,
            n_constr=self.num_constraints,
            xl=xl,
            xu=xu,
            vtype=int # Asumimos variables enteras
        )

        # Mapeo de variables a índices para pymoo
        self.var_to_idx = {var: i for i, var in enumerate(variables)}
        self.idx_to_var = {i: var for i, var in enumerate(variables)}

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evalúa una solución (x) y calcula los valores de los objetivos (f) y las restricciones (g).
        `x` es un array numpy de valores de variables.
        """
        assignments: List[Dict[str, Any]] = []
        for i in range(x.shape[0]): # Para cada individuo en la población
            assignment = {self.idx_to_var[j]: x[i, j] for j in range(self.n_var)}
            assignments.append(assignment)

        obj_values = np.zeros((x.shape[0], self.n_obj))
        constr_values = np.zeros((x.shape[0], self.n_constr))

        for i, assignment in enumerate(assignments):
            # Evaluar objetivos (restricciones SOFT)
            soft_idx = 0
            for level in ConstraintLevel:
                for constraint in self.hierarchy.get_constraints_at_level(level):
                    if constraint.hardness == Hardness.SOFT:
                        violation = constraint.predicate(assignment)
                        obj_values[i, soft_idx] = violation
                        soft_idx += 1
            
            # Evaluar restricciones (restricciones HARD)
            hard_idx = 0
            for level in ConstraintLevel:
                for constraint in self.hierarchy.get_constraints_at_level(level):
                    if constraint.hardness == Hardness.HARD:
                        violation = 0 if constraint.predicate(assignment) else 1
                        constr_values[i, hard_idx] = violation
                        hard_idx += 1
        
        out["F"] = obj_values
        out["G"] = constr_values

class PymooAdapter:
    """
    Clase principal para interactuar con pymoo.
    """
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], hierarchy: ConstraintHierarchy):
        self.variables = variables
        self.domains = domains
        self.hierarchy = hierarchy
        self.pymoo_problem = PymooProblemAdapter(variables, domains, hierarchy)
        self.res = None

    def solve(self, algorithm_name: str = "nsga2", pop_size: int = 100, n_evals: int = 10000, time_limit_seconds: int = 60) -> Optional[List[Dict[str, Any]]]:
        """
        Resuelve el problema utilizando un algoritmo de pymoo y retorna las soluciones en el frente de Pareto.
        """
        if algorithm_name == "nsga2":
            algorithm = NSGA2(pop_size=pop_size)
        else:
            raise ValueError(f"Algoritmo {algorithm_name} no soportado.")

        self.res = minimize(self.pymoo_problem, algorithm, termination=("time", str(time_limit_seconds)), verbose=False, seed=1)

        if self.res.X is not None:
            solutions: List[Dict[str, Any]] = []
            for x_sol in self.res.X:
                assignment = {self.pymoo_problem.idx_to_var[j]: x_sol[j] for j in range(self.pymoo_problem.n_var)}
                solutions.append(assignment)
            return solutions
        return None

