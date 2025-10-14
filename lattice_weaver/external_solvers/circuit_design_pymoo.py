from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Importar la definición del problema de diseño de circuitos para reutilizar las restricciones
from benchmarks.circuit_design_problem import create_circuit_design_problem
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, ConstraintType

class CircuitDesignPymooProblem(Problem):
    def __init__(self, n_gates: int, n_chips: int, hierarchy: ConstraintHierarchy):
        self.n_gates = n_gates
        self.n_chips = n_chips
        self.hierarchy = hierarchy

        # Variables: cada compuerta se asigna a un chip (0 a n_chips-1)
        # n_var = número de compuertas
        n_var = n_gates
        
        # Dominios: cada compuerta puede ir a cualquier chip
        xl = np.zeros(n_var, dtype=int)
        xu = np.full(n_var, n_chips - 1, dtype=int)

        # Objetivos: minimizar costo, minimizar latencia, minimizar consumo de energía
        # Asumimos que estas son las restricciones SOFT del problema original
        # Para este ejemplo, simplificaremos a un solo objetivo: la energía total de la jerarquía
        # En un caso real de pymoo, tendríamos múltiples objetivos aquí.
        # Para el benchmark, usaremos la energía total como un objetivo.
        
        # Contar restricciones HARD para pymoo
        num_hard_constraints = 0
        for level in ConstraintLevel:
            for constraint in hierarchy.get_constraints_by_level(level):
                if constraint.constraint_type == ConstraintType.HARD:
                    num_hard_constraints += 1

        super().__init__(
            n_var=n_var,
            n_obj=1, # Un solo objetivo para este benchmark: energía total
            n_constr=num_hard_constraints, # Restricciones HARD
            xl=xl,
            xu=xu,
            vtype=int
        )

        # Mapeo de variables de pymoo (índices) a nombres de variables del problema
        self.idx_to_var_name = {i: f"gate_{i}" for i in range(n_gates)}

    def _evaluate(self, x, out, *args, **kwargs):
        # x es la población de soluciones, cada fila es un individuo
        # x.shape = (pop_size, n_var)

        f = np.zeros((x.shape[0], self.n_obj))
        g = np.zeros((x.shape[0], self.n_constr))

        for i in range(x.shape[0]): # Para cada individuo (solución)
            assignment = {self.idx_to_var_name[j]: x[i, j] for j in range(self.n_var)}
            
            # Calcular energía total (objetivo)
            # Esto requiere una instancia de EnergyLandscape, que no está directamente aquí.
            # Para simplificar, evaluaremos las restricciones SOFT y sumaremos sus violaciones.
            total_soft_violation = 0.0
            for level in ConstraintLevel:
                for constraint in self.hierarchy.get_constraints_by_level(level):
                    if constraint.constraint_type == ConstraintType.SOFT:
                        # Asumimos que el predicado retorna la violación
                        total_soft_violation += constraint.predicate(assignment)
            f[i, 0] = total_soft_violation

            # Evaluar restricciones HARD (restricciones del problema)
            hard_constr_idx = 0
            for level in ConstraintLevel:
                for constraint in self.hierarchy.get_constraints_by_level(level):
                    if constraint.constraint_type == ConstraintType.HARD:
                        # pymoo espera g <= 0 para restricciones satisfechas
                        # Si el predicado retorna 0 para satisfecho y >0 para violado, funciona.
                        g[i, hard_constr_idx] = constraint.predicate(assignment)
                        hard_constr_idx += 1
        
        out["F"] = f
        out["G"] = g

def solve_circuit_design_pymoo(n_gates: int, n_chips: int, hierarchy: ConstraintHierarchy, pop_size: int = 100, n_evals: int = 10000) -> Optional[Dict[str, Any]]:
    """
    Resuelve el problema de diseño de circuitos usando pymoo.
    Retorna la mejor solución encontrada en el frente de Pareto (la de menor energía SOFT).
    """
    problem = CircuitDesignPymooProblem(n_gates, n_chips, hierarchy)
    
    algorithm = NSGA2(pop_size=pop_size)
    
    res = minimize(problem, algorithm, ("n_evals", n_evals), verbose=False)

    if res.X is not None and len(res.X) > 0:
        # En un problema multi-objetivo real, se analizaría el frente de Pareto.
        # Aquí, como tenemos un solo objetivo (energía SOFT), simplemente tomamos la mejor.
        best_idx = np.argmin(res.F[:, 0])
        best_x = res.X[best_idx]
        
        solution = {problem.idx_to_var_name[j]: best_x[j] for j in range(problem.n_var)}
        return solution
    return None

if __name__ == "__main__":
    # Ejemplo de uso
    n_gates = 5
    n_chips = 2
    variables, domains, hierarchy = create_circuit_design_problem(n_gates, n_chips)

    print(f"Resolviendo problema de diseño de circuitos ({n_gates} compuertas, {n_chips} chips) con pymoo...")
    solution = solve_circuit_design_pymoo(n_gates, n_chips, hierarchy)

    if solution:
        print("Solución encontrada:", solution)
        # Calcular la energía de la solución encontrada para comparar
        from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
        landscape = EnergyLandscapeOptimized(hierarchy)
        energy_result = landscape.compute_energy(solution)
        print(f"Energía total de la solución: {energy_result.total_energy:.3f}")
        print(f"Violaciones HARD: {energy_result.hard_violations:.3f}")
        print(f"Violaciones SOFT: {energy_result.soft_violations:.3f}")
    else:
        print("No se encontró solución.")

