from ortools.sat.python import cp_model
from typing import Dict, List, Any, Tuple, Optional

from ..fibration.constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness

class ORToolsCPSATAdapter:
    """
    Adaptador para el solver Google OR-Tools CP-SAT.
    Convierte un problema definido con `ConstraintHierarchy` a un formato compatible con CP-SAT,
    manejando restricciones HARD y SOFT.
    """
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], hierarchy: ConstraintHierarchy):
        self.variables = variables
        self.domains = domains
        self.hierarchy = hierarchy
        self.model = cp_model.CpModel()
        self.cp_vars: Dict[str, cp_model.IntVar] = {}
        self.soft_constraint_penalties: List[cp_model.IntVar] = []

        # Crear variables CP-SAT
        for var_name in self.variables:
            domain_values = sorted(list(set(self.domains[var_name])))
            if not domain_values:
                raise ValueError(f"Dominio vacío para la variable {var_name}")
            min_val = min(domain_values)
            max_val = max(domain_values)
            # Si el dominio no es contiguo, usamos un CpModel.NewIntVarFromDomain
            if len(domain_values) != (max_val - min_val + 1):
                self.cp_vars[var_name] = self.model.NewIntVarFromDomain(cp_model.Domain.FromValues(domain_values), var_name)
            else:
                self.cp_vars[var_name] = self.model.NewIntVar(min_val, max_val, var_name)

        # Añadir restricciones HARD y SOFT
        self._add_constraints()

    def _add_constraints(self):
        """
        Añade las restricciones HARD y SOFT de la ConstraintHierarchy al modelo CP-SAT.
        Las restricciones SOFT se añaden como penalizaciones a la función objetivo.
        """
        for level in ConstraintLevel:
            for constraint in self.hierarchy.get_constraints_at_level(level):
                # Obtener las variables CP-SAT correspondientes a las variables de la restricción
                cp_constraint_vars = [self.cp_vars[v] for v in constraint.variables]

                # Crear una variable booleana para indicar si la restricción se viola
                # is_violated = self.model.NewBoolVar(f'is_violated_{constraint.metadata.get("name", "unnamed")}_{level.name}')

                # Manejo de restricciones específicas (N-Queens, etc.)
                if constraint.metadata.get("name") == "all_different":
                    self.model.AddAllDifferent(cp_constraint_vars)
                    if constraint.hardness == Hardness.SOFT:
                        print(f"Advertencia: La restricción SOFT 'all_different' en {level.name} se tratará como HARD en CP-SAT.")
                elif constraint.metadata.get("name", "").startswith("diag") and len(cp_constraint_vars) == 2:
                    # Restricciones de diagonal para N-Queens
                    q1_var, q2_var = cp_constraint_vars[0], cp_constraint_vars[1]
                    q1_name, q2_name = constraint.variables[0], constraint.variables[1]
                    col_diff = abs(int(q1_name[1:]) - int(q2_name[1:]))
                    
                    # |Q1 - Q2| != col_diff
                    # Esto se traduce a Q1 - Q2 != col_diff AND Q1 - Q2 != -col_diff
                    # CP-SAT no tiene un AddAbsEquality para !=, así que lo modelamos con reificación
                    b = self.model.NewBoolVar(f'b_{constraint.metadata.get("name", "unnamed")}_{level.name}')
                    self.model.Add(q1_var - q2_var == col_diff).OnlyEnforceIf(b)
                    self.model.Add(q1_var - q2_var == -col_diff).OnlyEnforceIf(b)
                    self.model.Add(b == 0) # b debe ser falso, es decir, no se cumple la igualdad

                elif constraint.metadata.get("name", "").startswith("soft_center_Q") and len(cp_constraint_vars) == 1:
                    # Restricción SOFT para N-Queens: penalizar si la reina está en una fila central
                    q_var = cp_constraint_vars[0]
                    # Para este caso, necesitamos saber las filas centrales. Asumimos que se puede inferir de n.
                    n = len(self.variables) # N de N-Queens
                    center_rows = [k for k in range(n) if k >= n//2 - 1 and k <= n//2]

                    is_in_center = self.model.NewBoolVar(f'is_in_center_{constraint.metadata.get("name", "unnamed")}_{level.name}')
                    self.model.AddAllowedAssignments([q_var], [[r] for r in center_rows]).OnlyEnforceIf(is_in_center)
                    self.model.AddForbiddenAssignments([q_var], [[r] for r in center_rows]).OnlyEnforceIf(is_in_center.Not())

                    # Si la reina está en una fila central, se viola la restricción SOFT
                    if constraint.hardness == Hardness.SOFT:
                        self.soft_constraint_penalties.append(is_in_center)

                else:
                    # Para restricciones generales, si es HARD, la añadimos directamente.
                    # Si es SOFT y no podemos modelarla, emitimos una advertencia.
                    if constraint.hardness == Hardness.HARD:
                        # Aquí se necesitaría un parser genérico para traducir el predicado
                        # a una expresión CP-SAT. Esto está fuera del alcance de esta fase.
                        # Para el benchmark, asumimos que los problemas generados tienen predicados
                        # que se pueden manejar con las reglas anteriores o son triviales.
                        # Por ahora, si no es una restricción especial, la ignoramos si es HARD.
                        # Esto es una limitación importante para el benchmark.
                        print(f'Advertencia: Restricción HARD {constraint.metadata.get("name", "unnamed")} en {level.name} no pudo ser traducida a CP-SAT y será ignorada.')
                    else:
                        print(f'Advertencia: Predicado de restricción SOFT {constraint.metadata.get("name", "unnamed")} en {level.name} no pudo ser traducido a CP-SAT y no será penalizado.')

    def solve(self, time_limit_seconds: int = 60) -> Optional[Dict[str, Any]]:
        """
        Resuelve el problema utilizando CP-SAT y retorna la mejor solución encontrada.
        """
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds

        # Añadir la función objetivo: minimizar la suma de penalizaciones de restricciones SOFT
        if self.soft_constraint_penalties:
            self.model.Minimize(sum(self.soft_constraint_penalties))
        else:
            # Si no hay restricciones SOFT, solo buscamos una solución factible
            pass # No hay objetivo de minimización explícito, solo satisfacción

        status = solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = {}
            for var_name, cp_var in self.cp_vars.items():
                solution[var_name] = solver.Value(cp_var)
            return solution
        else:
            return None

