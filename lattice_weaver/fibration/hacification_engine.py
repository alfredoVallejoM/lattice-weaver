from typing import Dict, List, Tuple, Optional, Any, Set
from collections import deque
from dataclasses import dataclass

from .constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness
from .energy_landscape_optimized import EnergyLandscapeOptimized, EnergyComponents

@dataclass
class HacificationResult:
    is_coherent: bool
    level_results: Dict[ConstraintLevel, bool]
    energy: EnergyComponents
    violated_constraints: List[str]
    has_hard_violation: bool

class HacificationEngine:
    def __init__(self, hierarchy: ConstraintHierarchy, landscape: EnergyLandscapeOptimized, variables_domains: Dict[str, List[Any]]):
        self.hierarchy = hierarchy
        self.landscape = landscape
        self.variables_domains = variables_domains
        self.energy_thresholds = {
            ConstraintLevel.LOCAL: 0.0,
            ConstraintLevel.PATTERN: 0.0,
            ConstraintLevel.GLOBAL: 0.1
        }

    def hacify(self, assignment: Dict[str, Any], strict: bool = True) -> HacificationResult:
        energy_components = self.landscape.compute_energy(assignment)
        level_results = {}
        all_violated_constraints = []
        has_hard_violation = False

        for level in ConstraintLevel:
            level_energy = getattr(energy_components, f"{level.name.lower()}_energy")
            threshold = self.energy_thresholds.get(level, 0.0)
            level_has_hard_violation = False

            for constraint in self.hierarchy.get_constraints_at_level(level):
                satisfied, violation = constraint.evaluate(assignment)
                if not satisfied or violation > 0:
                    constraint_name = constraint.metadata.get("name", "unnamed")
                    all_violated_constraints.append(f"{level.name}:{constraint_name}")
                    if constraint.hardness == Hardness.HARD:
                        level_has_hard_violation = True
                        has_hard_violation = True
            
            level_results[level] = not level_has_hard_violation and level_energy <= threshold

        if strict:
            is_coherent = not has_hard_violation
        else:
            is_coherent = not has_hard_violation and all(level_results.values())

        return HacificationResult(
            is_coherent=is_coherent,
            level_results=level_results,
            energy=energy_components,
            violated_constraints=list(set(all_violated_constraints)),
            has_hard_violation=has_hard_violation
        )

    def filter_coherent_extensions(self, base_assignment: Dict[str, Any], variable: str, domain: List[Any], strict: bool = True) -> List[Any]:
        # Primero, aplicar la consistencia de arco (AC-3) para podar el dominio de la variable actual
        # antes de verificar la coherencia de las extensiones.
        # Esto es una simplificación de AC-3, enfocada en la variable 'variable' y su dominio.
        # Un AC-3 completo operaría sobre todas las variables no asignadas.
        
        # Crear un dominio temporal para la variable actual
        temp_domains = {v: list(self.variables_domains[v]) for v in self.variables_domains}
        temp_domains[variable] = list(domain)

        # Inicializar la cola de arcos para AC-3. Aquí, nos centramos en los arcos que involucran a 'variable'
        # y las restricciones HARD que la afectan.
        queue = deque()
        for const in self.hierarchy.get_constraints_at_level(ConstraintLevel.LOCAL) + \
                     self.hierarchy.get_constraints_at_level(ConstraintLevel.PATTERN) + \
                     self.hierarchy.get_constraints_at_level(ConstraintLevel.GLOBAL):
                if const.hardness == Hardness.HARD and variable in const.variables:
                    for other_var in const.variables:
                        if other_var != variable and other_var not in base_assignment:
                            queue.append((variable, other_var, const))
                            queue.append((other_var, variable, const))

        # Ejecutar una versión simplificada de AC-3
        while queue:
            (var_i, var_j, constraint) = queue.popleft()
            if self._revise(var_i, var_j, constraint, temp_domains, base_assignment):
                if not temp_domains[var_i]: # Dominio vacío, no hay soluciones
                    return []
                # Si el dominio de var_i fue revisado, añadir arcos relacionados a la cola
                for const_k in self.hierarchy.get_constraints_at_level(ConstraintLevel.LOCAL) + \
                               self.hierarchy.get_constraints_at_level(ConstraintLevel.PATTERN) + \
                               self.hierarchy.get_constraints_at_level(ConstraintLevel.GLOBAL):
                        if const_k.hardness == Hardness.HARD and var_i in const_k.variables:
                            for var_k in const_k.variables:
                                if var_k != var_i and var_k not in base_assignment:
                                    queue.append((var_k, var_i, const_k)) # Añadir (var_k, var_i) a la cola

        # Después de la poda con AC-3, verificar la coherencia de las extensiones con los dominios podados
        coherent_values = []
        for value in temp_domains[variable]: # Iterar sobre el dominio podado
            temp_assignment = base_assignment.copy()
            temp_assignment[variable] = value
            h_result = self.hacify(temp_assignment, strict=strict)
            if h_result.is_coherent:
                coherent_values.append(value)
        return coherent_values

    def _revise(self, var_i: str, var_j: str, constraint: Any, domains: Dict[str, List[Any]], base_assignment: Dict[str, Any]) -> bool:
        # var_i: variable cuyo dominio estamos revisando
        # var_j: otra variable en el scope de la restricción
        # constraint: la restricción entre var_i y var_j (y posiblemente otras)
        
        revised = False
        new_domain_i = []

        for val_i in domains[var_i]:
            # Intentar encontrar un valor en el dominio de var_j que satisfaga la restricción
            # dada la asignación actual de base_assignment y val_i para var_i
            found_support = False
            
            # Crear una asignación temporal para evaluar la restricción
            temp_assignment_for_constraint = base_assignment.copy()
            temp_assignment_for_constraint[var_i] = val_i

            # Si var_j ya está asignada en base_assignment, solo hay un valor para probar
            if var_j in base_assignment:
                temp_assignment_for_constraint[var_j] = base_assignment[var_j]
                satisfied, _ = constraint.evaluate(temp_assignment_for_constraint)
                if satisfied:
                    found_support = True
            else: # var_j no está asignada, iterar sobre su dominio
                for val_j in domains[var_j]:
                    temp_assignment_for_constraint[var_j] = val_j
                    satisfied, _ = constraint.evaluate(temp_assignment_for_constraint)
                    if satisfied:
                        found_support = True
                        break # Encontramos soporte, no necesitamos buscar más en el dominio de var_j
            
            if found_support:
                new_domain_i.append(val_i)
            else:
                revised = True # val_i no tiene soporte, se elimina del dominio
        
        domains[var_i] = new_domain_i
        return revised

    def get_statistics(self) -> Dict:
        return {
            "energy_thresholds": {level.name: threshold for level, threshold in self.energy_thresholds.items()}
        }

