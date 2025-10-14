from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict
import itertools

from .constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness, Constraint
from .multiscale_compiler_api import MultiscaleCompilerAPI

class SimpleMultiscaleCompiler(MultiscaleCompilerAPI):
    """
    Un compilador multiescala simplificado que realiza una forma básica de coarse-graining
    agrupando variables fuertemente conectadas por restricciones locales HARD.
    
    Este es un prototipo para demostrar la interfaz y la lógica de compilación/descompilación.
    """

    def compile_problem(self, 
                        original_hierarchy: ConstraintHierarchy,
                        original_variables_domains: Dict[str, List[Any]]) -> Tuple[ConstraintHierarchy, Dict[str, List[Any]], Dict[str, Any]]:
        print("Iniciando compilación multiescala (prototipo simple)...")
        
        optimized_hierarchy = ConstraintHierarchy()
        optimized_variables_domains = {}
        compilation_metadata = {
            "variable_mapping": {},
            "group_definitions": {},
            "original_variables_domains": original_variables_domains
        }

        # Paso 1: Identificar grupos de variables fuertemente conectadas
        # Para este prototipo, agruparemos variables que comparten restricciones HARD LOCAL
        # Esto es una simplificación; un enfoque real usaría algoritmos de particionamiento.
        
        # Construir un grafo de conectividad basado en restricciones HARD LOCAL
        connected_components = self._find_connected_components(original_hierarchy, original_variables_domains)

        # Mapear variables originales a variables optimizadas (grupos o individuales)
        current_group_id = 0
        variable_to_group_map = {}
        for component in connected_components:
            if len(component) > 1: # Si es un grupo de más de una variable
                group_name = f"Group_{current_group_id}"
                compilation_metadata["group_definitions"][group_name] = list(component)
                
                # Definir el dominio de la super-variable (Group_X)
                # Esto implica encontrar todas las asignaciones consistentes dentro del subproblema del grupo
                group_vars = list(component)
                group_domains = {v: original_variables_domains[v] for v in group_vars}
                group_constraints = [c for c in original_hierarchy.get_all_constraints_flat() 
                                     if c.hardness == Hardness.HARD and all(v in group_vars for v in c.variables)]
                
                # Encontrar todas las soluciones consistentes para el subproblema del grupo
                consistent_assignments = self._find_consistent_assignments(group_vars, group_domains, group_constraints)
                
                optimized_variables_domains[group_name] = consistent_assignments
                for var in component:
                    variable_to_group_map[var] = group_name
                current_group_id += 1
            else: # Variables individuales que no forman parte de un grupo grande
                var_name = list(component)[0]
                optimized_variables_domains[var_name] = original_variables_domains[var_name]
                variable_to_group_map[var_name] = var_name
        
        compilation_metadata["variable_mapping"] = variable_to_group_map

        # Paso 2: Traducir restricciones a la jerarquía optimizada
        for level in ConstraintLevel:
            for original_constraint in original_hierarchy.get_constraints_at_level(level):
                # Identificar las variables de la restricción en el nuevo espacio (super-variables o individuales)
                new_scope_vars = sorted(list(set(variable_to_group_map[v] for v in original_constraint.variables)))
                
                # Crear un nuevo predicado para la restricción optimizada
                # Este predicado debe operar sobre las super-variables
                def optimized_predicate(assigned_vars_from_solver: Dict[str, Any]):
                    original_pred = original_constraint.predicate
                    original_vars = original_constraint.variables
                    group_defs = compilation_metadata["group_definitions"]
                    
                    # Reconstruir una asignación parcial del problema original
                    # a partir de la asignación de las super-variables
                    reconstructed_assignment = {}
                    # Las claves en `assigned_vars_from_solver` son las variables del problema compilado (super-variables o individuales)
                    for compiled_var, compiled_val in assigned_vars_from_solver.items():
                        if compiled_var in group_defs: # Es una super-variable (grupo)
                            # compiled_val es una asignación consistente del grupo
                            reconstructed_assignment.update(compiled_val)
                        else: # Es una variable individual
                            reconstructed_assignment[compiled_var] = compiled_val
                    
                    # Filtrar solo las variables que el predicado original necesita y que están en la asignación reconstruida
                    pred_args = {v: reconstructed_assignment[v] for v in original_vars if v in reconstructed_assignment}
                    
                    # Si no todas las variables originales de la restricción están presentes en `pred_args`,
                    # significa que la asignación parcial aún no es suficiente para evaluar esta restricción.
                    # En este caso, se considera satisfecha temporalmente (comportamiento de CSP parcial).
                    if len(pred_args) < len(original_vars):
                        return True, 0.0 

                    # Evaluar el predicado original con la asignación reconstruida
                    # Evaluar el predicado original con la asignación reconstruida
                    # El predicado original espera un diccionario de asignaciones.
                    # Asegurarse de que el predicado original se llama con el diccionario `pred_args`.
                    # El predicado original devuelve (bool, float) o solo bool.
                    result = original_pred(pred_args)
                    if isinstance(result, tuple) and len(result) == 2:
                        return result
                    elif isinstance(result, bool):
                        return result, 0.0 if result else 1.0
                    else:
                        # Si el predicado original devuelve un valor numérico (ej. para soft constraints)
                        try:
                            violation = float(result)
                            return (violation == 0.0, violation)
                        except (ValueError, TypeError):
                            print(f"Warning: Predicado original {original_pred.__name__} devolvió un tipo inesperado: {type(result)}")
                            return False, 1.0 # Considerar violada en caso de tipo de retorno inesperado

                # Añadir la nueva restricción a la jerarquía optimizada
                new_constraint = Constraint(
                    level=original_constraint.level,
                    variables=new_scope_vars,
                    predicate=optimized_predicate,
                    weight=original_constraint.weight,
                    hardness=original_constraint.hardness,
                    metadata=original_constraint.metadata,
                    expression=original_constraint.expression # Mantener la expresión original para referencia
                )
                optimized_hierarchy.add_constraint(new_constraint)

        print("Compilación multiescala completada.")
        return optimized_hierarchy, optimized_variables_domains, compilation_metadata

    def decompile_solution(self, 
                           optimized_solution: Dict[str, Any],
                           compilation_metadata: Dict[str, Any]) -> Dict[str, Any]:
        print("Iniciando descompilación de la solución...")
        original_solution = {}
        group_definitions = compilation_metadata["group_definitions"]
        variable_mapping = compilation_metadata["variable_mapping"]

        for opt_var, opt_val in optimized_solution.items():
            if opt_var in group_definitions: # Es una super-variable (grupo)
                # opt_val es una asignación consistente del grupo
                original_solution.update(opt_val)
            else: # Es una variable individual
                original_solution[opt_var] = opt_val
        
        print("Descompilación de la solución completada.")
        return original_solution

    def _find_connected_components(self, hierarchy: ConstraintHierarchy, variables_domains: Dict[str, List[Any]]) -> List[Set[str]]:
        """
        Encuentra componentes conectados de variables basadas en restricciones HARD LOCAL.
        """
        graph = defaultdict(set)
        all_vars = set(variables_domains.keys())

        for constraint in hierarchy.get_constraints_at_level(ConstraintLevel.LOCAL):
            if constraint.hardness == Hardness.HARD:
                for i in range(len(constraint.variables)):
                    for j in range(i + 1, len(constraint.variables)):
                        v1 = constraint.variables[i]
                        v2 = constraint.variables[j]
                        graph[v1].add(v2)
                        graph[v2].add(v1)
        
        # Incluir variables que no tienen restricciones locales HARD pero existen
        for var in all_vars:
            if var not in graph: # Si una variable no tiene restricciones locales HARD, sigue siendo un componente de 1
                graph[var] = set()

        visited = set()
        components = []

        for var in all_vars:
            if var not in visited:
                current_component = set()
                stack = [var]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        current_component.add(node)
                        for neighbor in graph[node]:
                            if neighbor not in visited:
                                stack.append(neighbor)
                components.append(current_component)
        return components

    def _find_consistent_assignments(self, 
                                     group_vars: List[str],
                                     group_domains: Dict[str, List[Any]],
                                     group_constraints: List[Constraint]) -> List[Dict[str, Any]]:
        """
        Encuentra todas las asignaciones consistentes para un subproblema definido por un grupo de variables.
        """
        consistent_assignments = []
        
        # Generar todas las combinaciones posibles de valores para las variables del grupo
        domain_values = [group_domains[var] for var in group_vars]
        for assignment_tuple in itertools.product(*domain_values):
            current_assignment = dict(zip(group_vars, assignment_tuple))
            
            # Verificar si esta asignación satisface todas las restricciones HARD del grupo
            is_consistent = True
            for constraint in group_constraints:
                # Asegurarse de que el predicado de la restricción se evalúe con un diccionario
                # que contenga solo las variables relevantes para esa restricción.
                constraint_vars_in_assignment = {v: current_assignment[v] for v in constraint.variables if v in current_assignment}
                satisfied, _ = constraint.evaluate(constraint_vars_in_assignment)
                if not satisfied:
                    is_consistent = False
                    break
            
            if is_consistent:
                consistent_assignments.append(current_assignment)
        
        return consistent_assignments

