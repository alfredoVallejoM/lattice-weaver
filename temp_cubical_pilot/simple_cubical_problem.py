from typing import Dict, Any, Callable

# Definición de un problema cúbico simple para el módulo piloto
# Representa la necesidad de que dos 'puntos' (variables) sean iguales si están en el mismo 'intervalo'

def cubical_equality_predicate(assignment: Dict[str, Any]) -> bool:
    """Predicado: point1 == point2 si interval == 'i0'."""
    point1 = assignment.get('point1')
    point2 = assignment.get('point2')
    interval = assignment.get('interval')

    if point1 is None or point2 is None or interval is None:
        return True # No se puede evaluar completamente aún

    if interval == 'i0':
        return point1 == point2
    return True # Si no es 'i0', la igualdad no es una restricción

def cubical_type_predicate(assignment: Dict[str, Any]) -> bool:
    """Predicado: point1 y point2 deben ser booleanos (0 o 1)."""
    point1 = assignment.get('point1')
    point2 = assignment.get('point2')
    
    if point1 is None or point2 is None:
        return True
        
    return point1 in [0, 1] and point2 in [0, 1]


def cubical_problem_definition() -> Dict[str, Any]:
    """Define un problema cúbico simple.

    Variables: point1, point2 (valores 0 o 1), interval (valores 'i0', 'i1').
    Restricciones: 
    - point1 y point2 deben ser 0 o 1 (HARD)
    - Si interval es 'i0', entonces point1 debe ser igual a point2 (HARD)
    - Preferencia: Minimizar la diferencia absoluta entre point1 y point2 (SOFT)
    """
    return {
        'variables': {
            'point1': [0, 1],
            'point2': [0, 1],
            'interval': ['i0', 'i1']
        },
        'hard_rules': [
            (['point1', 'point2'], cubical_type_predicate),
            (['point1', 'point2', 'interval'], cubical_equality_predicate)
        ],
        'soft_preferences': [
            (['point1', 'point2'], lambda assign: abs(assign['point1'] - assign['point2']), 0.5) # (variables, predicate, weight)
        ]
    }

# Ejemplo de uso (para testing interno)
if __name__ == "__main__":
    problem = cubical_problem_definition()
    print("Variables:", problem['variables'])
    print("Hard Rules:", [rule[1].__name__ for rule in problem['hard_rules']])
    print("Soft Preferences:", [pref[1].__name__ for pref in problem['soft_preferences']])

    # Simulación de una asignación
    test_assignment = {'point1': 0, 'point2': 0, 'interval': 'i0'}
    print(f"\nEvaluando asignación {test_assignment}:")
    for vars, pred in problem['hard_rules']:
        print(f"  Hard rule {pred.__name__}: {pred({k: test_assignment[k] for k in vars})}")
    for vars, pred, weight in problem['soft_preferences']:
        print(f"  Soft preference {pred.__name__}: {pred({k: test_assignment[k] for k in vars})} (weight: {weight})")

    test_assignment_violated = {'point1': 0, 'point2': 1, 'interval': 'i0'}
    print(f"\nEvaluando asignación violada {test_assignment_violated}:")
    for vars, pred in problem['hard_rules']:
        print(f"  Hard rule {pred.__name__}: {pred({k: test_assignment_violated[k] for k in vars})}")
    for vars, pred, weight in problem['soft_preferences']:
        print(f"  Soft preference {pred.__name__}: {pred({k: test_assignment_violated[k] for k in vars})} (weight: {weight})")

