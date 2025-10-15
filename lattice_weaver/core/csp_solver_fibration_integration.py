from typing import Dict, Any, List, Optional
from ..core.csp_problem import CSP
from ..fibration import CSPToConstraintHierarchyAdapter, ConstraintHierarchyToCSPAdapter, FibrationSearchSolver

def solve_csp_with_fibration_flow(csp: CSP, solver: Optional[FibrationSearchSolver] = None) -> Optional[Dict[str, Any]]:
    """
    Resuelve un problema CSP utilizando Fibration Flow como motor de resolución.

    Este método convierte el CSP a una ConstraintHierarchy, lo resuelve con Fibration Flow,
    y luego convierte la solución de vuelta al formato CSP.

    Args:
        csp (CSP): El problema CSP a resolver.
        solver (Optional[FibrationSearchSolver]): Una instancia del solver de Fibration Flow.
                                                 Si es None, se usará una instancia por defecto.

    Returns:
        Optional[Dict[str, Any]]: Una asignación de variables si se encuentra una solución,
                                  o None si el CSP no es satisfacible.
    """
    adapter_to_hierarchy = CSPToConstraintHierarchyAdapter()
    hierarchy, fibration_domains, compilation_metadata = adapter_to_hierarchy.convert_csp_to_hierarchy(csp)

    if solver is None:
        # Placeholder para un solver de Fibration Flow. En fases posteriores, se usará el solver refactorizado.
        # Por ahora, se puede usar un solver básico o mock para la integración.
        # Para la integración actual, simplemente devolveremos None si no se proporciona un solver.
        # La idea es que aquí se instanciaría y se llamaría al solver real.
        # solver = FibrationSearchSolver(hierarchy, fibration_domains)
        # fibration_solution = solver.solve()
        print("Advertencia: No se proporcionó un solver de Fibration Flow. Devolviendo None.")
        return None

    # Asumiendo que el solver de Fibration Flow tiene un método `solve` que devuelve una solución
    # o None si no se encuentra una solución.
    fibration_solution = solver.solve(hierarchy, fibration_domains)

    if fibration_solution:
        adapter_to_csp = ConstraintHierarchyToCSPAdapter()
        csp_solution = adapter_to_csp.convert_csp_solution_to_hierarchy_solution(fibration_solution, compilation_metadata)
        return csp_solution
    else:
        return None

