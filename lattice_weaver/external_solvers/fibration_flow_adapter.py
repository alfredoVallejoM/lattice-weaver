from ..fibration.fibration_search_solver import FibrationSearchSolver
from ..fibration.constraint_hierarchy import ConstraintHierarchy
from typing import Dict, List, Any

class FibrationFlowAdapter:
    """
    Adaptador para el solver de Flujo de Fibración.
    """
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], hierarchy: ConstraintHierarchy, max_iterations: int = 10000, max_backtracks: int = 50000):
        self.variables = variables
        self.domains = domains
        self.hierarchy = hierarchy

    def solve(self, time_limit_seconds: int = 60) -> List[Dict[str, Any]]:
        """
        Resuelve el problema usando el Flujo de Fibración.
        """
        flow = FibrationSearchSolver(self.variables, self.domains, self.hierarchy, max_iterations=max_iterations, max_backtracks=max_backtracks)
        # El Flujo de Fibración, tal como está implementado, no tiene un mecanismo de tiempo límite.
        # Se ejecutará hasta que encuentre una solución o termine.
        solution = flow.solve(time_limit_seconds=time_limit_seconds)
        return [solution] if solution else []

