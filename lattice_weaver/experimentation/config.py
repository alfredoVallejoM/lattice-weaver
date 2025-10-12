from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ExperimentConfig:
    """
    Configuración para un experimento de benchmarking.

    Define los parámetros para generar problemas, los solvers a usar
    y las métricas a recolectar.
    """
    problem_family: str
    problem_params: Dict[str, Any]
    solvers: List[str] = field(default_factory=lambda: ["default"])
    metrics: List[str] = field(default_factory=lambda: [
        "time_taken", "nodes_visited", "backtracks", "solutions_found",
        "constraints_checked", "initial_domain_size", "final_domain_size"
    ])
    repetitions: int = 1
    output_dir: str = "./experiments_results"
    description: Optional[str] = None

@dataclass
class SolverConfig:
    """
    Configuración específica para un solver.
    """
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProblemInstanceConfig:
    """
    Configuración para una instancia específica de problema.
    """
    problem_family: str
    params: Dict[str, Any]
    instance_id: Optional[str] = None

